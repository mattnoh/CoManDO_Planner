# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
source /opt/ros/humble/setup.zsh
source ~/ros_ws/install/setup.zsh
cd ~/ros_ws
colcon build --symlink-install --packages-select comando_planner
```

ALIPDDP is compiled as a static library from the sibling directory `../ALIPDDP-main` via `add_subdirectory()`. Optional Qualisys support (`mocap4r2_msgs`) is detected at build time and guarded by `HAS_MOCAP4R2_MSGS`.

### Standalone validation test (no ROS required)

```bash
cd ~/ros_ws/build/comando_planner
./test_poly_ocp
```

Source file: `test/test_poly_ocp_standalone.cpp`.

## Running the planner

Two-step launch: start infrastructure first, then push an OCP profile.

```bash
# Terminal A — starts node in paused state
ros2 launch comando_planner planner_launch.py drone_name:=cf_1

# Terminal B — push an OCP and trigger execution
ros2 launch comando_planner ocp_launch.py ocp_type:=hover mode:=mpc \
    hover_target_x:=0.0 hover_target_y:=0.0 hover_target_z:=1.0 command_seq:=1
```

`ocp_launch.py` staggers parameter updates 150ms apart then fires `command_seq` last, ensuring the node sees a consistent config snapshot before acting. Increment `command_seq` each time you push a new profile to the same running node.

## Architecture

### Registry-Driven OCP Design

Each OCP is an `OCPDescriptor` registered in `OCPRegistry::getTable()` (`include/core/ocp_registry.hpp`). Adding a new behavior is two steps:

1. Create `include/ocp/ocp_mynew.hpp` with a `descriptor()` function returning `OCPDescriptor`.
2. Add one line to the `table` in `ocp_registry.hpp`: `{"mynew", MyNewOCP::descriptor()}`.

No changes to `planner_node.cpp`, `TrajectoryReplayer`, or logging are needed.

**Do not include `ocp_registry.hpp` from OCP headers** — it would be circular. Include `core/ocp_descriptor.hpp` instead.

### Solve-while-Replay Pattern

Solver and replay run in separate callback groups (both `MutuallyExclusive`):

- **Solver loop** fires every `n_replay` replay cycles, snapshots drone+target state, calls ALIPDDP, hands a new trajectory to `TrajectoryReplayer`.
- **Replay timer** fires at `ocp_dt` (e.g. 50 Hz), samples the active trajectory, publishes commands.

This keeps command output consistent across 100ms+ solve times.

### Key Types

**`OCPDescriptor`** (`include/core/ocp_descriptor.hpp`) — declares identity, timing, dimensions, mode flags, and callbacks:

| Field | Purpose |
|-------|---------|
| `command_mode` | `CmdFullState` (wrench) or `CmdBodyRate` (specific thrust + angular rates) |
| `drone_odom_mode` | Which topic / frame the planner reads for drone state (see below) |
| `transform_state` | Converts raw sensor state into OCP state at solve time |
| `validate_target` | Returns false → solver withheld (stale/invalid target) |
| `post_process_result` | Modify `SolverResult` after solve (e.g. mark relative plan) |
| `merge_prev_augmented` | Re-injects augmented states from previous trajectory into the new `x0` |
| `prepare_extra` | Build OCP-specific extra params passed to `create()` |

**`SolverResult`** — wraps ALIPDDP output: `state_trajectory`, `control_trajectory`, relative-plan flag, and target snapshot embedded at solve time.

**`TargetSnapshot`** (`include/core/target_snapshot.hpp`) — atomic copy of target state at solve time: position, velocity, acceleration, angular velocity, orientation.

### DroneOdomMode

Controls which topic the planner subscribes to for drone state:

| Mode | Topic | Description |
|------|-------|-------------|
| `Absolute` | `/{name}/pose` + `/{name}/odom` | Standard world-frame state |
| `AbsoluteShiftedTarget` | Same as above | `transform_state` subtracts target position at solve time |
| `BodyFrameRelative` | `/drone/body_relative_odom` | Published by `target_publisher` with `enable_body_relative_odom:=true` |
| `TargetFrameRelative` | `/drone/target_frame_odom` | Published by `target_publisher`; drone position/velocity in target frame N |

### Platform Adapters (`include/platform/`)

| Platform | Input | Output | Frame |
|----------|-------|--------|-------|
| Crazyflie | `/{name}/pose`, `/{name}/odom` | `/{name}/cmd_full_state` or `/{name}/cmd_bodyrate` | ENU |

Crazyflie angular velocity from crazyswarm2 arrives in deg/s and is converted to rad/s internally.

### target_publisher Node

Two modes (`target_mode` param): `"circle"` (fully synthetic) or `"qualisys"` (real vehicle with velocity computed from circular model). Set `enable_body_relative_odom:=true` or `enable_target_frame_odom:=true` to publish the additional relative odometry topics needed by body-frame / target-frame OCPs.

### Terminal Freeze

Solver stops when drone is within `terminal_freeze_enter_pos` and velocity below `terminal_freeze_enter_vel`. Replayer continues executing the last plan. Disabled automatically for tracking OCPs.

## OCP Catalog

| OCP key | File | Frame | State dim | Control dim | Notes |
|---------|------|-------|-----------|-------------|-------|
| `hover` | `ocp_hover.hpp` | Absolute | 13 | 4 | Fixed-point stabilization |
| `landing` | `ocp_landing.hpp` | Absolute | 13 | 4 | Vertical descent |
| `stateswitch` | `ocp_stateswitch.hpp` | AbsoluteShiftedTarget | 13 | 5 | Variable DT, 30 steps |
| `tracking_circle_target` | `ocp_tracking_circle_target.hpp` | Absolute | 13 | 4 | Receding-horizon circle via predicted trajectory |
| `tracking_bodyrate_tf_noimu` | `ocp_tracking_bodyrate_tf_noimu.hpp` | TargetFrameRelative | 11 (10+DT) | 5 | No augmented IMU states |
| `tracking_bodyrate_tf_imu` | `ocp_tracking_bodyrate_tf_imu.hpp` | TargetFrameRelative | 20 (19+DT) | 5 | Augmented Ω_N, â_N, β_N |
| `tracking_bodyrate_bf_noimu` | `ocp_tracking_bodyrate_bf_noimu.hpp` | BodyFrameRelative | 11 (10+DT) | 5 | No augmented IMU states |
| `tracking_bodyrate_bf_imu` | `ocp_tracking_bodyrate_bf_imu.hpp` | BodyFrameRelative | 20 (19+DT) | 5 | Augmented Ω_N, a_T^B, β_N |

### 13D State (hover, landing, stateswitch, tracking_circle_target)

`[px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]`

Control: `[fz_B, Mx, My, Mz]` — body-frame thrust + moments.

### 10D Physical State (bodyrate OCPs without IMU)

**Target frame (tf):** `[p_B^N (3), v_B^N (3), q_NB (4)]` — drone position/velocity/attitude in target frame N.

**Body frame (bf):** `[p_T^B (3), v_rel^B (3), q_NB (4)]` — target position in drone body frame, relative velocity in body frame.

Control: `[T (1), ω_B (3), Theta (1)]` — specific thrust [m/s²], body angular rates, time segment.

### 19D Augmented Physical State (bodyrate OCPs with IMU)

Adds `[Ω_N (3), a_T (3), β_N (3)]` to the 10D physical state at indices 10–18; DT appended at index 19.

`merge_prev_augmented` in the descriptor re-injects these from the previous solve trajectory so the solver carries forward propagated target kinematics rather than resetting from the snapshot.

## Logging

When `enable_logging:=true`, logs written to `./logs/{drone}_{ocp}_{mode}_{solver}_{timestamp}/`:
- `all_solves.csv` — per-solve metadata + full state/control trajectories
- `commanded_state.csv` — per-command output
- `actual_state.csv` — raw sensor measurements
