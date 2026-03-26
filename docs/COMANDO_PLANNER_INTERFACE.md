# CoManDO Planner ROS Interface Documentation

## Overview

CoManDO Planner is a platform-agnostic MPC planner for quadrotor control. It uses a **Registry-based architecture** to support multiple Optimal Control Problem (OCP) formulations (Landing, Hover, Moving Target Tracking) using the **ALIPDDP solver**.

**Node name:** `comando_planner`
**Package:** `comando_planner`

### Architecture

The planner is decoupled from specific OCP physics via the `OCPRegistry`. This allows adding new behaviors by simply registering a new `OCPDescriptor`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          comando_planner                                │
│                                                                         │
│  StateMonitor (thread-safe state ownership)                             │
│  ├─ cf_state_ / px4_state_ ◄── /{drone}/pose, /{drone}/odom            │
│  └─ target_state_ ◄── /target/odom, /target/accel                      │
│                                                                         │
│  OCP Registry                                                           │
│  └─► [hover, landing, stateswitch, tracking_circle]                     │
│       └─► OCPDescriptor (dt, mass, transform_cb, post_process_cb)       │
│                                                                         │
│  Solver Callback Group                                                  │
│  └─► solverLoop()                                                       │
│       ├─► getCurrentState()                                             │
│       ├─► transform_state() (optional, e.g. for relative OCPs)          │
│       ├─► ALIPDDP solve                                                 │
│       └─► TrajectoryReplayer.updatePlan()                               │
│                                                                         │
│  Replay Callback Group                                                  │
│  └─► mpcReplayTick() (Timer: ocp_dt)                                    │
│       ├─► trajectory_replayer_.sample()                                 │
│       ├─► resolve relative state to absolute (if is_relative_plan)      │
│       └─► publishCommand() ──► /{drone}/cmd_full_state                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. ROS Parameters

### Launch Parameters (Infrastructure)

Defined in `launch/planner_launch.py`. These connect the planner to the hardware and environment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drone_name` | string | `"cf_1"` | Drone namespace |
| `platform` | string | `"crazyflie"` | Platform: `"crazyflie"` or `"px4"` |
| `solver` | string | `"alipddp"` | Solver backend |
| `enable_logging` | bool | `true` | Enable CSV logging to `./logs/` |
| `target_odom_topic`| string | `"/target/odom"` | External target tracking topic |
| `target_accel_topic`| string | `"/target/accel"`| External target acceleration topic |

### Runtime Parameters (Configurable via `ocp_launch.py`)

These parameters define the behavior and can be updated dynamically via ROS parameters or manually using `ros2 param set`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ocp_type` | string | `""` | OCP formulation: `"landing"`, `"hover"`, `"stateswitch"`, `"tracking_circle"` |
| `mode` | string | `"mpc"` | `"mpc"` (closed-loop) or `"open_loop"` |
| `n_replay` | int | `4` | Number of setpoints sent per solve cycle |
| `hover_target_x/y/z` | double | `0.0` | Global target position (for absolute OCPs) |
| `mass_kg` | double | Varies | Quadrotor mass (default provided by OCP) |
| `command_seq` | int | `0` | Increment to trigger a newly configured profile |

### OCP-Specific Parameters (e.g. `tracking_circle`)

When using specific OCPs like `tracking_circle`, additional parameters are required:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `circle_center_x/y/z` | `0, 0, 0` | Center of the target circle |
| `circle_R` | `0.5` | Radius of the circle [m] |
| `circle_omega` | `1.0` | Angular velocity [rad/s] |
| `circle_phi0` | `0.0` | Initial phase [rad] |

### Terminal Freeze

Stops solving when the drone is stably within the target zone to save resources.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_terminal_freeze` | bool | `true` | Toggle terminal freeze |
| `terminal_freeze_enter_pos`| double | `0.20` | Distance error to engage freeze [m] |
| `terminal_freeze_enter_vel`| double | `0.10` | Velocity error to engage freeze [m/s] |
| `terminal_freeze_require_vel`| bool | `false`| If true, velocity must also be low to freeze |

---

## 2. Inputs (Subscriptions)

### State Estimation

The planner subscribes to standard ROS2 state topics. High-rate updates are handled by a dedicated callback group.

- **Pose**: `/{drone_name}/pose` (`geometry_msgs/PoseStamped`)
- **Odometry**: `/{drone_name}/odom` (`nav_msgs/Odometry`) - **Note**: Angular rates from Crazyswarm2 are in `deg/s` and are converted to `rad/s` internally.

### Target Tracking

For relative-frame OCPs (like `stateswitch`), the planner monitors an external target:

- **Target Odometry**: `{target_odom_topic}`
- **Target Acceleration**: `{target_accel_topic}`

Target state must be fresh (age < 0.2s) for the solver to engage.

---

## 3. Outputs (Publications)

### Control Commands

- **Full State**: `/{drone_name}/cmd_full_state` (`crazyflie_interfaces/msg/FullState`)
- **Rate**: `ocp_dt` (set by the active OCP).

The command includes **Acceleration Feedforward**, computed as:
$$a_{world} = R(q) \cdot [0, 0, f_z/m]^T + [0, 0, -g]^T$$

### Visualization

- **Planned Trajectory**: `/{drone_name}/planned_trajectory` (`nav_msgs/msg/Path`) - Displays the N-step horizon in RViz.

---

## 4. OCP Registry & Formulations

The planner uses an `OCPRegistry` to manage different problem types.

| OCP Type | Formulation | State Space | Relative? | Notes |
|----------|-------------|-------------|-----------|-------|
| `"hover"` | Fixed Target | Absolute (13D) | No | Simple stabilization |
| `"landing"` | Terminal Constraint | Absolute (13D) | No | Precise vertical landing |
| `"stateswitch"`| Variable Time | Relative (14D) | **Yes** | Landing on moving targets |
| `"tracking_circle"`| TV Dynamics | Absolute (13D) | No* | Landing on circular trajectory |

*\*`tracking_circle` solves in a "baked-in" relative frame and post-processes the result to absolute coordinates before replaying.*

---

## 5. Timing & Replay Model

The planner operates on a **Solve-while-Replay** model:

1. **Solve Tick**: Triggered every `n_replay` steps. It captures a state snapshot ($x_0$), solves the OCP, and hands the plan to the `TrajectoryReplayer`.
2. **Replay Tick**: Every `ocp_dt`, the `TrajectoryReplayer` samples the current active plan based on the elapsed time since the solve started.

This architecture ensures that even if a solve takes 100ms, the drone continues to follow the last valid trajectory at high frequency (e.g., 20Hz or 50Hz).

---

## 6. Logging (`logs/`)

For every execution, a new timestamped directory is created:

- `all_solves.csv`: The full predicted trajectory (X and U) for every solve iteration. For relative OCPs, this includes the target state snapshot.
- `commanded_state.csv`: The actual commands streamed to the hardware.
- `actual_state.csv`: The measured states received during the run.

---

## 7. Integration Checklist for New OCPs

1. **Define OCP Header**: Implement stage/terminal costs and dynamics in `include/ocp/`.
2. **Add to Registry**: Register in `include/ocp_registry.hpp` with appropriate callbacks (transform, validate, post-process).
3. **Configure Launch**: Add parameters to `launch/planner_launch.py` or the configuration scripts.
