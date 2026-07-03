# CLAUDE.md

This file gives coding-agent guidance for the `comando_planner` package on the
**ROS 1 Noetic branch** (a git worktree of the ROS 2 repo). The README's
`ros2 launch` commands belong to the ROS 2 branch; everything below is what
actually works here.

## Branch facts

- Build system: **catkin** (`catkin build comando_planner`), not colcon.
- **Only `platform:=mavros` is implemented.** `include/platform/crazyflie.hpp`
  is a compile-only stub (State struct so `state_monitor.hpp` compiles);
  `planner_node.cpp` throws for any other platform.
- `mavros_msgs` is a **hard** dependency on this branch (CMakeLists
  `CATKIN_DEPENDS`), not optional.
- Launch files: use the `.launch` XML files. The `.py` files in `launch/`
  are ROS 2 (`launch`/`launch_ros`) and do not work under Noetic. There is
  **no `ocp_launch.launch`** — OCPs are triggered via rosparam (below).

## Build

```bash
source /opt/ros/noetic/setup.zsh
cd /home/lab/mavros_ws
catkin build comando_planner
source devel/setup.zsh
```

ALIPDDP is pulled in via `add_subdirectory(../ALIPDDP-main ...)`; it is never
built by catkin directly (root-level `CATKIN_IGNORE`).

## Common Run Commands

Planner infrastructure (starts paused, unconfigured):

```bash
roslaunch comando_planner planner_launch.launch platform:=mavros drone_name:=px4_drone
```

Trigger an OCP profile (params live under the node's private namespace
`/comando_planner/`; bump `command_seq` to trigger):

```bash
rosrun comando_planner apply_profile.sh                          # config/profile_hover_sitl.yaml
rosrun comando_planner apply_profile.sh /comando_planner my.yaml # custom
# or by hand:
rosparam set /comando_planner/ocp_type hover
rosparam set /comando_planner/mode mpc
rosparam set /comando_planner/hover_target_z 1.5
rosparam set /comando_planner/command_seq 1
```

Target publisher (needed by target-relative OCPs):

```bash
roslaunch comando_planner target_launch.launch \
  target_trajectory:=circle \
  drone_odom_topic:=/mavros/local_position/odom \
  drone_pose_topic:=/mavros/local_position/pose
```

Full PX4 SITL bring-up (PX4 + gz + mavros + planner): **`docs/MAVROS_SITL.md`**.

## Current Architecture

| Area | Files | Notes |
| --- | --- | --- |
| ROS node | `src/planner_node.cpp` | Parameters (10 Hz rosparam polling), timers, state snapshots, logging, command publication |
| ROS-free core | `include/planner_core/` | OCP descriptors, frame/command adapters, solve/replay orchestration |
| OCPs | `include/ocp/` | Concrete ALIPDDP problem definitions and descriptors |
| Platforms | `include/platform/` | MAVROS (real), Crazyflie (stub), target tracking, state monitor |
| Replay | `include/trajectory_replayer.hpp` | Thread-safe active/pending plan interpolation |

### MAVROS adapter (`include/platform/mavros.hpp`)

Hard-coded global topics (`/mavros/...`) — launch mavros **un-namespaced**.
A 10 Hz heartbeat streams setpoints continuously (warmup hold → last command),
requests OFFBOARD then ARM with ≤1 Hz retries, and drives its arm state from
`/mavros/state` truth. OFFBOARD/ARM is only auto-requested during initial
engagement — after a later disarm/mode change it logs and stands down.
ENU data in/out; mavros converts ENU→NED internally (do not add conversions).

## OCP Registry

Registry source: `include/planner_core/ocp_registry.hpp`.

To add an OCP:

1. Add `include/ocp/ocp_mynew.hpp`.
2. Implement `MyNewOCP::descriptor()` returning `OCPDescriptor`.
3. Include the header in `ocp_registry.hpp`.
4. Add one table entry: `{"mynew", MyNewOCP::descriptor()}`.

Current keys:

```text
hover
landing
stateswitch            (alias: state_switch)
stc_landing            (alias: rh_stc)
tracking_circle_target (alias: circle_target)
tracking_bodyrate_tf_noimu / tf_noimu
tracking_bodyrate_tf_imu   / tf_imu
tracking_bodyrate_bf_imu   / bf_imu
tracking_bodyrate_bf_noimu / bf_noimu
```

`stc_landing` is the CT-cSTC landing OCP ported from
`ALIPDDP-main/problem_examples/STC/quad_landing_rh_clean_stc.cpp`. It carries
a 15-dim solver state (13 physical + cumulative time + constraint
accumulator), rebases the time/accumulator rows inside `create()` on each
warm-started solve, and keeps all of the benchmark's `SZMUK_*` env-var
tunables (export before launching to retune without rebuilding).

## Runtime Contracts

| OCP family | Drone input | Target input | Command output |
| --- | --- | --- | --- |
| `hover`, `landing` | absolute MAVROS odom | none | full state |
| `stateswitch`, `stc_landing` | absolute drone state | `/target/odom`, `/target/accel` | full state reconstructed to world |
| `tracking_circle_target` | absolute drone state | `/target/odom`, `/target/predicted_accel`; open-loop only | full state reconstructed to world |
| `tracking_bodyrate_bf_*` | `/drone/body_relative_odom` | `/target/odom`, `/target/accel` | MAVROS `AttitudeTarget` |
| `tracking_bodyrate_tf_*` | `/drone/target_frame_odom` | `/target/odom`, `/target/accel` | MAVROS `AttitudeTarget` |

Note: all OCPs use Crazyflie-scale constants (mass 0.027 kg). Full-state OCPs
are fine on larger PX4 airframes (PX4 tracks the setpoints); body-rate OCPs
need retuning first (see `docs/MAVROS_SITL.md` §7).

## Validation Commands

After building (standalone, no ROS master needed):

```bash
rosrun comando_planner test_poly_ocp
rosrun comando_planner test_trajectory_replayer
rosrun comando_planner test_target_frame_warm_start
rosrun comando_planner test_planner_core
rosrun comando_planner test_stateswitch_predictor
rosrun comando_planner test_stc_landing
```

Manual MAVROS body-rate path check:

```bash
rosrun comando_planner test_bodyrate_cmd.py \
  _drone_name:=drone _thrust:=9.81 _wz:=0.52 _duration:=3.0
```

## Documentation

| File | Purpose |
| --- | --- |
| `README.md` | Quick start (ROS 2-centric; see the banner for this branch) |
| `docs/MAVROS_SITL.md` | PX4 SITL + MAVROS bring-up guide for this branch |
| `docs/COMANDO_PLANNER_INTERFACE.md` | Full ROS topics, parameters, registry, logging |
| `docs/dynamic_target_hardware_integration.md` | External target estimator/predictor contract |
| `docs/CoManDO_planner/` | Chapter-style architecture notes |
