# CLAUDE.md

This file gives coding-agent guidance for the current `comando_planner`
package.

## Build

Build from the ROS workspace root:

```bash
source /opt/ros/humble/setup.zsh
cd /home/lab/projects/cf/online/ros_ws
colcon build --symlink-install --packages-select comando_planner
source install/setup.zsh
```

`CMakeLists.txt` adds ALIPDDP from the sibling package path
`../ALIPDDP-main`. Optional dependencies:

| Dependency | Build behavior |
| --- | --- |
| `mavros_msgs` | Enables MAVROS platform support and defines `HAS_MAVROS_MSGS` |
| `mocap4r2_msgs` | Enables mocap target mode and defines `HAS_MOCAP4R2_MSGS` |

The package still builds without those optional message packages, but related
runtime paths are unavailable or fall back.

## Common Run Commands

Two-step launch:

```bash
ros2 launch comando_planner planner_launch.py \
  drone_name:=cf_1 platform:=crazyflie record_bag:=false

ros2 launch comando_planner ocp_launch.py \
  ocp_type:=hover mode:=mpc n_replay:=4 \
  hover_target_x:=0.0 hover_target_y:=0.0 hover_target_z:=1.0 \
  command_seq:=1
```

Target publisher for body-frame OCPs:

```bash
ros2 run comando_planner target_publisher --ros-args \
  -p target_mode:=gazebo_circle \
  -p drone_odom_mode:=body_frame \
  -p drone_odom_topic:=/cf_1/odom \
  -p drone_pose_topic:=/cf_1/pose
```

Target publisher for target-frame OCPs:

```bash
ros2 run comando_planner target_publisher --ros-args \
  -p target_mode:=gazebo_circle \
  -p drone_odom_mode:=target_frame \
  -p drone_odom_topic:=/cf_1/odom \
  -p drone_pose_topic:=/cf_1/pose
```

## Current Architecture

The planner is split into:

| Area | Files | Notes |
| --- | --- | --- |
| ROS node | `src/planner_node.cpp` | Parameters, timers, state snapshots, logging, command publication |
| ROS-free core | `include/planner_core/` | OCP descriptors, frame/command adapters, solve/replay orchestration |
| OCPs | `include/ocp/` | Concrete ALIPDDP problem definitions and descriptors |
| Platforms | `include/platform/` | Crazyflie, MAVROS, target tracking, state monitor |
| Replay | `include/trajectory_replayer.hpp` | Thread-safe active/pending plan interpolation |

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
stateswitch
tracking_circle_target
tracking_bodyrate_tf_noimu
tracking_bodyrate_tf_imu
tracking_bodyrate_bf_imu
tracking_bodyrate_bf_noimu
```

There is no current `tracking_circle` registry key.

## Runtime Contracts

| OCP family | Drone input | Target input | Command output |
| --- | --- | --- | --- |
| `hover`, `landing` | absolute Crazyflie or MAVROS odom | none | full state |
| `stateswitch` | absolute drone state | `/target/odom`, `/target/accel` | full state reconstructed to world |
| `tracking_circle_target` | absolute drone state | `/target/odom`, `/target/predicted_accel`; open-loop only | full state reconstructed to world |
| `tracking_bodyrate_bf_*` | `/drone/body_relative_odom` | `/target/odom`, `/target/accel` | Crazyflie `cmd_hover` or MAVROS `AttitudeTarget` |
| `tracking_bodyrate_tf_*` | `/drone/target_frame_odom` | `/target/odom`, `/target/accel` | Crazyflie `cmd_hover` or MAVROS `AttitudeTarget` |

## Validation Commands

After building:

```bash
ros2 run comando_planner test_poly_ocp
ros2 run comando_planner test_trajectory_replayer
ros2 run comando_planner test_target_frame_warm_start
ros2 run comando_planner test_planner_core
ros2 run comando_planner test_stateswitch_predictor
```

Manual Crazyflie body-rate path check:

```bash
ros2 run comando_planner test_bodyrate_cmd.py --ros-args \
  -p drone_name:=cf_1 -p thrust:=9.81 -p wz:=0.52 -p duration:=3.0
```

## Documentation

Primary docs:

| File | Purpose |
| --- | --- |
| `README.md` | Quick start, OCP catalog, workflows |
| `docs/COMANDO_PLANNER_INTERFACE.md` | Full ROS topics, parameters, registry, logging |
| `docs/dynamic_target_hardware_integration.md` | External target estimator/predictor contract |
| `docs/CoManDO_planner/` | Chapter-style architecture notes |
