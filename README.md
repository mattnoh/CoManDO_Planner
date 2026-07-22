# CoManDO Planner

`comando_planner` is a ROS 2 Humble MPC planner for Crazyflie and MAVROS-based
quadrotors. It wraps ALIPDDP optimal-control problems behind a registry, keeps
state and target tracking in ROS adapters, and streams either full-state or
body-rate commands depending on the active OCP.

The current workflow is two-step:

1. Start the planner infrastructure in a paused state with `planner_launch.py`.
2. Push an OCP profile and increment `command_seq` with `ocp_launch.py`.

## Package Layout

| Path | Purpose |
| --- | --- |
| `src/planner_node.cpp` | ROS 2 node, launch/runtime parameter handling, timers, logging, platform dispatch |
| `include/planner_core/` | ROS-free planner contracts, OCP registry, command adapters, replay orchestration |
| `include/ocp/` | Registered ALIPDDP OCP formulations |
| `include/platform/` | Crazyflie, MAVROS, state monitor, and target-tracker adapters |
| `src/target_publisher.cpp` | Gazebo-style synthetic pose or mocap target estimator plus optional relative odometry |
| `launch/` | Infrastructure launch and runtime OCP trigger launch |
| `rviz/comando_debug.rviz` | RViz layout for planned paths and debug markers |
| `docs/` | Detailed interface, architecture, math, and target-integration notes |

## Requirements

| Component | Notes |
| --- | --- |
| Ubuntu 22.04 and ROS 2 Humble | Base ROS environment |
| ALIPDDP | Expected as sibling directory `../ALIPDDP-main` from this package |
| `crazyflie_interfaces` | Required; used by the Crazyflie adapter |
| `mavros_msgs` | Optional at build time; enables `platform:=mavros` when found |
| `mocap4r2_msgs` | Optional at build time; enables `target_mode:=mocap` when found |
| Eigen3 | Required by planner, OCPs, and target helpers |

Build from the workspace root:

```bash
source /opt/ros/humble/setup.zsh
cd /home/lab/projects/cf/online/ros_ws
colcon build --symlink-install --packages-select comando_planner
source install/setup.zsh
```

MAVROS and mocap support are detected by CMake. If `mavros_msgs` is missing,
the package still builds but `platform:=mavros` is unavailable. If
`mocap4r2_msgs` is missing, `target_publisher` falls back from `mocap` to
`gazebo_circle`.

## Crazyflie Workflow

Start CrazySim/Crazyswarm2 as usual, then run:

```bash
# Terminal A: planner infrastructure
ros2 launch comando_planner planner_launch.py \
  drone_name:=cf_1 platform:=crazyflie record_bag:=false

# Terminal B: optional target publisher for tracking OCPs
ros2 launch comando_planner target_launch.py \
  target_mode:=gazebo_circle planning_frame:=body_frame \
  drone_odom_topic:=/cf_1/odom drone_pose_topic:=/cf_1/pose

# Terminal C: hover at 1 m
ros2 launch comando_planner ocp_launch.py \
  ocp_type:=hover mode:=mpc n_replay:=4 \
  hover_target_x:=0.0 hover_target_y:=0.0 hover_target_z:=1.0 \
  command_seq:=1

# Terminal C: body-frame tracking after target_publisher is running
ros2 launch comando_planner ocp_launch.py \
  ocp_type:=bf_noimu mode:=mpc n_replay:=7 command_seq:=2
```

For target-frame body-rate OCPs, start the target publisher with
`planning_frame:=target_frame` and select `ocp_type:=tf_imu` or `tf_noimu`.
For `stateswitch`, use `planning_frame:=world` because the planner reads the
drone in the absolute frame and subtracts the target snapshot internally.

## MAVROS Workflow

Build with `mavros_msgs` available, start your MAVROS bridge, then run:

```bash
ros2 launch comando_planner planner_launch.py \
  drone_name:=drone platform:=mavros record_bag:=false

ros2 launch comando_planner ocp_launch.py \
  ocp_type:=tf_noimu mode:=mpc n_replay:=7 command_seq:=1
```

The MAVROS adapter subscribes to `/mavros/local_position/odom`, publishes
full-state OCPs to `/mavros/setpoint_raw/local`, and publishes body-rate OCPs to
`/mavros/setpoint_raw/attitude`. It sends two seconds of neutral setpoints before
requesting `OFFBOARD` and `ARM`. The node has a constructor parameter
`hover_thrust` with default `0.3`; the current `planner_launch.py` does not
expose it as a launch argument.

## Target Publisher

`target_publisher` always publishes:

| Topic | Type | Purpose |
| --- | --- | --- |
| `/target/odom` | `nav_msgs/msg/Odometry` | Target position, velocity, orientation, angular velocity |
| `/target/accel` | `geometry_msgs/msg/AccelStamped` | Target linear and angular acceleration |
| `/target/true_odom` | `nav_msgs/msg/Odometry` | Synthetic truth for `gazebo_*` modes only |
| `/target/true_accel` | `geometry_msgs/msg/AccelStamped` | Synthetic truth for `gazebo_*` modes only |

`/target/odom` and `/target/accel` are estimator outputs in `gazebo_*` and
`mocap` modes. In `gazebo_*` modes, the analytic motion model is used only to
generate pose measurements and the separate `/target/true_*` topics for bag
comparison. In `mocap` mode there is no separate truth topic; the mocap pose is
the measurement feeding the same estimator. In `trace_replay` mode,
`/target/odom` and `/target/accel` are replayed directly from CSV
(`t,x,y,z,vx,vy,vz,ax,ay,az`) with interpolation.

Target mode, planning frame, and OCP selection are separate:

```bash
# Synthetic circular target, target-frame relative odometry for tf_imu/tf_noimu.
ros2 launch comando_planner target_launch.py \
  target_mode:=gazebo_circle planning_frame:=target_frame

# Synthetic figure-8 target, body-frame relative odometry for bf_imu/bf_noimu.
ros2 launch comando_planner target_launch.py \
  target_mode:=gazebo_figure8 planning_frame:=body_frame

# Mocap target pose measurements, world-frame state for stateswitch.
ros2 launch comando_planner target_launch.py \
  target_mode:=mocap rigid_body_name:=stmini planning_frame:=world

# Replayed hardware target trace, world-frame state for stateswitch.
ros2 launch comando_planner target_launch.py \
  target_mode:=trace_replay planning_frame:=world \
  trace_csv_path:=/tmp/target_trace_circling.csv
```

It can also publish relative drone odometry:

| `planning_frame` | Extra topic | Used by |
| --- | --- | --- |
| `world` | none | Absolute OCPs and target-relative OCPs that transform internally |
| `body_frame` | `/drone/body_relative_odom` | `tracking_bodyrate_bf_*` |
| `target_frame` | `/drone/target_frame_odom` | `tracking_bodyrate_tf_*` |
| `shifted` | `/drone/relative_odometry` | Diagnostic/historical output; not selected by current OCP descriptors |

Valid target modes are `gazebo_circle`, `gazebo_figure8`, `mocap`, and
`trace_replay`. OCP aliases are also available: `tf_imu`, `tf_noimu`,
`bf_imu`, `bf_noimu`, `state_switch`, and `circle_target`.

## Runtime Parameters

`planner_launch.py` declares infrastructure parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `drone_name` | `cf_1` | Drone namespace for Crazyflie topics and planner output topics |
| `platform` | `crazyflie` | `crazyflie` or `mavros` |
| `solver` | `alipddp` | Only implemented backend |
| `enable_logging` | `true` | Writes CSV logs under `./logs` |
| `target_odom_topic` | `/target/odom` | Target odometry input |
| `target_accel_topic` | `/target/accel` | Target acceleration input |
| `target_predicted_accel_topic` | `/target/predicted_accel` | Future acceleration buffer for `tracking_circle_target` |
| `body_relative_odom_topic` | `/drone/body_relative_odom` | Body-relative drone-state input |
| `record_bag` | `true` | Starts `ros2 bag record`; set false if you do not want a bag |
| `bag_output` | empty | Optional override; by default bags are written under the active run log folder |

`ocp_launch.py` sets the active profile and triggers it:

| Parameter | Default | Notes |
| --- | --- | --- |
| `ocp_type` | `landing` | Must be one of the registered keys above |
| `mode` | `mpc` | `mpc` or `open_loop`; `tracking_circle_target` rejects `mpc` |
| `n_replay` | `7` | Minimum replay steps before a new MPC plan can swap in |
| `hover_target_x/y/z` | `0.0` | Terminal point for absolute full-state OCPs |
| `command_seq` | `1` | Must increase to start a new command profile |

## Logging and RViz

When `enable_logging:=true`, each run creates:

```text
./logs/{drone}_{ocp}_{mode}_{solver}_{timestamp}/
```

with:

| File | Contents |
| --- | --- |
| `all_solves.csv` | Solver metadata plus every OCP state/control node and target snapshot data |
| `commanded_state.csv` | Full-state commands and computed acceleration feedforward |
| `commanded_hover_state.csv` | Crazyflie body-rate OCP dispatch as `[vx, vy, z_distance, yaw_rate]` |
| `commanded_bodyrate_state.csv` | MAVROS body-rate dispatch as `[T_ms2, omega_x, omega_y, omega_z]` |
| `actual_state.csv` | Raw measured state rows labelled by coordinate mode |
| `bags/comando_debug/` | ROS bag recorded for the same run when `record_bag:=true` |

Live visualization topics:

| Topic | Type |
| --- | --- |
| `/{drone}/planned_trajectory` | `nav_msgs/msg/Path` |
| `/{drone}/planner_debug_markers` | `visualization_msgs/msg/MarkerArray` |

Open the provided layout from the workspace root after building:

```bash
source /opt/ros/humble/setup.zsh
source install/setup.zsh
ros2 bag play src/CoManDO_planner/logs/RUN_FOLDER/bags/comando_debug --clock

source /opt/ros/humble/setup.zsh
source install/setup.zsh
rviz2 -d install/comando_planner/share/comando_planner/rviz/comando_debug.rviz

```

Automatic bag recording is per `planner_launch.py` process, not per
`ocp_launch.py` trigger. If you run four OCP profiles while one planner stays
alive, they are all in the same bag. Restart the planner, or pass a different
`bag_output:=...`, when you want a separate bag per test.

## Validation Targets

The package installs small executable test targets:

```bash
ros2 run comando_planner test_poly_ocp
ros2 run comando_planner test_trajectory_replayer
ros2 run comando_planner test_target_frame_warm_start
ros2 run comando_planner test_planner_core
ros2 run comando_planner test_stateswitch_predictor
```

`scripts/test_bodyrate_cmd.py` is also installed as `test_bodyrate_cmd.py` for
manual Crazyflie body-rate command checks.

## More Documentation

Start with `docs/COMANDO_PLANNER_INTERFACE.md` for the full ROS interface and
`docs/CoManDO_planner/index.md` for the chapter-style architecture notes.
