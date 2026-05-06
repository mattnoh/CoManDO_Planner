# Chapter 1: Crazyflie, MAVROS, and Target ROS Interfaces

`planner_node.cpp` delegates platform-specific message conversion to headers in
`include/platform/`. The planner core works with Eigen vectors and descriptor
metadata; platform adapters turn those vectors into ROS messages.

## Crazyflie Adapter

Source: `include/platform/crazyflie.hpp`

### Absolute State Input

For absolute OCPs, the Crazyflie adapter subscribes directly to:

| Topic | Type | State fields |
| --- | --- | --- |
| `/{drone_name}/pose` | `geometry_msgs/msg/PoseStamped` | `px, py, pz, qw, qx, qy, qz` |
| `/{drone_name}/odom` | `nav_msgs/msg/Odometry` | `vx, vy, vz, wx, wy, wz` |

The resulting 13D state is:

```text
[px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
```

Crazyswarm2 angular velocity is converted from degrees per second to radians per
second in this absolute-input path.

### Relative State Input

When the active OCP descriptor requests a relative drone odometry mode, the
planner rebuilds the Crazyflie subscriptions and reads a single odometry topic:

| Descriptor mode | Topic |
| --- | --- |
| `BodyFrameRelative` | `body_relative_odom_topic`, default `/drone/body_relative_odom` |
| `TargetFrameRelative` | `/drone/target_frame_odom` |

Relative odometry override topics are assumed to already use SI units and
radians per second. They are not converted again.

### Crazyflie Command Output

| OCP command mode | Topic | Type |
| --- | --- | --- |
| `CmdFullState` | `/{drone_name}/cmd_full_state` | `crazyflie_interfaces/msg/FullState` |
| `CmdBodyRate` | `/{drone_name}/cmd_hover` | `crazyflie_interfaces/msg/Hover` |

For `CmdFullState`, the adapter computes acceleration feedforward from the OCP
thrust and commanded attitude:

```text
a_world = R(q) * [0, 0, fz / mass] + [0, 0, -9.81]
```

For `CmdBodyRate`, the planner converts the OCP state/control to Crazyflie hover
commands:

```text
[vx_body, vy_body, z_world, yaw_rate]
```

The command adapter clamps `vx_body` and `vy_body` to `[-1, 1]` and `z_world` to
`[0.1, 3.0]`.

## MAVROS Adapter

Source: `include/platform/mavros.hpp`

The MAVROS adapter is compiled only when `mavros_msgs` is found by CMake.

| Direction | Topic | Type |
| --- | --- | --- |
| Input | `/mavros/local_position/odom` | `nav_msgs/msg/Odometry` |
| Full-state output | `/mavros/setpoint_raw/local` | `mavros_msgs/msg/PositionTarget` |
| Body-rate output | `/mavros/setpoint_raw/attitude` | `mavros_msgs/msg/AttitudeTarget` |

The adapter starts a pre-arm sequence immediately:

1. Publish neutral attitude setpoints for two seconds.
2. Request `OFFBOARD`.
3. Request arm.
4. Publish commands only after the planner marks the vehicle armed.

For body-rate output, the OCP control vector is interpreted as:

```text
[T_ms2, omega_x, omega_y, omega_z, Theta]
```

Only the first four values are sent. `T_ms2` is converted to normalized MAVROS
thrust using the `hover_thrust` parameter.

## Target Tracker

Source: `include/platform/target_tracker.hpp`

The planner always subscribes to target topics so runtime OCP switching does not
require a node restart.

| Topic parameter | Default | Type | Stored fields |
| --- | --- | --- | --- |
| `target_odom_topic` | `/target/odom` | `nav_msgs/msg/Odometry` | Position, velocity, orientation, angular velocity |
| `target_accel_topic` | `/target/accel` | `geometry_msgs/msg/AccelStamped` | Linear and angular acceleration |
| `target_predicted_accel_topic` | `/target/predicted_accel` | `trajectory_msgs/msg/MultiDOFJointTrajectory` | Future acceleration samples |

Target odometry and acceleration have separate timestamps. Freshness checks use
both timestamps when an OCP needs a fully valid target snapshot.

## Target Publisher Helper

Source: `src/target_publisher.cpp`

`target_publisher` publishes `/target/odom` and `/target/accel` in `circle` or
`qualisys` mode. It can also publish relative drone odometry for body-frame and
target-frame OCPs.

Common commands:

```bash
# Synthetic target only
ros2 run comando_planner target_publisher --ros-args \
  -p target_mode:=circle -p drone_odom_mode:=none

# Body-frame relative odometry for tracking_bodyrate_bf_*
ros2 run comando_planner target_publisher --ros-args \
  -p target_mode:=circle \
  -p drone_odom_mode:=body_frame \
  -p drone_odom_topic:=/cf_1/odom \
  -p drone_pose_topic:=/cf_1/pose

# Target-frame relative odometry for tracking_bodyrate_tf_*
ros2 run comando_planner target_publisher --ros-args \
  -p target_mode:=circle \
  -p drone_odom_mode:=target_frame \
  -p drone_odom_topic:=/cf_1/odom \
  -p drone_pose_topic:=/cf_1/pose
```

The current circle shape is defined in `include/target/circular_target.hpp`, not
by runtime launch parameters.

[Next Chapter: Planner Node Setup](02_planner_node_setup.md)
