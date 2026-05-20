# Dynamic Target Hardware Integration

This guide describes what an external target estimator or predictor must publish
for the current `comando_planner` implementation.

The planner separates live target feedback from future target prediction:

| Stream | Default topic | Required by | Purpose |
| --- | --- | --- | --- |
| Target odometry | `/target/odom` | all target-tracking OCPs | Current target position, velocity, orientation, angular velocity |
| Target acceleration | `/target/accel` | body-rate OCP freshness, `stateswitch` acceleration snapshot | Current target linear and angular acceleration |
| Predicted acceleration | `/target/predicted_accel` | `tracking_circle_target` | Future linear acceleration samples over the planning horizon |

## Target Odometry

Publish `nav_msgs/msg/Odometry` in the planner world frame, normally ENU:

| Field | Meaning |
| --- | --- |
| `header.stamp` | Measurement time. Use ROS time; do not leave zero unless you intentionally want the planner callback to substitute `now()` |
| `header.frame_id` | `world` or your local ENU equivalent |
| `pose.pose.position` | Target position |
| `pose.pose.orientation` | Target orientation as quaternion |
| `twist.twist.linear` | Target linear velocity |
| `twist.twist.angular` | Target angular velocity |

The planner treats target odom older than about `0.2 s` as stale for OCPs that
need target feedback.

## Target Acceleration

Publish `geometry_msgs/msg/AccelStamped`:

| Field | Meaning |
| --- | --- |
| `header.stamp` | Acceleration measurement time |
| `accel.linear` | Target linear acceleration |
| `accel.angular` | Target angular acceleration, if available |

If `accel.angular` is not available, `target_tracker` estimates angular
acceleration from odometry finite differences and lets this stream override it
when present.

## Predicted Acceleration

`tracking_circle_target` is the only current OCP that requires the predicted
acceleration stream. It is open-loop only in `planner_node.cpp`.

Publish `trajectory_msgs/msg/MultiDOFJointTrajectory`:

| Field | Requirement |
| --- | --- |
| `header.stamp` | Absolute ROS time for prediction origin |
| `points[i].time_from_start` | Monotonic offset from `header.stamp` |
| `points[i].accelerations[0].linear` | Future target acceleration consumed by the planner |
| `points[i].transforms` | Optional; ignored by the current OCP path |
| `points[i].velocities` | Optional; ignored by the current OCP path |

The planner stores these samples in a `TargetAccelBuffer` with a nominal `dt`
computed from the first two trajectory points. The buffer is considered fresh
for two seconds from `header.stamp`.

Minimal C++ publisher pattern:

```cpp
trajectory_msgs::msg::MultiDOFJointTrajectory traj;
traj.header.stamp = node->now();
traj.header.frame_id = "world";
traj.joint_names.push_back("target");

const double dt = 0.05;
const int n = 100;
for (int i = 0; i < n; ++i) {
    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;
    pt.time_from_start = rclcpp::Duration::from_seconds(i * dt);

    geometry_msgs::msg::Twist acc;
    acc.linear.x = ax;
    acc.linear.y = ay;
    acc.linear.z = az;
    pt.accelerations.push_back(acc);

    traj.points.push_back(pt);
}
predicted_accel_pub->publish(traj);
```

## Relative Drone Odometry

The planner can read different drone-state frames depending on the active OCP.
For real hardware you can either publish these directly from your estimator or
use `target_publisher` as a development helper.

| OCP family | Drone state required by planner | Helper output |
| --- | --- | --- |
| `hover`, `landing`, `stateswitch`, `tracking_circle_target` | Absolute drone pose/odom | none |
| `tracking_bodyrate_bf_*` | Target position in drone body frame, relative velocity in body frame, relative attitude | `/drone/body_relative_odom` |
| `tracking_bodyrate_tf_*` | Drone position/velocity in target frame, relative attitude | `/drone/target_frame_odom` |

`target_publisher` creates the helper outputs when launched with
`drone_odom_mode:=body_frame` or `drone_odom_mode:=target_frame` and supplied
with `drone_odom_topic` plus `drone_pose_topic`.

## Built-In Target Publisher Modes

`target_publisher` supports:

| `target_mode` | Behavior |
| --- | --- |
| `gazebo_circle` | Synthetic circular pose measurements through the shared estimator |
| `gazebo_figure8` | Synthetic figure-8 pose measurements through the shared estimator |
| `mocap` | Reads `/rigid_bodies`, selects `rigid_body_name`, and feeds the shared estimator |

Runtime parameters currently declared by the node:

| Parameter | Default | Notes |
| --- | --- | --- |
| `target_mode` | `gazebo_circle` | `gazebo_circle`, `gazebo_figure8`, or `mocap` |
| `target_yaw_rate` | `0.3` | Synthetic yaw rate in `gazebo_*` modes |
| `publish_hz` | `100.0` | Output rate |
| `frame_id` | `world` | Header frame for target messages |
| `rigid_body_name` | `stmini` | Mocap rigid body name |
| `drone_odom_topic` | `/cf_1/odom` | Drone odom input for helper relative odometry |
| `drone_pose_topic` | `/cf_1/pose` | Drone pose input for helper relative odometry |
| `drone_odom_mode` | `none` | `none`, `shifted_world`, `body_frame`, or `target_frame` |
| `body_relative_input_angular_unit` | `deg_s` | Set `rad_s` if the source odom angular velocity is already radians per second |
| `debug_body_relative_trace` | `false` | Throttled state conversion logs |

The source comments still mention runtime circle shape parameters, but the
current implementation does not declare them. Change `include/target/circular_target.hpp`
or replace the publisher if you need runtime-selectable trajectories.

## Startup and Gating Behavior

For target-tracking OCPs, the planner waits until required streams are fresh
before solving.

`tracking_circle_target` startup in open loop:

1. Wait for drone state.
2. Wait for valid target odometry.
3. Wait for fresh `/target/predicted_accel`.
4. Solve once.
5. Reconstruct the world-frame command trajectory from target odom plus the
   predicted acceleration buffer.
6. Replay the trajectory, then transition to hover hold.

For body-rate MPC OCPs, stale target data prevents new solves. The replay path
continues using the last accepted plan until it becomes stale, then the node
holds/pauses according to the normal MPC stale-plan behavior.
