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
| `circle` | Synthetic circular target using constants in `include/target/circular_target.hpp` |
| `qualisys` | Reads `/rigid_bodies`, selects `rigid_body_name`, and finite-differences pose history |

Runtime parameters currently declared by the node:

| Parameter | Default | Notes |
| --- | --- | --- |
| `target_mode` | `circle` | `circle` or `qualisys` |
| `target_yaw_rate` | `0.3` | Synthetic yaw rate in circle mode |
| `publish_hz` | `100.0` | Output rate |
| `frame_id` | `world` | Header frame for target messages |
| `rigid_body_name` | `stmini` | Qualisys rigid body name |
| `drone_odom_topic` | `/cf_1/odom` | Drone odom input for helper relative odometry |
| `drone_pose_topic` | `/cf_1/pose` | Drone pose input for helper relative odometry |
| `drone_odom_mode` | `none` | `none`, `shifted_world`, `body_frame`, or `target_frame` |
| `body_relative_input_angular_unit` | `deg_s` | Set `rad_s` if the source odom angular velocity is already radians per second |
| `debug_body_relative_trace` | `false` | Throttled state conversion logs |

The source comments still mention runtime circle shape parameters, but the
current implementation does not declare them. Change `include/target/circular_target.hpp`
or replace the publisher if you need runtime-selectable trajectories.

## Relative Target Estimate (drone body frame)

For relative-sensing landings (`stc_landing`), the target can be driven by a
**body-frame relative measurement** instead of an absolute target feed — what a
downward camera plus an attitude estimate produces. `target_publisher` splits
this into two stages that communicate only over a ROS topic, so a real estimator
can replace the simulator without touching the synthesis side.

```
stage 1 (sim, optional)                stage 2 (synthesis)
generator + drone odom  ──▶  /drone/relative_target_estimate  ──▶  /target/odom
                                                                   /target/accel
```

**Message contract** — `nav_msgs/Odometry` on `relative_estimate_topic`
(default `/drone/relative_target_estimate`):

| Field | Content |
| --- | --- |
| `header.frame_id` | `drone_body` (FLU body frame, ENU-consistent) |
| `child_frame_id` | `target_in_drone_body` |
| `pose.pose.position` | target position relative to the drone, in body axes |
| `twist.twist.linear` | target velocity relative to the drone, in body axes |
| `pose.pose.orientation` | identity; `pose.covariance[21] = -1.0` marks it unmeasured |
| `pose/twist.covariance[14]` | Reserved z-validity marker (`-1.0` = estimator has no z). **Stage 2 does not read it today** — z is always taken from `pad_z`. |

Publish all three position/velocity components. Stage 2 rotates the full vector
into world **before** overriding z, so keeping body-z is what makes the world x/y
correct while the drone is tilted.

**Synthesis math** (`R` = drone attitude from `/mavros/local_position/odom`):

```
p_target_world = p_drone + R * p_rel_body ;  z ← pad_z
v_target_world = v_drone + R * v_rel_body ;  vz ← 0
```

The measurement convention is a pure frame projection (`R^T (x_t - x_d)`, no
`omega x r` term), so the synthesis is its exact algebraic inverse — in
simulation the round trip reproduces ground truth to machine precision.

`v_drone` comes from the drone odom twist; `/mavros/local_position/odom` has
`child_frame_id: base_link`, i.e. a **body-frame** twist, which is the
`drone_odom_twist_frame:=body` default. Set it to `world` only if your odom
source publishes a world-frame twist.

Synthesized `/target/accel` is zero (the relative estimate carries no
acceleration) and is published at the same stamp as `/target/odom`, because the
tracker's 0.2 s freshness gate requires both streams. Publish the estimate at
**20 Hz or better** to keep margin against that gate.

| Parameter | Default | Notes |
| --- | --- | --- |
| `target_source` | `ground_truth` | `relative_estimate` makes `/target/odom` come from the synthesis stage |
| `sim_relative_estimate` | `false` | Simulate the measurement from the built-in trajectory generator |
| `pad_z` | `0.0` | World z assigned to the synthesized target (landing pad height / mocap z) |
| `relative_estimate_topic` | `/drone/relative_target_estimate` | Swap point for a real estimator |
| `publish_ground_truth_debug` | `true` | In relative mode, generator truth moves to `/target/odom_gt` + `/target/accel_gt` |
| `drone_odom_twist_frame` | `body` | Interpretation of the drone odom twist |

Simulated round trip (verifies the pipeline against ground truth):

```bash
roslaunch comando_planner target_launch.launch \
  target_source:=relative_estimate sim_relative_estimate:=true pad_z:=0.2
# compare /target/odom against /target/odom_gt
```

Real estimator (synthesis stage unchanged):

```bash
roslaunch comando_planner target_launch.launch \
  target_source:=relative_estimate sim_relative_estimate:=false pad_z:=0.0 \
  publish_ground_truth_debug:=false
# your node publishes /drone/relative_target_estimate
```

Note that in this mode the target's acceleration is hidden from the OCP's
constant-accel predictor (zeros are published), and the synthesized target
carries no yaw — both are inherent to the relative-sensing contract, not bugs.

### Why the world-frame hop is not a frame change

The measurement is **drone-centered** (`target relative to drone`, body axes),
matching what a camera outputs and the existing `/drone/body_relative_odom`
convention. The `stc_landing` OCP is **target-centered, world-aligned**:
`transform_state` subtracts the target from the drone state and never rotates.
These are reconciled by stage 2, not by any change to the dynamics:

```
sensor          p_rel^B  = R^T (p_target - p_drone)      drone-centered, body axes
stage 2         p_target = p_drone + R p_rel^B           absolute world
transform_state x_rel    = x_drone - p_target            target-centered, world axes
```

Substituting stage 2 into `transform_state` gives `x_rel = -R p_rel^B`: the
absolute drone position **cancels**. The world frame is only a transport format
that lets the existing tracker and snapshot plumbing be reused unchanged. No new
dynamics are involved.

The cancellation is exact only if both sides use the same drone-odom sample.
Stage 2 uses the odom cached when the estimate arrived; the planner uses its own
snapshot at solve time. Slow estimator drift cancels (the point of the design);
timing skew between the two samples does not. Keep the relative measurement and
the odom it is paired with time-aligned.

### What still requires an absolute source

| Quantity | Cancels? | Consequence |
| --- | --- | --- |
| Drone x, y | Yes | Horizontal drift in the local estimate is invisible to the solver |
| Drone velocity x, y | Yes | Same |
| **Drone z** | **No** | `pad_z` is a constant, so `x_rel,z = drone_odom_z - pad_z`. Height above the pad comes straight from the drone's altitude estimate and needs a real source (mocap z, rangefinder, or baro referenced to takeoff). |
| Attitude `R` | No | Required — it is the rotation between sensor body axes and the world axes the setpoints live in. Comes from the IMU/EKF, no mocap needed. |

Relative sensing therefore removes the dependence on an absolute **horizontal**
target fix, not on altitude. To make z relative as well, the estimator must
supply relative z and stage 2 must stop overriding it
(`p_target_z = p_drone_z + (R p_rel^B)_z`), at which point z cancels like x/y —
that is what the reserved `covariance[14]` marker is for.

Independently of all this, `stc_landing` emits absolute `PositionTarget`
setpoints, so PX4 still needs a self-consistent local position estimate to track
them. Drift cancels, but the estimate must exist: offboard position control does
not function without one.

Stage 2 fails safe — it skips synthesis when drone odom is missing or older than
0.5 s, so `/target/odom` goes stale and the planner's freshness gate holds the
command rather than acting on a bad transform.

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
