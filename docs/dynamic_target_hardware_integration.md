# CoManDO Planner: Dynamic Target Hardware Integration

When integrating an actual hardware target (e.g., a real moving robot tracked by a Vicon system or an onboard Kalman Filter), your target estimator node must publish specific ROS messages for the `comando_planner` to track it successfully.

## Required ROS Topics

Your target estimator must publish on **three topics**. By default, the planner expects:

| Topic | Message Type | Rate | Purpose |
|---|---|---|---|
| `/target/odom` | `nav_msgs/msg/Odometry` | Fast (~50-100Hz) | Immediate feedback. Provides the drone with the target's current position and velocity to compute instantaneous tracking errors. |
| `/target/accel` | `geometry_msgs/msg/AccelStamped` | Fast (~50-100Hz) | Immediate feedback. Provides the target's current instantaneous acceleration to feed forward into the MPC dynamics. |
| `/target/predicted_trajectory` | `trajectory_msgs/msg/MultiDOFJointTrajectory` | Slow (~1-10Hz) | Future prediction. Required by OCPs like `tracking_circle_target` that need to know the *future* trajectory to plan intercepts or landings over the MPC horizon. |

---

## 1. Setting up the Target Estimator Node

### The `Odometry` and `AccelStamped` Streams

These two topics are standard. When you observe the moving target via your Kalman Filter or Motion Capture:

1. **Stamp properly**: Set `header.stamp = now()`. The planner uses these timestamps to enforce freshness checks. If `now() - stamp > 0.2s`, the planner considers the target "lost" and will pause the solver to hover safely.
2. **Coordinate Frame**: Set `header.frame_id = "world"` (or your local ENU equivalent).
3. **Values**: 
   - `odom.pose.pose.position` -> Target (x, y, z)
   - `odom.twist.twist.linear` -> Target velocity (vx, vy, vz)
   - `accel.accel.linear` -> Target acceleration (ax, ay, az)

### The `MultiDOFJointTrajectory` Prediction Stream

This is the most critical for dynamic planning. You must predict where the target *will be* over the next few seconds (the planner horizon is roughly 4 seconds). 

Even if you just assume constant velocity or constant acceleration from your Kalman filter, you must explicitly populate and publish this trajectory array so the planner's integration engine can look it up in `O(1)` time.

**Requirements for the Trajectory message:**
1. Let `N` be the number of predicted points (e.g., 80 to 400).
2. The `header.stamp` of the message MUST be the absolute ROS time (`t=0` for the prediction).
3. Populate the `points` array where each point has a `time_from_start` offset.
4. **Crucial:** You must populate the `accelerations[0].linear` field for each point. The `tracking_circle_target` dynamics model primarily integrates the target over time using these future acceleration vectors.

#### Minimal C++ Example (publishing a constant-acceleration prediction)

```cpp
trajectory_msgs::msg::MultiDOFJointTrajectory traj;
traj.header.stamp = this->now();
traj.header.frame_id = "world";
traj.joint_names.push_back("target");

const double dt = 0.05; // 50ms resolution 
const int N = 100;      // 5 seconds into the future

// Assuming you have current pos (p), vel (v), and accel (a)
for (int i = 0; i < N; ++i) {
    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;
    pt.time_from_start = rclcpp::Duration::from_seconds(i * dt);
    
    // Constant acceleration integration
    double t = i * dt;
    auto pt_pos = p + v*t + 0.5*a*t*t;
    auto pt_vel = v + a*t;
    auto pt_acc = a;
    
    geometry_msgs::msg::Transform trans;
    trans.translation.x = pt_pos.x();
    trans.translation.y = pt_pos.y();
    trans.translation.z = pt_pos.z();
    pt.transforms.push_back(trans);
    
    geometry_msgs::msg::Twist vel;
    vel.linear.x = pt_vel.x();
    vel.linear.y = pt_vel.y();
    vel.linear.z = pt_vel.z();
    pt.velocities.push_back(vel);
    
    geometry_msgs::msg::Twist acc;
    acc.linear.x = pt_acc.x();
    acc.linear.y = pt_acc.y();
    acc.linear.z = pt_acc.z();
    pt.accelerations.push_back(acc);
    
    traj.points.push_back(pt);
}
traj_pub_->publish(traj);
```

---

## 2. Planner Safe-Mode Gating Behavior

The planner features a robust state machine when waiting for hardware data.

**If the OCP needs future target data (e.g., `tracking_circle_target`):**
1. The planner will power on and immediately check for the `/target/predicted_trajectory` topic.
2. If it is empty, or the message's `header.stamp` is older than `2.0` seconds, the planner enters **Wait Mode**. 
3. The drone will maintain a local position hold (hover), rejecting the open-loop command, and log: `[OpenLoop] IDLE — waiting for /target/predicted_trajectory...`.
4. As soon as a fresh trajectory message arrives, the solver fires instantly, initializes the variables, and dispatches the tracking maneuver.

This guarantees your drone will never blindly fly into undefined space waiting for a telemetry uplink.
