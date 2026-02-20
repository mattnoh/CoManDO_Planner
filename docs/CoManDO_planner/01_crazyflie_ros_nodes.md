# Chapter 1: Crazyflie ROS Nodes Documentation

## 1. Subscribers (Data from the Crazyflie)

To receive data from the Crazyflie, we must configure the logging topics in our robot's YAML file. This tells Crazyswarm2 which variables to stream and at what frequency.

### YAML Configuration Example (`crazyflie.yaml`)

```yaml
robot_types:
  cf_sim:
    motion_capture:
      tracking: "vendor"
    big_quad: false
    firmware_logging:
      enabled: true
      default_topics:
        pose:
          frequency: 100      # Hz
        odom:
          frequency: 100      # Hz
```

> **Note:** The `firmware_logging` section enables streaming of internal Crazyflie variables. We can add more topics like `scan` or `status` if needed. We can also add motion capture for later use

### Default Topics and Variables

The mapping between ROS topic names, message types, Crazyflie variables, and callback functions is defined in the Crazyswarm2 server source:

```python
# From crazyflie_server.py
self.default_log_type = {
    "pose":   PoseStamped,
    "scan":   LaserScan,
    "odom":   Odometry,
    "status": Status
}

self.default_log_vars = {
    "pose":  ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
              'stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw'],
    "scan":  ['range.front', 'range.left', 'range.back', 'range.right'],
    "odom":  ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
              'stabilizer.yaw', 'stabilizer.roll', 'stabilizer.pitch',
              'kalman.statePX', 'kalman.statePY', 'kalman.statePZ',
              'gyro.z', 'gyro.x', 'gyro.y'],
    "status": ['supervisor.info', 'pm.vbatMV', 'pm.state', 'radio.rssi']
}

self.default_log_fnc = {
    "pose":   self._log_pose_data_callback,
    "scan":   self._log_scan_data_callback,
    "odom":   self._log_odom_data_callback,
    "status": self._log_status_data_callback
}
```

For a **full state estimate** (position, velocity, attitude, angular rates), we can typically combine data from the `pose` and `odom` topics. 

### Available ROS 2 Topics

When the Crazyflie is running, we see topics similar to these:

```bash
$ ros2 topic list
/cf_1/cmd_full_state              /cmd_full_state
/cf_1/cmd_hover                   /cmd_vel
/cf_1/cmd_position                /cmd_vel_legacy
/cf_1/cmd_vel_legacy              /joy
/cf_1/cmd_velocity_world          /joy/set_feedback
/cf_1/odom                        /parameter_events
/cf_1/pose                        /poses
/cf_1/robot_description           /rosout
/cf_1/status                      /tf
/tf_static
```

---

## 2. How We Subscribe to and Assemble the State

The full state vector **x** is 13-dimensional. No single ROS topic provides all 13 fields, so we fuse two sources: the motion-capture pose and the onboard Kalman/gyro odometry.

```
x = [ x, y, z,       ← position       (indices 0–2)
      vx, vy, vz,    ← velocity        (indices 3–5)
      qw, qx, qy, qz,← quaternion      (indices 6–9)
      wx, wy, wz ]   ← angular rate    (indices 10–12)
```

We maintain a single `current_state_` vector of size 13. Each callback writes only the fields it owns, so the vector is always a composite of the two most recent messages.

### 2.1 `/pose` Callback — Position and Orientation

The `/pose` topic publishes a `PoseStamped` at 100 Hz from the motion-capture system. It provides the most accurate position and orientation available.

```cpp
void poseCallback(const PoseStamped::SharedPtr msg)
{
    current_state_(0) = msg->pose.position.x;   // x[0]
    current_state_(1) = msg->pose.position.y;   // x[1]
    current_state_(2) = msg->pose.position.z;   // x[2]

    current_state_(6) = msg->pose.orientation.w; // x[6]  qw
    current_state_(7) = msg->pose.orientation.x; // x[7]  qx
    current_state_(8) = msg->pose.orientation.y; // x[8]  qy
    current_state_(9) = msg->pose.orientation.z; // x[9]  qz

    pose_received_ = true;
}
```

This callback writes **7 of the 13** state fields (indices 0–2 and 6–9).

### 2.2 `/odom` Callback — Velocity and Angular Rate

The `/odom` topic publishes an `Odometry` message at 100 Hz. The linear velocity comes from the onboard Kalman filter (`kalman.statePX/Y/Z`) and is expressed in the **world frame**. The angular rates come from the gyroscope (`gyro.x/y/z`) and are expressed in the **body frame**.

```cpp
void odomCallback(const Odometry::SharedPtr msg)
{
    current_state_(3) = msg->twist.twist.linear.x;  // x[3]  vx (world, Kalman)
    current_state_(4) = msg->twist.twist.linear.y;  // x[4]  vy (world, Kalman)
    current_state_(5) = msg->twist.twist.linear.z;  // x[5]  vz (world, Kalman)

    constexpr double DEG2RAD = M_PI / 180.0;
    current_state_(10) = msg->twist.twist.angular.x * DEG2RAD; // x[10] wx
    current_state_(11) = msg->twist.twist.angular.y * DEG2RAD; // x[11] wy
    current_state_(12) = msg->twist.twist.angular.z * DEG2RAD; // x[12] wz

    odom_received_ = true;
}
```

> **Unit conversion:** Crazyswarm2 populates `twist.angular` directly from the firmware's `gyro.x/y/z` variables, which are logged in **deg/s**. The conversion `× π/180` is applied here before storing into `x[10–12]`. If you confirm your firmware build streams rad/s instead, remove the `DEG2RAD` factor.

This callback writes **6 of the 13** state fields (indices 3–5 and 10–12).

### 2.3 How the Two Sources Are Mixed

The two callbacks run independently and write into different slices of the same `current_state_` vector. The resulting assembly is:

```
current_state_ after both callbacks:

  index  source   field
  ─────────────────────────────────────────
  0      /pose    position x
  1      /pose    position y
  2      /pose    position z
  3      /odom    velocity vx  (Kalman, world frame)
  4      /odom    velocity vy
  5      /odom    velocity vz
  6      /pose    quaternion qw
  7      /pose    quaternion qx
  8      /pose    quaternion qy
  9      /pose    quaternion qz
  10     /odom    angular rate wx  (gyro, body frame, converted to rad/s)
  11     /odom    angular rate wy
  12     /odom    angular rate wz
```

There is no explicit synchronisation between the two callbacks. The MPC solver is gated on both `pose_received_` and `odom_received_` being true before it starts, and after that it reads `current_state_` at whatever moment the solver timer fires. Because both topics run at 100 Hz and the solver runs at ≤10 Hz, in practice `current_state_` is always fresh.

---

## 3. Publishers (Commands to the Crazyflie)

To command the Crazyflie, you will publish to one of the available command topics. The most flexible (and recommended) topic for full feedforward control is **`cmd_full_state`** that is usually published to the **Mellinger Controller** 

### Topic: `/cf_1/cmd_full_state`

This message expects:
- **pose**   : position and orientation
- **twist**  : linear and angular velocity
- **acc**    : linear acceleration

Here is an example of a published `cmd_full_state` message (viewed with `ros2 topic echo /cf_1/cmd_full_state`):

```yaml
header:
  stamp:
    sec: 1771575901
    nanosec: 424698092
  frame_id: world
pose:
  position:
    x: 1.289e-19
    y: 5.293e-20
    z: 0.0149992
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
twist:
  linear:
    x: 2.456e-11
    y: -2.327e-11
    z: 1.7317
  angular:
    x: -3.394e-08
    y: -3.509e-08
    z: -5.122e-14
acc:
  x: 1.277e-10
  y: -1.185e-10
  z: 17.2149
```

### 3.1 How We Publish a Command

Each command is built directly from a state trajectory node returned by the MPC solver. Given a trajectory node `x[k]`, the fields map as follows:

```
msg.pose.position     ← x[k][0–2]   (position)
msg.twist.linear      ← x[k][3–5]   (velocity)
msg.pose.orientation  ← x[k][6–9]   (quaternion qw, qx, qy, qz)
msg.twist.angular     ← x[k][10–12] (angular rates, rad/s)
msg.acc               ← a_ff        (feedforward acceleration, see below)
```

### 3.2 Feedforward Acceleration Calculation

The Mellinger controller accepts an acceleration feedforward term that improves tracking. We compute it as a finite-difference of consecutive velocity states from the MPC trajectory:

```
a_ff = ( x[k+1][3–5] − x[k][3–5] ) / dt
```

where `dt` is the OCP time step. This gives the rate of change of velocity between two consecutive trajectory nodes, which approximates the desired acceleration at node `k`.

In code:

```cpp
int next_idx = min(idx + 1, N - 1);

acc_cmd = (state_traj_[next_idx].segment(3, 3)
         - state_traj_[idx   ].segment(3, 3)) / ocp_dt_;
```

If `idx` is already the last node (solver was late), both `idx` and `next_idx` point to the same node and `acc_cmd` is zero — a safe fallback.

---

## **Key points:**
 - The `cmd_full_state` topic implements a **feedforward** controller – you provide the full desired state and the firmware tries to track it.
 - The orientation is represented as a quaternion (`x,y,z,w`).
 - Position and orientation come from MoCap via `/pose`; velocity and angular rates come from the Kalman filter and gyroscope via `/odom`.
 - The acceleration feedforward is not measured — it is derived from the MPC solution by finite-differencing consecutive velocity nodes.

---

[Next Chapter: Planner Node Setup](02_planner_node_setup.md)