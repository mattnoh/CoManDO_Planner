# 01 — Crazyflie Messaging and Trajectory Sending

---

## 1.1 Crazyswarm2 Interface Overview

The Crazyflie is controlled via **Crazyswarm2**, a ROS 2 package that bridges ROS topics to the Crazyflie firmware over Bluetooth/USB radio. CoManDO communicates with it using two directions:

**Incoming state** (platform → CoManDO):

| Topic | Type | Content |
|-------|------|---------|
| `/<drone_name>/pose` | `geometry_msgs/PoseStamped` | position + orientation (quaternion) |
| `/<drone_name>/odom` | `nav_msgs/Odometry` | linear velocity + angular rate |

**Outgoing command** (CoManDO → platform):

| Topic | Type | Content |
|-------|------|---------|
| `/<drone_name>/cmd_full_state` | `crazyflie_interfaces/FullState` | full 13-dim state setpoint |

---

## 1.2 The FullState Message

`crazyflie_interfaces/FullState` is the richest setpoint type available for the Crazyflie.
It feeds directly into Mellinger's full-state feedback controller onboard the firmware.

```
Header header

geometry_msgs/Pose  pose      # position + orientation
geometry_msgs/Twist twist     # linear velocity + angular rate
geometry_msgs/Vector3 acc     # acceleration feedforward
```

**How CoManDO fills it:**

```cpp
msg.pose.position.{x,y,z}     = x_cmd(0..2)    // position   (m)
msg.twist.linear.{x,y,z}      = x_cmd(3..5)    // velocity   (m/s)
msg.pose.orientation.{w,x,y,z}= x_cmd(6..9)    // quaternion
msg.twist.angular.{x,y,z}     = x_cmd(10..12)  // body rates (rad/s)
msg.acc.{x,y,z}               = 0              // zero — Mellinger computes from attitude PD
```

Acceleration is set to zero because the Mellinger controller onboard the Crazyflie
recomputes it internally from the attitude error. Feeding a non-zero acceleration
would double-count the correction.

---

## 1.3 The Angular Rate deg/s Quirk

Crazyswarm2 publishes angular rates in **degrees per second** in the `/odom` topic's
`twist.angular` fields. This is inconsistent with the ROS convention (rad/s) and
must be converted before feeding the state to the OCP.

**In `crazyflie_bridge.cpp`:**

```cpp
constexpr double DEG2RAD = M_PI / 180.0;
state_(10) = msg->twist.twist.angular.x * DEG2RAD;
state_(11) = msg->twist.twist.angular.y * DEG2RAD;
state_(12) = msg->twist.twist.angular.z * DEG2RAD;
```

Failure to apply this conversion causes the OCP to see angular rates ~57× too large,
which destabilises the solver and produces physically impossible warm-starts.

---

## 1.4 Split State Assembly

The Crazyflie bridge subscribes to **two separate topics** to assemble the full 13-dim state.
`/pose` carries position + attitude; `/odom` carries velocity + angular rate.
The state is only forwarded to `/mpc/state` once **both** have been received at least once.

```
/cf_1/pose ──► poseCallback()  ──┐
                                  ├──► state_[0..9]  ──► maybePublishState()
/cf_1/odom ──► odomCallback()  ──┘       state_[10..12]
                                               │
                                  if (pose_received_ && odom_received_)
                                               │
                                               ▼
                                         /mpc/state
                                    (nav_msgs/Odometry)
```

Both callbacks share a mutex (`state_mutex_`) to prevent a partial state
(e.g. new position with stale velocity) from being forwarded mid-update.

---

## 1.5 How Trajectory Commands Are Sent

The CoManDO solver does **not** send one command per solve. Instead it uses a
**trajectory replay** pattern:

1. The solver publishes a full `MpcCommand` containing the entire predicted state
   trajectory `X[0..N]` and control trajectory `U[0..N-1]`.

2. The bridge's `replay_timer_` fires at `ocp_dt` Hz (e.g. 20 Hz for `ocp_dt=0.05s`).
   On each tick it reads the next entry from the stored trajectory and sends it to
   the Crazyflie as a `FullState` command.

3. When a new `MpcCommand` arrives, the bridge replaces the stored trajectory
   and resets the replay index to `replay_start_idx` (which accounts for solve latency
   — see `02_mpc_timing.md`).

```
Solve arrives every 500ms (2 Hz solver_rate):

Timeline:  |──────────── 500ms ──────────────|
           ▼                                 ▼
      New MpcCommand                   New MpcCommand
           │  │  │  │  │  │  │  │  │  │
           ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
          t0 t1 t2 t3 t4 t5 t6 t7 t8 t9     ← replay ticks at 20Hz
     X[skip] X[s+1] ... X[N] X[N] X[N]      ← trajectory entries dispatched
```

This means the Crazyflie receives smooth, high-rate (20 Hz) commands even when
the MPC solver only runs at 1–2 Hz.

---

## 1.6 Transition Blend

When a new trajectory arrives, naively switching to it causes a discontinuous
position/velocity command. The Mellinger controller then overshoots trying to
track the step change.

**Fix:** linearly interpolate the full 13-dim state from the last commanded point
to the new trajectory's first entry over `BLEND_STEPS = 12` ticks (~600ms at 20Hz).

```
blend_alpha = 1 - (blend_steps_remaining / BLEND_STEPS)
blend_beta  = 1 - blend_alpha

blended_pos_vel  = beta * blend_from[0:6]  + alpha * x_cmd[0:6]
blended_quat     = normalise(beta * q_from + alpha * q_cmd)   // shortest path
blended_omega    = beta * blend_from[10:13] + alpha * x_cmd[10:13]
```

Quaternion blending uses linear interpolation (LERP) followed by renormalisation.
This is valid for the small attitude jumps between consecutive MPC solutions.
For large attitude changes SLERP would be more accurate, but LERP is sufficient here
given the blend window (~600ms) is much longer than a typical inter-solve attitude divergence.

**Shortest-path correction:** if `q_from · q_cmd < 0`, negate `q_cmd` before blending
to ensure the interpolation takes the short arc rather than the long way around.