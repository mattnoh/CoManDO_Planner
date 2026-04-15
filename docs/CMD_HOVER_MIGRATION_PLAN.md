# cmd_hover Migration Plan

Replace `cmd_vel_legacy` (roll/pitch/thrust RPY commander) with `cmd_hover`
(velocity + altitude crazyflie commander) in CoManDO planner.

---

## Background

`cmd_vel_legacy` sends `(roll_deg, pitch_deg, yawrate_deg_s, thrust_u16)` and
relies on the drone's attitude controller to produce body velocity. The OCP
plans in full SE(3) and we then approximate roll/pitch from the planned
quaternion — lossy and sensitive to thrust-lock quirks.

`cmd_hover` sends `(vx_body, vy_body, z_distance, yaw_rate)` directly to the
Crazyflie's built-in velocity+altitude controller. Because the OCP state
already contains body-frame velocity (`vx`, `vy`) and altitude (`z`), this is
a **direct feedforward with no approximation step**.

The viability was confirmed in the standalone test
`ALIPDDP-main/problem_examples/quad_cf_tracking_cmdhover.cpp`, where
injecting planned `vx/vy` directly into the hover plant matched the wrench
plant trajectory closely.

---

## Current state (what exists today)

| Item | Current |
|------|---------|
| `CommandMode` enum | `CmdFullState`, `CmdVelLegacy` |
| `hover_body` OCP | registered with `CmdVelLegacy` |
| `publishCommand` in `planner_node.cpp` | branches on `CmdVelLegacy` → `publishLegacyCommand` |
| `crazyflie.hpp` | has `publishLegacyCommand`, no `publishHoverCommand` |
| `cmd_hover` message type | `crazyflie_interfaces/msg/HoverTwist` (or equivalent) |

---

## Target state

| Item | Target |
|------|--------|
| `CommandMode` enum | add `CmdHover`; keep `CmdVelLegacy` (don't remove yet) |
| `hover_body` OCP | switch to `CmdHover` |
| new OCP `hover_cmdhover` | registered with `CmdHover`, uses `quad_cf_tracking_cmdhover` dynamics |
| `crazyflie.hpp` | add `publishHoverCommand` function |
| `publishCommand` in `planner_node.cpp` | add `CmdHover` branch |

---

## Step-by-step plan

### Step 1 — Check the crazyflie_interfaces message type

Before writing any code, confirm the exact message type and field names.

```bash
ros2 interface show crazyflie_interfaces/msg/Hover
```

Expected fields: `vx` (m/s body), `vy` (m/s body), `z_distance` (m from
ground, positive up), `yaw_rate` (rad/s). Confirm field names match before
proceeding.

---

### Step 2 — Add `CmdHover` to `CommandMode` (ocp_registry.hpp)

File: `include/core/ocp_registry.hpp` lines ~76–78.

```cpp
enum class CommandMode {
    CmdFullState,
    CmdVelLegacy,   // keep — do not remove
    CmdHover,       // ADD
};
```

No other changes to the registry struct needed at this step.

---

### Step 3 — Add `publishHoverCommand` to `crazyflie.hpp`

File: `include/platform/crazyflie.hpp`

Add publisher handle:
```cpp
struct Handles {
    // ... existing ...
    rclcpp::Publisher<crazyflie_interfaces::msg::Hover>::SharedPtr cmd_hover_pub;
};
```

Add to `setup()` (alongside the existing `cmd_vel_legacy_pub` creation):
```cpp
handles.cmd_hover_pub = node->create_publisher<crazyflie_interfaces::msg::Hover>(
    "/" + drone_name + "/cmd_hover", 10);
```

Add `publishHoverCommand` function. The OCP state vector is 13D:
`[px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]`. For `cmd_hover`:
- `vx` → `state(3)` (body-frame, already body if coming from hover-body OCP)
- `vy` → `state(4)`
- `z_distance` → `state(2)` (absolute z; confirm what the firmware expects —
  distance from ground or absolute altitude — match what CrazySim expects)
- `yaw_rate` → `state(12)` (wz, body-frame yaw rate)

```cpp
inline void publishHoverCommand(
    Handles& handles,
    const Eigen::VectorXd& state)
{
    if (!handles.cmd_hover_pub || state.size() < 13) return;

    crazyflie_interfaces::msg::Hover msg;
    msg.vx           = static_cast<float>(state(3));
    msg.vy           = static_cast<float>(state(4));
    msg.z_distance   = static_cast<float>(state(2));   // confirm semantics
    msg.yaw_rate     = static_cast<float>(state(12));

    handles.cmd_hover_pub->publish(msg);
}
```

**Important:** Verify whether `z_distance` in the Crazyflie firmware means
absolute altitude or distance above the takeoff point. CrazySim's hover
controller likely interprets it as absolute z. Double-check in the CrazySim
source before running on hardware.

---

### Step 4 — Add `CmdHover` branch in `planner_node.cpp`

File: `src/planner_node.cpp`, function `publishCommand` (~line 956).

```cpp
void publishCommand(const Eigen::VectorXd& s, const Eigen::VectorXd& u) {
    if (platform_ == "crazyflie") {
        if (command_mode_ == OCPDescriptor::CommandMode::CmdVelLegacy) {
            const auto dbg = platform::crazyflie::publishLegacyCommand(cf_handles_, s, u);
            // ... existing logging ...
        } else if (command_mode_ == OCPDescriptor::CommandMode::CmdHover) {
            platform::crazyflie::publishHoverCommand(cf_handles_, s);
        } else {
            platform::crazyflie::publishCommand(this, cf_handles_, s, u, mass_kg_);
        }
    } else if (platform_ == "px4") {
        platform::px4::publishCommand(this, px4_handles_, s);
    }
}
```

Also update the log string wherever `command_mode_` is printed (lines ~123,
~639) to include `"cmd_hover"` for the new mode.

---

### Step 5 — Switch `hover_body` OCP to `CmdHover`

File: `include/core/ocp_registry.hpp`, `hover_body` entry (~line 131–148).

Change:
```cpp
OCPDescriptor::CommandMode::CmdVelLegacy,
```
To:
```cpp
OCPDescriptor::CommandMode::CmdHover,
```

This is the first real test: run `hover_body` with `cmd_hover` instead of
`cmd_vel_legacy` and verify the drone holds altitude and has no yaw drift.

**Test protocol for Step 5 (CrazySim):**
1. Launch CrazySim + CoManDO with `ocp_type:=hover_body`
2. Confirm `/cf1/cmd_hover` topic appears and is publishing
3. Confirm `/cf1/cmd_vel_legacy` is silent (publisher still exists, just unused)
4. Hover for 10s — check altitude holds within ±0.05m, no yaw drift
5. If OK → proceed to Step 6

---

### Step 6 — Declare drone odom mode per-OCP in `OCPDescriptor`

Currently `drone_state_is_relative_` and `odom_topic_override` are operator-set
launch parameters. The operator must remember to set them correctly for each OCP.
Instead, each OCP should declare what kind of drone state it expects, and the
planner should reconfigure automatically when the OCP switches.

#### 6a — Add `DroneOdomMode` to `OCPDescriptor`

File: `include/core/ocp_registry.hpp`

```cpp
struct OCPDescriptor {
    // ...existing fields...

    enum class DroneOdomMode {
        AbsoluteWorld,       // standard: /cf1/pose + /cf1/odom (world-frame pos+vel)
                             // used by: hover, hover_body, landing
        AbsoluteMinusTarget, // world-frame difference: drone_pos - target_pos
                             // used by: stateswitch, tracking_circle, tracking_circle_target
                             // (this is the current "enable_relative_odom" path)
        BodyRelative,        // true body-relative: C(q)^T*(target-drone), C(q)^T*v_drone
                             // used by: tracking_cmdhover
    };

    DroneOdomMode drone_odom_mode = DroneOdomMode::AbsoluteWorld;
};
```

Each existing OCP gets tagged:
- `hover`, `hover_body`, `landing` → `AbsoluteWorld`
- `stateswitch`, `tracking_circle`, `tracking_circle_target` → `AbsoluteMinusTarget`
- `tracking_cmdhover` → `BodyRelative`

#### 6b — Planner reacts to `DroneOdomMode` on OCP switch

File: `src/planner_node.cpp`

The planner reads `desc.drone_odom_mode` when the OCP changes (in
`onSetParameters` where `command_mode_` is already updated, ~line 616) and
updates its subscription accordingly.

For `AbsoluteWorld` and `AbsoluteMinusTarget`: subscribe to the raw
`/cf1/pose` + `/cf1/odom` topics as today. The `drone_state_is_relative_` flag
and `transform_state` callback handle the difference between the two at solve
time — no subscription change needed.

For `BodyRelative`: re-call `platform::crazyflie::setup()` with
`odom_topic_override = "/drone/body_relative_odom"` (or whatever the configured
topic is). This switches the subscription to the pre-transformed body-relative
odometry topic. Also set `drone_state_is_relative_ = true` so the planner
passes `x0` directly to the OCP without any further transform.

Implementation sketch inside `onSetParameters` (after `command_mode_` is set):

```cpp
const auto& desc = OCPRegistry::getDescriptor(ocp_type_);
const bool needs_body_relative =
    (desc.drone_odom_mode == OCPDescriptor::DroneOdomMode::BodyRelative);

if (needs_body_relative != drone_state_is_relative_) {
    drone_state_is_relative_ = needs_body_relative;
    const std::string override = needs_body_relative
        ? body_relative_odom_topic_   // new config param, default "/drone/body_relative_odom"
        : "";
    // Tear down and rebuild the crazyflie subscriptions with the new topic
    cf_handles_.reset();
    platform::crazyflie::setup(this, sensor_cb_group_, drone_name_,
                               override, cf_state_, cf_state_mutex_, cf_handles_);
    RCLCPP_INFO(this->get_logger(),
        "[OCP switch] drone odom mode: %s",
        needs_body_relative ? "body-relative" : "absolute");
}
```

Add `body_relative_odom_topic_` as a declare/get parameter in
`planner_runtime_config.hpp` (default: `"/drone/body_relative_odom"`).
This keeps it configurable at launch without being per-OCP.

#### 6c — Remove `drone_state_is_relative` from operator-facing launch params

Since it is now driven by `DroneOdomMode`, remove it from `planner_launch.py`
and `planner_runtime_config.hpp` as a user-facing parameter. It becomes an
internal flag that only the planner sets based on the active OCP descriptor.

The `drone_odom_topic` launch param (used for the override in existing relative
OCPs) can also be removed and replaced by `body_relative_odom_topic`.

---

### Step 7 — Fix `target_publisher.cpp`: add true body-relative sensor mode

The existing `enable_relative_odom` path computes a **world-frame difference**:
```
position = drone_pos - target_pos    (wrong sign, wrong frame)
velocity = drone_vel - target_vel    (relative velocity, NOT v_d_body)
```
This is what `stateswitch` and `tracking_circle` use — keep it, just rename it
clearly.

The OCP state needed by `tracking_cmdhover` is:
```
p_T_body  = C(q)^T * (target_pos - drone_pos)   ← target in DRONE body frame
v_d_body  = C(q)^T * v_drone_world               ← drone velocity in body frame
q, ω      = drone attitude + rates (unchanged)
```

**Changes to `target_publisher.cpp`:**

1. Rename param `enable_relative_odom` → `enable_absolute_relative_odom` and
   `relative_odom_topic` → `absolute_relative_odom_topic`. Behaviour unchanged.
   Log string updated so it's clear what this publishes.

2. Add new param `enable_body_relative_odom` (default: `false`) and
   `body_relative_odom_topic` (default: `/drone/body_relative_odom`).

3. When `enable_body_relative_odom` is true, additionally subscribe to
   `/<drone_name>/pose` (for quaternion + world-frame position) and
   `/<drone_name>/odom` (for world-frame linear velocity + angular rate):

```cpp
// On each timer tick, if body-relative enabled and both drone sub callbacks fired:
Eigen::Quaterniond q(drone_pose.orientation.w, drone_pose.orientation.x,
                     drone_pose.orientation.y, drone_pose.orientation.z);
Eigen::Matrix3d R = q.toRotationMatrix();   // body → world

Eigen::Vector3d p_T_body = R.transpose() *
    (target_pos_world - drone_pos_world);   // target in body frame

Eigen::Vector3d v_d_body = R.transpose() * drone_vel_world;  // drone vel in body frame

nav_msgs::msg::Odometry msg;
msg.header.stamp         = stamp;
msg.header.frame_id      = "drone_body";
msg.child_frame_id       = "body_relative";
msg.pose.pose.position.x = p_T_body.x();
msg.pose.pose.position.y = p_T_body.y();
msg.pose.pose.position.z = p_T_body.z();
// Orientation and angular rate pass through unchanged
msg.pose.pose.orientation = drone_pose.orientation;
msg.twist.twist.linear.x  = v_d_body.x();
msg.twist.twist.linear.y  = v_d_body.y();
msg.twist.twist.linear.z  = v_d_body.z();
// Angular velocity in rad/s (crazyswarm2 odom is deg/s — convert here)
msg.twist.twist.angular.x = drone_odom.angular.x * DEG2RAD;
msg.twist.twist.angular.y = drone_odom.angular.y * DEG2RAD;
msg.twist.twist.angular.z = drone_odom.angular.z * DEG2RAD;
body_rel_odom_pub_->publish(msg);
```

The planner's `odom_topic_override` path in `crazyflie.hpp` sets all 13 state
elements from one Odometry message — it works as-is for this output since we
control the message contents and units.

**Note:** `target_publisher` already subscribes to a drone odom topic
(`drone_odom_topic_`). Reuse that subscription; just add a pose subscription
for the quaternion (odom only has velocity, not orientation).

#### Summary of what each OCP uses after Steps 6+7

| OCP | `DroneOdomMode` | Drone input topic | What arrives at planner |
|-----|----------------|-------------------|------------------------|
| `hover`, `landing` | `AbsoluteWorld` | `/cf1/pose` + `/cf1/odom` | 13D world-frame state |
| `hover_body` | `AbsoluteWorld` | `/cf1/pose` + `/cf1/odom` | 13D world-frame state |
| `stateswitch` | `AbsoluteMinusTarget` | `/cf1/pose` + `/cf1/odom` | 13D world-frame state, `transform_state` subtracts target |
| `tracking_circle*` | `AbsoluteMinusTarget` | `/cf1/pose` + `/cf1/odom` | 13D world-frame state, `transform_state` subtracts target |
| `tracking_cmdhover` | `BodyRelative` | `/drone/body_relative_odom` | 13D body-relative state, passed directly as x0 |

---

### Step 7 — Add `tracking_cmdhover` OCP

**New file:** `include/ocp/ocp_tracking_cmdhover.hpp`

Copy the OCP formulation directly from
`ALIPDDP-main/problem_examples/quad_cf_tracking_cmdhover.cpp`:
- State (13D physical, 14D augmented): `[p_T_body(3), v_d_body(3), q(4), ω(3), acc_dt(1)]`
- Control (4D physical, 5D augmented): `[fz, Mx, My, Mz, Theta]` (variable timestep)
- Dynamics: `Quad6DOFVarTimeRelativeTV`
- Costs, constraints: same glideslope cone, capture sphere, attitude regularization

Expose as constants: `IDX_P`, `IDX_VD`, `IDX_OM`, `TARGET_Z`, `TH_INIT`, `DEFAULT_N_REPLAY`.

#### 7a — Hover command extraction from replayed body-relative state

In MPC mode the planner re-solves constantly and the replayer streams commands
between solves. **No pre-computation needed.** At each replay tick, the replayer
returns `x_cmd` — the interpolated body-relative state at that instant. The
hover command is derived directly:

```
vx         = x_cmd(IDX_VD + 0)            // body-frame drone vx
vy         = x_cmd(IDX_VD + 1)            // body-frame drone vy
z_distance = TARGET_Z - x_cmd(IDX_P + 2)  // drone world z = target_z - p_T_body[2]
yaw_rate   = x_cmd(IDX_OM + 2)            // body-frame wz
```

The z formula is exact (not an approximation): `p_T_body[2]` is the vertical
component of the target-in-body-frame vector. Since the target doesn't move
vertically (`TARGET_Z` is fixed), `drone_z = TARGET_Z - p_T_body[2]`
regardless of attitude. This is exactly `deriveHoverCmdFromPlannedState` from
the standalone test.

To make `publishHoverCommand` work for both OCPs without a special case in the
planner node, add an optional callback to `OCPDescriptor`:

```cpp
// In OCPDescriptor:
std::function<std::array<float,4>(const Eigen::VectorXd& x_cmd)> extract_hover_cmd;
```

- `hover_body` → `extract_hover_cmd = nullptr`, fall back to default:
  `[state(3), state(4), state(2), state(12)]` (absolute z, works for absolute-state OCP)
- `tracking_cmdhover` → `extract_hover_cmd` set to:
  ```cpp
  [](const Eigen::VectorXd& x) -> std::array<float,4> {
      return {
          static_cast<float>(x(TrackingCmdHoverOCP::IDX_VD + 0)),
          static_cast<float>(x(TrackingCmdHoverOCP::IDX_VD + 1)),
          static_cast<float>(TrackingCmdHoverOCP::TARGET_Z - x(TrackingCmdHoverOCP::IDX_P + 2)),
          static_cast<float>(x(TrackingCmdHoverOCP::IDX_OM + 2))
      };
  }
  ```

In `publishCommand` in `planner_node.cpp`:

```cpp
} else if (command_mode_ == OCPDescriptor::CommandMode::CmdHover) {
    auto& desc = OCPRegistry::getDescriptor(ocp_type_);
    std::array<float,4> cmd;
    if (desc.extract_hover_cmd) {
        cmd = desc.extract_hover_cmd(s);
    } else {
        cmd = { float(s(3)), float(s(4)), float(s(2)), float(s(12)) };
    }
    platform::crazyflie::publishHoverCommandDirect(cf_handles_, cmd);
}
```

No changes to `TrajectoryReplayer`, no changes to `SolverResult`. The replayer
already interpolates the 13D body-relative state correctly via the existing
variable-DT path.

#### 7b — Registry entry

```cpp
{"tracking_cmdhover", {
    "tracking_cmdhover",
    TrackingCmdHoverOCP::TH_INIT,
    TrackingCmdHoverOCP::DEFAULT_N_REPLAY,
    TrackingCmdHoverOCP::DEFAULT_MASS_KG,
    OCPDescriptor::WarmStart::Shift,
    OCPDescriptor::CommandMode::CmdHover,
    false,    // needs_target_trajectory
    nullptr,  // transform_state — body-relative state arrives pre-formed from sensor topic
    [](const TargetSnapshot& t, double /*now*/, double /*max_age*/) { return t.valid; },
    nullptr,  // post_process_result
    nullptr,  // prepare_extra
    nullptr,  // prepare_log_meta
    TrackingCmdHoverOCP::getSolverParams,
    [](const OCPCreateArgs& a) { return TrackingCmdHoverOCP::create(a); },
    // extract_hover_cmd:
    [](const Eigen::VectorXd& x) -> std::array<float,4> {
        return {
            static_cast<float>(x(TrackingCmdHoverOCP::IDX_VD + 0)),
            static_cast<float>(x(TrackingCmdHoverOCP::IDX_VD + 1)),
            static_cast<float>(TrackingCmdHoverOCP::TARGET_Z
                               - x(TrackingCmdHoverOCP::IDX_P + 2)),
            static_cast<float>(x(TrackingCmdHoverOCP::IDX_OM + 2))
        };
    }
}},
```

#### 7c — Target velocity buffer

The OCP dynamics need `v_T_inertial(t)` (target velocity in world frame) as a
time-varying input — this is the `TargetVelocityBuffer` in the standalone test.
In the planner this comes from `TargetSnapshot.velocity` and the existing
`target_accel_buffer` infrastructure already used by `stateswitch`. The
`prepare_extra` callback in the registry entry for `stateswitch` shows exactly
how to populate this buffer from the target snapshot. `tracking_cmdhover`
should use the same pattern.

#### 7d — Test protocol

1. Start CrazySim, take off to `TARGET_Z` with `hover_body`
2. Start `target_publisher` with `enable_body_relative_odom:=true`,
   `drone_pose_topic:=/cf1/pose`, `drone_odom_topic:=/cf1/odom`
3. Sanity-check `/drone/body_relative_odom`: when drone is at world `(dx, dy, TARGET_Z)`
   and target at origin, `pose.position ≈ (−dx_body, −dy_body, −TARGET_Z_body)`
   rotated by drone quaternion
4. Switch OCP to `tracking_cmdhover` — planner automatically re-subscribes to
   `/drone/body_relative_odom` because `DroneOdomMode::BodyRelative` triggers
   the subscription swap in `onSetParameters`
5. Confirm `/cf1/cmd_hover` publishes with non-zero `vx` as drone approaches target
6. Confirm `p_T_body` horizontal norm decreases — capture converging
7. Confirm altitude stays near `TARGET_Z`

---

### Step 8 — Remove `cmd_vel_legacy` (deferred)

Only after `hover_body` (Step 5) and `tracking_cmdhover` (Step 7) are validated
in CrazySim:

1. Remove `CmdVelLegacy` from `CommandMode` enum
2. Remove `publishLegacyCommand`, `mapThrustNToLegacyU16`, `LegacyCommandDebug`,
   `quaternionToRollPitchYaw`, `bodyRatesToYawRate` from `crazyflie.hpp`
3. Remove `cmd_vel_legacy_pub` from `Handles` and from `setup()`
4. Remove the `CmdVelLegacy` branch and `logger_.logLegacyCommand` from
   `planner_node.cpp`
5. Remove `legacy_thrust_unlocked` logic

Build and verify all other OCPs still compile and run.

---

## File change summary

| File | Step | Change |
|------|------|--------|
| `include/core/ocp_registry.hpp` | 2 | Add `CmdHover` to `CommandMode` enum |
| `include/platform/crazyflie.hpp` | 3 | Add `cmd_hover_pub` to `Handles`; add to `setup()`; add `publishHoverCommandDirect` |
| `src/planner_node.cpp` | 4 | Add `CmdHover` branch calling `extract_hover_cmd`; update log string |
| `include/core/ocp_registry.hpp` | 5 | Switch `hover_body` to `CmdHover`; tag `DroneOdomMode::AbsoluteWorld` |
| `include/core/ocp_registry.hpp` | 6 | Add `DroneOdomMode` enum + field to `OCPDescriptor`; tag all existing OCPs |
| `src/planner_node.cpp` | 6 | React to `DroneOdomMode` on OCP switch: swap subscription; set `drone_state_is_relative_` |
| `launch/planner_launch.py` + `config/planner_runtime_config.hpp` | 6 | Remove `drone_state_is_relative` param; add `body_relative_odom_topic` param |
| `src/target_publisher.cpp` | 7 | Rename old relative mode; add `enable_body_relative_odom` mode with true body-frame transform |
| `include/ocp/ocp_tracking_cmdhover.hpp` | 7 | New file — body-relative tracking OCP |
| `include/core/ocp_registry.hpp` | 7 | Add `extract_hover_cmd` field; add `tracking_cmdhover` entry with `DroneOdomMode::BodyRelative` |
| `include/core/ocp_registry.hpp` | 8 | Remove `CmdVelLegacy` |
| `include/platform/crazyflie.hpp` | 8 | Remove all legacy functions |
| `src/planner_node.cpp` | 8 | Remove legacy branch and logging |

---

## Safety notes

- `cmd_vel_legacy` code is kept through Step 7. Harmless while idle.
- **Existing OCPs are unaffected by Step 6**: `stateswitch` and `tracking_circle*`
  are tagged `AbsoluteMinusTarget` which maps to the existing subscription path +
  `transform_state` callback. Their behaviour does not change.
- **No `transform_state` for `tracking_cmdhover`**: body-relative state arrives
  pre-formed. `transform_state = nullptr` is intentional.
- **Subscription swap on OCP switch**: `cf_handles_.reset()` + `setup()` re-creates
  subscriptions mid-flight. Confirm this doesn't cause a missed-message gap > one
  solve period (~100ms). If it does, gate the swap so it only fires when the drone
  is paused.
- **z formula is exact**: `z_drone = TARGET_Z - p_T_body[2]` holds regardless of
  attitude. `p_T_body[2]` is a geometric range component, not a projected altitude.
- **`extract_hover_cmd` default** (nullptr → use `state(2)` for z) is only correct
  for absolute-frame OCPs. Every body-relative OCP must set this explicitly.
- **`validateTrajectory`** checks `X[k](2) < -0.05` for altitude. For body-relative
  state `X[k](2)` = `p_T_body[2]` (vertical separation to target), not drone
  altitude. This check must be skipped for `tracking_cmdhover` — add an
  `OCPDescriptor::skip_altitude_validation = false` flag or check `drone_odom_mode`.
