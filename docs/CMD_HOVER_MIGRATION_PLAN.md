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

### Step 6 — Add `ocp_tracking_cmdhover` OCP (the tracking OCP)

This adds the body-relative tracking OCP from the standalone test as a
first-class registered OCP.

**New file:** `include/ocp/ocp_tracking_cmdhover.hpp`

The OCP formulation is identical to what was tested in
`quad_cf_tracking_cmdhover.cpp`:
- State: 14D body-frame relative (`[p_T_body, v_d_body, q, ω, accumulated_dt]`)
- Control: 5D `[fz, Mx, My, Mz, Theta]` (variable timestep)
- Dynamics: `Quad6DOFVarTimeRelativeTV` (body-frame, SE(3)-relative)

Command extraction for `cmd_hover` uses planned `X[k+1]`:
- `vx` = `x_next(IDX_VD + 0)`
- `vy` = `x_next(IDX_VD + 1)`
- `z`  = `TARGET_Z - x_next(IDX_P + 2)`
- `yaw_rate` = `x_next(IDX_OM + 2)`

Because the OCP state is **relative/body-frame** but `publishHoverCommand`
reads from the **planned state trajectory**, the `publishCommand` path for
this OCP needs to extract hover commands from the planned trajectory instead
of the replayed absolute state. Two options:

**Option A (simpler):** Store the hover command sequence alongside the
control trajectory in `SolverResult::extra` and replay from there.

**Option B:** Add a `extractHoverCmd(state, control) → HoverCmd` callback to
`OCPDescriptor` and call it inside `publishCommand`.

Recommend **Option A** for now — it's self-contained and doesn't change the
planner node's publish interface.

Implementation sketch for `SolverResult::extra`:
```cpp
struct HoverCmdSequence {
    std::vector<std::array<float,4>> cmds;  // [vx, vy, z, yaw_rate] per step
};
```
Populated in `post_process_result` callback; replayer sends `cmds[replay_idx]`
instead of calling `publishHoverCommand(state)`.

**Registry entry:**
```cpp
{"hover_cmdhover", {
    "hover_cmdhover",
    /* dt */      HoverCmdHoverOCP::TH_INIT,
    /* n_replay */ HoverCmdHoverOCP::DEFAULT_N_REPLAY,
    /* mass */    HoverCmdHoverOCP::DEFAULT_MASS_KG,
    OCPDescriptor::WarmStart::Shift,
    OCPDescriptor::CommandMode::CmdHover,
    false,       // needs_target_trajectory
    // transform_state: convert absolute drone+target state to relative body frame
    [](const Eigen::VectorXd& x, const TargetSnapshot& t) { ... },
    // validate_target: require valid target
    [](const TargetSnapshot& t, double now, double max_age) { return t.valid; },
    // post_process_result: populate HoverCmdSequence into extra
    [](SolverResult& r, const TargetSnapshot& t) { ... },
    nullptr, // prepare_extra
    nullptr, // prepare_log_meta
    HoverCmdHoverOCP::getSolverParams,
    [](const OCPCreateArgs& a) { return HoverCmdHoverOCP::create(a); }
}},
```

---

### Step 7 — Remove `cmd_vel_legacy` (deferred, after validation)

Only after Step 5 and Step 6 are validated in CrazySim:

1. Remove `CmdVelLegacy` from `CommandMode` enum
2. Remove `publishLegacyCommand` and `mapThrustNToLegacyU16` from `crazyflie.hpp`
3. Remove `cmd_vel_legacy_pub` from `Handles`, remove from `setup()`
4. Remove the `CmdVelLegacy` branch from `publishCommand` in `planner_node.cpp`
5. Remove `LegacyCommandDebug` struct and `logger_.logLegacyCommand` calls
6. Remove `legacy_thrust_unlocked` logic

**Do not do this until both hover_body (Step 5) and hover_cmdhover (Step 6)
are confirmed working in simulation.**

---

## File change summary

| File | Change |
|------|--------|
| `include/core/ocp_registry.hpp` | Add `CmdHover` to enum; update `hover_body` entry; add `hover_cmdhover` entry |
| `include/platform/crazyflie.hpp` | Add `cmd_hover_pub` to `Handles`; add to `setup()`; add `publishHoverCommand()` |
| `src/planner_node.cpp` | Add `CmdHover` branch in `publishCommand`; update mode log string |
| `include/ocp/ocp_hover_cmdhover.hpp` | New file — body-relative tracking OCP |
| `include/ocp/ocp_registry.hpp` (top-level) | Include new OCP header |
| `include/core/ocp_registry.hpp` (later) | Remove `CmdVelLegacy` in Step 7 |
| `include/platform/crazyflie.hpp` (later) | Remove legacy functions in Step 7 |

---

## Safety notes

- `cmd_vel_legacy` publisher and `publishLegacyCommand` are kept until Step 7.
  `hover_body` will simply stop using them at Step 5; the publisher is harmless
  while idle.
- The `CmdVelLegacy` enum value is kept until Step 7 — removing it earlier
  breaks compilation of any code that references it.
- Confirm `z_distance` semantics in CrazySim before first flight. Wrong
  sign or reference frame → immediate altitude loss.
- The hover-body OCP (`Step 5`) uses the full 13D absolute state, so
  `publishHoverCommand(state)` reading `state(2)` for z is straightforward.
  The tracking OCP (`Step 6`) uses a 14D relative state — command extraction
  must come from the planned trajectory, not the feedback state.
