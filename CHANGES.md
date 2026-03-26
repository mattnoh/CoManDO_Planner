# CHANGES: Porting quad_cf_tracking_rh_circle to CoManDO_planner

This document describes all modifications made to fix the `tracking_circle` OCP implementation.

**Reference:** `ALIPDDP-main/problem_examples/quad_cf_tracking_rh_circle.cpp`

---

## Bug 1: Warm-start controls not applied (CRITICAL)

**File:** `include/ocp/ocp_tracking_circle.hpp`

**Problem:** The `create()` function accepted `prev_U` parameter but never applied it. Warm-start controls were computed in `QuadrotorMPC::makeUwarm()` but immediately discarded when `OCPRegistry::create()` rebuilt the cold-start rollout, overwriting them with `problem->setInitialControl(k, u0)`.

**Fix:** Apply warm-start controls AFTER the cold-start rollout loop:

```cpp
// After cold-start state initialization loop (line ~340)
if (!prev_U.empty()) {
    for (int k = 0; k < HORIZON && k < static_cast<int>(prev_U.size()); ++k) {
        problem->setInitialControl(k, prev_U[k]);
    }
}
```

---

## Bug 2: t_abs never updated (CRITICAL)

**File:** `src/planner_node.cpp`

**Problem:** The absolute time `t_abs` (time since tracking started) was never tracked or passed to the OCP. The terminal constraint `CircularInterceptCon` needs `t_abs` to compute the correct target position at intercept time.

**Fix:** 
1. Added member variables to `PlannerNode`:
   - `double t_abs_ = 0.0;`
   - `rclcpp::Time tracking_start_time_;`
   - `TrackingCircleOCP::CircularTarget circle_target_;`

2. In `resetForNewCommand()`: Reset `t_abs_ = 0.0` and set `tracking_start_time_ = this->now()`

3. In `solverLoop()`: Initialize tracking circle on first solve, update `t_abs_` based on elapsed wall-clock time

4. In `mpcReplayTick()`: After executing each step, increment `t_abs_` by the executed `Theta`:
   ```cpp
   if (ocp_type_ == "tracking_circle" && u_cmd.size() >= 5) {
       double Th_executed = u_cmd(4);  // IDX_THETA = 4
       t_abs_ += Th_executed;
   }
   ```

---

## Bug 3: CircularTarget and t_abs hardcoded (CRITICAL)

**Files:** 
- `include/quadrotor_mpc.hpp`
- `include/ocp_registry.hpp`

**Problem:** `OCPRegistry::create()` had hardcoded placeholder values for `CircularTarget` and `t_abs`:
```cpp
tgt.center = Eigen::Vector3d(0.0, 0.0, 1.0);
tgt.R = 3.0; tgt.omega = 0.5; tgt.phi0 = 0.0;
double t_abs = 0.0;  // Always zero!
```

**Fix:**

1. Extended `QuadrotorMPC::Config` to include:
   ```cpp
   TrackingCircleOCP::CircularTarget circle_target;
   double t_abs = 0.0;
   ```

2. Updated `OCPRegistry::create()` signature to accept live values:
   ```cpp
   inline std::shared_ptr<OptimalControlProblem<double>> create(
       // ... existing params ...
       const TrackingCircleOCP::CircularTarget& circle_target = {},
       double t_abs = 0.0
   );
   ```

3. For `tracking_circle` OCP type, pass live values to `TrackingCircleOCP::create()`

---

## Bug 4: Missing theta clamping in warm-start (MODERATE)

**File:** `src/quadrotor_mpc.cpp`

**Problem:** `makeUwarm()` only zeroed moments in the tail control but didn't clamp thrust and theta to valid bounds. The reference implementation clamps both:
```cpp
u_tail(0) = std::max(FMIN, std::min(FMAX, u_tail(0)));
u_tail(IDX_THETA) = std::max(THL, std::min(THH, u_tail(IDX_THETA)));
```

**Fix:** Added thrust and theta clamping to tail control:
```cpp
if (u_tail.size() >= 5) {  // NU_SS = 5 for tracking_circle
    u_tail(1) = 0.0;  // Mx
    u_tail(2) = 0.0;  // My
    u_tail(3) = 0.0;  // Mz
    u_tail(0) = std::max(FMIN, std::min(FMAX, u_tail(0)));  // thrust
    u_tail(4) = std::max(THL, std::min(THH, u_tail(4)));   // theta
}
```

Also added `tracking_circle` to the warm-start branch in `solve()`:
```cpp
if (config_.ocp_type == "stateswitch" || config_.ocp_type == "tracking_circle") {
    warm_u = makeUwarm(config_.n_shift);
    // ...
}
```

---

## Bug 5: Missing tiltQuat and RK4 simulation (MINOR)

**File:** `include/ocp/ocp_tracking_circle.hpp`

**Problem:** Cold-start rollout didn't pre-tilt the quaternion toward the predicted intercept point, and didn't actually simulate forward (RK4 call was commented out). This gave the solver a dynamically inconsistent initial guess.

**Fix:** 

1. Added helper functions from reference:
   - `tiltQuat()`: Computes quaternion tilting z-axis toward a direction
   - `calcC()`: Rotation matrix from quaternion
   - `calcOmega()`: Omega matrix for quaternion derivative
   - `xdot()`: Continuous state derivative
   - `rk4()`: RK4 integration with quaternion normalization

2. Updated cold-start rollout in `create()`:
   ```cpp
   const Eigen::Vector4d qt = tiltQuat(fw.normalized());
   Eigen::VectorXd sim = x0;
   sim.segment(6, 4) = qt;  // Apply pre-tilt

   for (int k = 0; k < HORIZON; ++k) {
       problem->setInitialControl(k, u0);
       sim = rk4(sim, u_g, th_init);  // Actual forward simulation
       // ...
   }
   ```

---

## Additional: StateMonitor tracking_circle support

**File:** `include/state_monitor.hpp`

**Fix:** Updated `hasFreshTargetState()` and `getTargetSnapshot()` to treat `tracking_circle` the same as `stateswitch` - requiring fresh target state from ROS topics.

```cpp
if (ocp_type != "stateswitch" && ocp_type != "tracking_circle") {
    return true;
}
```

---

## Summary of Files Modified

| File | Changes |
|------|---------|
| `include/ocp/ocp_tracking_circle.hpp` | Bug 1: Apply prev_U warm-start; Bug 5: tiltQuat, rk4, calcC, calcOmega helpers |
| `include/quadrotor_mpc.hpp` | Bug 3: Add circle_target and t_abs to Config |
| `include/ocp_registry.hpp` | Bug 3: Accept circle_target and t_abs in create(); pass to TrackingCircleOCP |
| `src/quadrotor_mpc.cpp` | Bug 4: Theta clamping in makeUwarm(); tracking_circle in warm-start branch; pass new params |
| `src/planner_node.cpp` | Bug 2: t_abs tracking, circle_target update, increment after execution |
| `include/state_monitor.hpp` | Add tracking_circle to target state freshness checks |

---

## Testing Recommendations

1. **Launch target_circular node:**
   ```bash
   ros2 run comando_planner target_circular --ros-args -p radius:=3.0 -p omega:=0.5
   ```

2. **Launch planner with tracking_circle OCP:**
   ```bash
   ros2 launch comando_planner planner_launch.py ocp_type:=tracking_circle
   ```

3. **Trigger OCP:**
   ```bash
   ros2 launch comando_planner ocp_launch.py ocp_type:=tracking_circle command_seq:=1
   ```

4. **Verify:**
   - `t_abs` increments in logs
   - Target position updates each solve
   - Drone tracks the circular trajectory

---

## Known Limitations

1. **Circle parameters are currently hardcoded defaults** in `solverLoop()`. These should be exposed as ROS parameters:
   - `circle_center_x`, `circle_center_y`, `circle_center_z`
   - `circle_radius` (R)
   - `circle_omega`
   - `circle_phi0`

2. **No closed-loop feedback from target state estimation** - the circle is assumed to match the `target_circular` publisher exactly. For real targets, the parameters would need to be estimated online.

3. **Terminal freeze is disabled for tracking_circle** - the drone will continue tracking until manually stopped or the RH loop is paused.
