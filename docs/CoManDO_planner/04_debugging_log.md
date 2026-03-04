# Chapter 4: Quadrotor MPC Debugging Log – Full Timeline

This report documents every attempted fix, observed problem, and outcome in chronological order. It is meant as a reference to avoid revisiting dead ends and to provide context for ongoing work.

> **Note:** This log does not start from the very beginning of the project but will be updated as new developments arise. Some ideas may be repeated, contradicted, or reverted as understanding evolves.

---

## 1. Initial Problem – Hover Oscillations

**Problem:** Drone hovers but slowly develops growing oscillations in `wx`/`wy` (angular rates), eventually crashing. The first OCP solve is clean, instability develops over multiple MPC cycles.

**Fix (not yet flight-tested):**
- Added angular rate penalty to `StageCost` in `ocp_hover.hpp`.
- Added attitude and angular rate penalties to `TerminalCost` in `ocp_hover.hpp`.
- Made solver persistent in `quadrotor_mpc.cpp` (solver object now lives across solves).
- Added `warmStart()` method in solver (preserves `lambda`, resets inequality slacks `S`, `Y`).

**Result:** The partial fix (stage angular rate only, no terminal attitude/angrate) still oscillated and crashed. The complete fix was written but not yet tested.

---

## 2. Double-Shift Bug – Warm-Start Corruption

**Problem:** After making solver persistent, the warm-started trajectory became misaligned. The second solve's `X[1]` bore little resemblance to the first solve's `X[1]`. The drone's state diverged rapidly.

**Fix:**
- Identified that `shiftWarmStart()` was called twice per solve cycle – once in `quadrotor_mpc.cpp::setupProblem()` and again in `solve()`.
- Removed the `shiftWarmStart()` call and the dead `setInitialControl()` writes from `setupProblem()`.

**Result:** Hover worked perfectly. The drone now tracked the reference cleanly. This confirmed the double-shift was the root cause of the hover instability.

---

## 3. Landing OCP – NaN in Augmented Cost

**Problem:** The landing OCP (hard terminal equality constraint) caused `Aug_Cost = -nan` starting from the second solve. `Upt = 0` throughout, solver stuck.

**Observation:** First solve terminated with `rhoT = 10^8`. The persistent solver kept this value, causing the terminal Hessian `V_xx = 10^8 * I`, which propagated backward and overflowed `Q_uu`.

**Fix:**
- Added `param_init_` to `alipddp.h` to store initial parameter values.
- In `alipddp.tpp::init()`, copy `this->param_init_ = param`.
- In `alipddp.tpp::warmStart()`, reset `mu`, `muT`, `rho`, `rhoT` to `param_init_` values.

**Result:** The NaN disappeared – `log(rhoT)` now started at 0.0000. However, the solver still failed to converge; second solve still produced poor trajectories.

---

## 4. Persistent NaN After Penalty Reset – lambdaT Overwhelming

**Problem:** After resetting penalties, the second solve still showed `-nan` and huge constant `OptError` (~1.37e6) on iteration 1.

**Diagnosis:** The terminal dual variable `lambdaT` had grown large during the first solve (due to `rhoT` ramp). Even with `rhoT` reset, `lambdaT` persisted and dominated `V_x_terminal`, causing the backward pass to compute massive gains.

**Fix:**
- In `alipddp.tpp::warmStart()`, added `lambdaT.setZero()`. Stage `lambda` (inequality multipliers) were kept.

**Result:** User reported that zeroing `lambdaT` still crashed. Increasing `dt` from 0.05 to 0.07 helped avoid the crash but divergence remained.

---

## 5. SOC Slack Partial Reset – Hidden NaN Source

**Problem:** Even after resetting penalties and `lambdaT`, `Aug_Cost` was still `-nan` on iteration 1, while the base cost remained finite. The culprit was traced to the SOC slack variables.

**Diagnosis:** The original `warmStart()` only reset the top (scalar) element of each SOC slack to 1.0, leaving the vector components at stale values from the previous solve. If those components were large enough, the log‑barrier term `log(t² - ||x||²)` would become `log(negative)` → NaN.

**Fix:**
- In `alipddp.tpp::warmStart()`, zero the entire SOC portion of `S`, `Y`, `e` first, then set the top element to 1.0 for each SOC constraint.

**Result:** The NaN was permanently fixed. The solver now runs without numerical blowup.

---

## 6. Thrust Cone Disabled – First Solve Still Good, Replanning Diverges

**Problem:** To isolate the issue, the user temporarily disabled the thrust cone constraint in the landing OCP. The first solve trajectory looked reasonable (converged to origin), but commanded states from subsequent solves still diverged with large angular rates.

**Fix:** None – this was a diagnostic step.

**Result:** Confirmed that the problem is not solely the thrust cone, but disabling it allowed the first solve to produce a dynamically feasible trajectory (within the OCP model). The divergence persisted, indicating a warm‑start or model‑reality gap issue.

---

## 7. MinThrust Constraint Added – Standalone Converges

**Problem:** The first solve's early nodes had very low `fz` (e.g., 0.018 N at node 0) while using large lateral forces, violating the thrust cone (though the constraint was disabled). This pointed to a lack of a minimum thrust bound.

**Fix:**
- Added `MinThrustConstraint` (with `fmin = 0.08 N`) to the landing OCP in the standalone solver.
- Re‑enabled the thrust cone constraint.
- Also added a soft terminal cost (`SoftTerminalCost`) to aid convergence.

**Result:** The standalone solver converged perfectly, reaching `X[100]` at the origin with all states zero. This proved that with proper constraints and penalty resets, the solver can solve the landing problem.

---

## 8. rhoT Still Not Set in MPC Registry

**Problem:** When transferring the fixes to the MPC, the user noticed that the solver parameters in `ocp_registry.hpp` did not set `rhoT` explicitly. The `Param` struct initializes `rhoT = rho` at compile time, but if `rho` is later changed, `rhoT` stays at the default (1.0).

**Fix:**
- In `ocp_registry.hpp` for the landing case, added `p.rhoT = LandingOCP::SOLVER_RHOT;` (after setting `p.rho`).

**Result:** The terminal constraint now has proper penalty scaling from the first solve onward. This should have improved convergence, but divergence remained.

---

## 9. Constraint Start Index Mismatch – Node 0 Unconstrained

**Problem:** The standalone solver applied constraints from `i = 1` (skipping node 0). The MPC applied constraints from `i = 0`. The user changed the MPC to `i = 1` to match the standalone, hoping to replicate its convergence.

**Diagnosis:** Even with the thrust cone enabled, if constraints start at `i = 1`, the first control `U[0]` is free. The solver exploits this by dumping all initial deceleration into a single step, producing an `X[1]` with large lateral velocity but no tilt. This is dynamically infeasible for a real quadrotor, and Mellinger's attempt to track it generates the observed angular rates.

The standalone converged offline because the full trajectory was tracked over many steps, but in MPC, each replan starts from the now‑tilted drone, and the same unconstrained first step repeats the inconsistency.

**Observation from data:**
- First solve `X[1]` has `v = (-1.40, -1.30, -1.87)` while `q` remains identity and `ω ≈ 0`.
- To achieve that acceleration from rest, the drone would need `ax = -20 m/s²`, requiring lateral force `0.54 N`, which violates the thrust cone. Because node 0 is unconstrained, this is allowed in the OCP but impossible in reality.

**Fix (proposed):** Move the constraint loop to start at `i = 0` in both standalone and MPC. This forces the solver to plan a first step that respects the thrust cone and glideslope, thereby generating a feasible trajectory from the start. The warm‑start control `u0 = [0,0,m*g,0,0,0]` already satisfies all constraints, so no infeasibility is introduced.

---

## 10. Model Validation via Open-Loop Testing

**Problem:** Hover worked well, but landing exhibited strange behavior. Needed to isolate whether the issue was in the MPC feedback loop or the underlying dynamics model.

### 10.1 Diagnostic Tool: Open-Loop Test Node

Created `open_loop_test_node.cpp` to:
- Run MPC solver exactly once on initial state
- Store full open-loop trajectory X[0..N] and U[0..N-1]
- Replay trajectory step-by-step at `ocp_dt` rate
- Compare commanded vs actual state in real-time
- Log `commanded_vs_actual.csv` for offline analysis

**Interpretation:**
- If `actual ≈ commanded` → first-solve trajectory is physically correct; the issue is in receding-horizon feedback logic.
- If `actual diverges from commanded` → OCP dynamics model, mass, or frame convention is wrong; fix the model before debugging feedback.

### 10.2 Finding #1 – Large Discrepancy in w and quat

Initial open-loop tests showed significant divergence between commanded and actual angular rates (`w`) and quaternion components. This indicated the dynamics model was not accurately representing the real Crazyflie behavior.

### 10.3 Model Change – 6 Inputs → 4 Inputs

| Version | Control Input | Description |
|---------|---------------|-------------|
| **Old** | `u = [fx, fy, fz, mx, my, mz]` (6-dim) | Full body-frame force vector + torques |
| **Current** | `u = [fz_B, Mx, My, Mz]` (4-dim) | Body-z thrust only + 3-axis moments |

**Rationale:** Quadrotor rotors produce thrust only along the body-z axis. The 6-input model incorrectly assumed independent control of all three body-frame force components.

**Implementation:**
- `Quad6DOF` dynamics class in `quad_6dof_dynamics.h:17` sets `dim_u = 4`
- Thrust acts only along body-z axis: `f_B = (0, 0, fz_B)`
- Moments `M_B = [Mx, My, Mz]` control attitude
- Velocity dynamics: `v_next = v + dt * ((C(q) * f_B) / mass + gravity)`

### 10.4 Retuning

After reducing to 4 inputs, re-tuned cost weights in `ocp_hover.hpp` and `ocp_landing.hpp` for convergence:
- Stage costs: position, velocity, attitude, angular rate penalties
- Terminal costs: stronger penalties on all states
- Control costs: thrust and moment effort penalties

### 10.5 Finding #2 – qy and qz Upside Down

Open-loop tests improved significantly after the 4-input change, but quaternion components `qy` and `qz` showed inverted signs. This indicated a frame convention mismatch between the solver (ENU/FLU) and the Crazyflie firmware.

### 10.6 Frame Fix

Corrected `qy`/`qz` sign convention to match ENU/FLU frame expected by Crazyswarm2:
- ENU (East-North-Up) world frame
- FLU (Forward-Left-Up) body frame

### 10.7 Current State

| Behavior | Status |
|----------|--------|
| Hover | Works perfectly |
| Open-loop test | Matches reality closely |
| Landing | Still exhibits issues (under investigation) |

### 10.8 Key Files Referenced

| File | Purpose | Key Lines |
|------|---------|-----------|
| `src/open_loop_test_node.cpp` | Open-loop test node | Line 289: single solve |
| `quad_6dof_dynamics.h` | 4-input dynamics model | Line 17: `dim_u = 4` |
| `include/ocp_hover.hpp` | Hover OCP formulation | Line 244: uses `Quad6DOF` |
| `include/ocp_landing.hpp` | Landing OCP formulation | Line 305: uses `Quad6DOF` |
| `src/quadrotor_mpc.cpp` | MPC solver wrapper | Lines 145-230: `solve()` method |
| `include/ocp_registry.hpp` | OCP factory | Lines 52-61: `create()` function |

---

[Back to Chapter 3: How the MPC Works](03_mpc.md)
