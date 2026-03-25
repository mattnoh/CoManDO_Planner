# Porting Plan: `quad_cf_tracking_rh_landing_stateswitch` → CoManDO Planner

## Corrections to the Existing Porting Guide

Before the step-by-step plan, these are confirmed bugs in `STATESWITCH_PORTING_GUIDE.md`
and `COMANDO_PLANNER_INTERFACE.md`. Address them before writing any code.

---

### BUG 1 — Critical sign error in re-basing formula (affects both docs)

The relative state is defined as:
```
x_rel = x_drone - x_target   →   x_drone = x_rel + x_target
```
The cpp confirms this at line 818:
```cpp
xq.head(3) = X[s+1].head(3) + circ.pos(t_abs);   // +, not -
```

**PORTING_GUIDE says (WRONG):**
```
pos_cmd_abs = x_target_now.pos - X_traj_rel[k][0:3]
```
**COMANDO_PLANNER_INTERFACE says (WRONG):**
```
p_cmd = p_target_now - p_rel_planned
```
**Correct formula:**
```cpp
x_cmd_abs.segment(0, 3) = target_pos_now + x_rel_planned.segment(0, 3);
x_cmd_abs.segment(3, 3) = target_vel_now + x_rel_planned.segment(3, 3);
x_cmd_abs.segment(6, 7) = x_rel_planned.segment(6, 7);   // attitude/rates unchanged
```
This error would cause the drone to fly to the mirror-image position on the opposite
side of the target. It will not be caught in simulation if the target is at the origin.

---

### BUG 2 — `GlideslopeSTC` does not exist

The porting guide invents a class called `GlideslopeSTC` (state-triggered cone). 
**The actual cpp has no such class.** The smooth sigmoid activation in lines 82–93 of 
the cpp is computed only inside `printDiag()` for diagnostic printing. It is never 
a constraint in the OCP.

The real constraint is `GlideslopeCon_rel`, a plain SOC applied on every stage:
```cpp
class GlideslopeCon_rel : public StageConstraintBase<Scalar> {
    // Hard SOC: -[-tan_gs*z, x, y] ≤ 0
    // No sigmoid, no activation logic
    ConstraintType::SOC, dim_c=3
};
```
The cpp header comment explicitly says:
> NO STATE TRIGGER — glideslope always active on all stages.

When porting, use `GlideslopeCon_rel` exactly as written (lines 350–375 of the cpp).
Do not implement GlideslopeSTC — it will change the problem.

---

### BUG 3 — K-gain purpose is misunderstood in both docs

`STATESWITCH_PORTING_GUIDE.md` and `COMANDO_PLANNER_INTERFACE.md` describe K as
a runtime correction to the `acc` field of `cmd_full_state`.

**What K actually does in the cpp (lines 450–463):**
```cpp
// K is used inside buildOCP to warm-start the initial trajectory rollout.
// u[k] = prev_U[k] + K[k] * (sim_ss[k] - prev_X[k])
// This corrects for a_tgt snapshot mismatch BEFORE the solver runs,
// not after publishing.
```
K corrects for the fact that the target acceleration used in the PREVIOUS solve
may not match the current one. It makes the warm-start trajectory consistent
with the new target state. It is a solver-input quality improvement, not a
feedback controller on the drone.

K-gain correction at publish time (as described in both docs) is an optional
enhancement but is NOT what the original code does and should be treated
as a separate, later addition.

---

### BUG 4 — `COMANDO_PLANNER_INTERFACE.md` has opposite sign convention for relative state

That doc defines: `I_P_B2 = p_target - p_drone` (target minus drone).

The cpp (line 749) defines: `x_rel.head(3) = xq.head(3) - circ.pos(t_abs)` (drone minus target).

These are opposite. Follow the cpp convention: **rel = drone − target**.

---

### BUG 5 — Variable timestep replay logic is incompatible with fixed `ocp_dt` timer

The guide's launch file sets `n_replay=7` and the replay timer fires every `TH_INIT=0.1s`.
With variable Theta ∈ [0.05, 0.2]s per step, each of the 7 executed steps has a
*different* hold duration. Using a fixed 0.1s timer will drift from the planned trajectory.

The correct approach: for each replay tick, retrieve `U[k](IDX_THETA)` as the actual
hold duration and use a deadline-based timer or time-elapsed lookup via `X[k](IDX_DT)`.

---

### BUG 6 — `sdyn_out = dyn` is assigned twice in `buildOCP`

Lines 411 and 421 of the cpp both do `sdyn_out = dyn`. This is harmless but should
not be replicated in `ocp_stateswitch.hpp`. Remove the duplicate assignment (keep
line 411, delete the one at 421).

---

---

## Phase 1 — OCP Port (no ROS yet)

### Step 1.1 — Copy `Quad6DOFVarTimeRelative` into CoManDO include path

**What to do:**
Copy `quad_6dof_dynamics_aug.h` (which contains `Quad6DOFVarTimeRelative<Scalar>`)
from the ALIPDDP `problem_maker/dynamics/` directory into
`CoManDO_planner/include/dynamics/`.

**What to watch out for:**
- Verify that `Quad6DOFVarTimeRelative<Scalar>` has a `propagate(x_rel, u, Th)`
  method. The warm-start rollout in `buildOCP` calls `dyn->propagate(sim, up, Th)`
  at line 484. If this method name differs from what `QuadrotorMPC` currently calls
  for other dynamics, add an adapter or rename consistently.
- Check whether the existing `quad_6dof_dynamics_aug.h` already lives in the
  CoManDO include path (from prior hover/landing OCP work). If so, verify it already
  includes the `VarTimeRelative` variant — the plain `Quad6DOF` and the augmented
  version may be separate templates.
- The dynamics take a `setTargetAccel(Eigen::Vector3d)` call. Confirm the method
  signature exactly matches what `buildOCP` passes (it is a const Vector3d snapshot,
  not a function pointer or callback).

---

### Step 1.2 — Create `include/ocp_stateswitch.hpp`

**What to do:**
Lift all cost classes, constraint classes, `buildOCP`, and `makeParam` directly from 
the cpp file and wrap them in `namespace StateswitchOCP`. The following elements transfer
verbatim with minimal change:

| Element | Source lines | Action |
|---|---|---|
| `TimeCost<Scalar>` | 166–197 | Copy as-is |
| `RelTermCost<Scalar>` | 210–275 | Copy as-is |
| `FminCon`, `FmaxCon`, `MomentCon` | 278–308 | Copy as-is |
| `ThetaBounds` | 310–320 | Copy as-is |
| `ZFloorCon` | 322–331 | Copy as-is |
| `VzMinCon` | 333–342 | Copy as-is |
| `GlideslopeCon_rel` | 350–375 | Copy as-is (NOT GlideslopeSTC) |
| `buildOCP` | 393–489 | Refactor into factory `create()` |
| `makeParam` | 493–499 | Move into `getSolverParams()` in registry |

**What to watch out for:**
- `buildOCP` takes a raw `CircularTarget& tgt` to get `tgt.accel(t_abs)`. When 
  porting, replace `CircularTarget` with a plain `Eigen::Vector3d target_accel`
  argument (the snapshot value). The dynamics line becomes:
  ```cpp
  dyn->setTargetAccel(target_accel);  // snapshot from planner_node
  ```
- The `create()` factory must accept `prev_U`, `prev_X`, `prev_K` for warm-starting.
  On the first (cold) solve all three are empty `{}`. The `use_feedback` guard in
  `buildOCP` (lines 443–446) already handles this correctly — keep it.
- The `makeUwarm`, `makeXshifted`, `makeKshifted` shift functions (lines 710–728 of
  the cpp) belong in `QuadrotorMPC::solve()`, NOT inside the OCP factory, because
  they depend on `NEX` (n_replay), which is a planner parameter, not an OCP parameter.
- The Jacobian in `GlideslopeCon_rel::cx` is sized `Matrix<Scalar>::Zero(3, NX_SS)` 
  (3 × 14). Make sure `NX_SS=14` is visible in scope. Do not accidentally use `NX=13`.
- `is_quaternion_in_state = false` in `makeParam()` is critical. The other OCPs
  likely set this to `true`. Setting the wrong value will break ALIPDDP's internal
  quaternion handling for this problem.

---

### Step 1.3 — Register in `ocp_registry.hpp`

**What to do:**
Add `"stateswitch"` to `getDT()`, `getSolverParams()`, and `create()`.

```cpp
// getDT: return TH_INIT as the nominal step for the replay timer
if (ocp_type == "stateswitch") return StateswitchOCP::TH_INIT;

// getSolverParams: from makeParam() in the cpp
p.reg1_min = 1e-2;  p.reg2_min = 0.5;
p.mu_mul   = 0.1;   p.rho = 10.0;  p.rhoT = 1.0;  p.rho_mul = 10.0;
p.tolerance = 1e-4; p.max_iter = 500;
p.is_quaternion_in_state = false;   // ← do not inherit from other OCPs
```

**What to watch out for:**
- The `create()` factory signature for stateswitch needs two extra arguments not
  present for hover/landing: `target_accel` and `prev_K`. Add them with defaults
  so existing callers are not broken:
  ```cpp
  std::shared_ptr<OCP> create(
      const std::string& ocp_type,
      const Eigen::VectorXd& x0,
      // ... existing args ...
      const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero(),  // NEW
      const std::vector<Eigen::MatrixXd>& prev_K = {}                  // NEW
  );
  ```
- The stateswitch OCP ignores `terminal_state` (the target is always the origin in
  relative frame). Pass it through but the factory can ignore it.

---

### Step 1.4 — Extend `QuadrotorMPC` for 14-dim state and K storage

**What to do:**
- Add `std::vector<Eigen::MatrixXd> prev_K_` field to `QuadrotorMPC`.
- Add `Eigen::Vector3d target_accel_` field for the current target acceleration snapshot.
- In `solve()`, implement `makeUwarm`, `makeXshifted`, `makeKshifted` (shift by `n_replay`
  steps, matching the cpp's `makeUwarm` at lines 710–728).
- Return `feedback_gains` in `Result` so `planner_node` can cache them for the next solve.

**What to watch out for:**
- The tail clamping in `makeUwarm` uses `N-NEX` (not `N`) as the copy boundary:
  ```cpp
  for(int i=0; i<N; ++i)
      uw[i] = (i+NEX < N-NEX) ? pU[i+NEX] : u_tail;
  ```
  Using `N` instead of `N-NEX` causes a read-past-end access on the last few indices.
- The augmented state is 14-dim but the planner's `current_state_` is 13-dim. Do NOT
  store 14-dim states in `current_state_`. The 14th element (`IDX_DT = 0.0`) is always
  appended inside the OCP factory, not in the planner state.
- Warm-start quality degrades if `prev_K_` is empty (cold solve) or if `n_replay` 
  changed between solves. Guard with:
  ```cpp
  bool use_feedback = (!prev_K_.empty() && prev_X_.size() > (size_t)N && ...);
  ```
- `solver.getResK()` must be called after `solver.solve()` to populate `prev_K_`.
  Confirm ALIPDDP exposes this method — if not, check whether it's `getK()` or stored
  in the result struct.

---

### Step 1.5 — Standalone integration test (no ROS)

**What to do:**
Write a minimal `test_stateswitch_ocp.cpp` that replicates the cpp's `main()` with
a static `CircularTarget` but calls the CoManDO `QuadrotorMPC::solve()` instead
of the inline ALIPDDP setup.

**What to watch out for:**
- Run the stationary-target case first (set `omega=0`, `R=0`). The result should
  match the existing `ocp_landing` closely since dynamics reduce to absolute frame
  when `target_accel = 0`.
- Verify that `X[N].head(3).norm()` (terminal relative position error) converges
  below 0.1 m within the cold solve.
- Check that all K-matrix shapes are `(NU_SS, NX_SS)` = `(5, 14)`. A mismatch 
  will silently produce wrong corrections in the warm-start rollout.

---

## Phase 2 — ROS Integration

### Step 2.1 — Create `include/platform/target_tracker.hpp`

**What to do:**
Implement a `TargetState` struct and two subscriptions: one to `nav_msgs/Odometry`
for position/velocity, one to `geometry_msgs/AccelStamped` for acceleration. 
Protect with its own `target_mutex_` (separate from `state_mutex_`).

**What to watch out for:**
- Keep `target_mutex_` and `state_mutex_` separate. Do not lock both simultaneously
  in the same function — that is a deadlock path. The solver loop takes a snapshot of
  both independently.
- Validate the `target_state_.valid` flag before allowing solves. If the target topic
  drops out mid-flight, the solver will use a stale (or zero) acceleration, which may
  produce an infeasible or unsafe trajectory. Add a timestamp staleness check:
  ```cpp
  bool is_fresh = (now - target_state_.timestamp).seconds() < 0.2;
  ```
- If acceleration is computed by a derivative filter (not a native sensor), the filter
  must run in the same callback group as the target odom subscription. A derivative
  filter in a different thread can introduce a lag between position and acceleration
  that corrupts the dynamics snapshot.

---

### Step 2.2 — Modify `solverLoop()` in `planner_node.cpp`

**What to do:**
Replace the existing single-state snapshot with a synced drone+target snapshot.
Compute the relative initial state before calling `QuadrotorMPC::solve()`.

```cpp
// Correct relative state construction
Eigen::VectorXd x0_rel(13);
x0_rel.segment(0, 3) = drone_state.segment(0, 3) - target_pos;   // pos_rel
x0_rel.segment(3, 3) = drone_state.segment(3, 3) - target_vel;   // vel_rel
x0_rel.segment(6, 7) = drone_state.segment(6, 7);                 // q, omega unchanged
```

Store the `target_pos`, `target_vel`, `target_accel` snapshot alongside the solve result.

**What to watch out for:**
- The drone and target states must be from the same timestamp (or as close as possible).
  If drone comes from MoCap at 100 Hz and target from a slower or asynchronous source,
  a 20 ms lag at 2 m/s target speed = 4 cm positional error in the initial condition.
  Use the freshest available snapshot within a 50 ms window, or use message_filters
  for approximate time synchronisation.
- After solve completes, store `last_solve_.solve_timestamp` using
  `std::chrono::steady_clock::now()` (wall clock), not `this->now()` (ROS clock).
  The replay tick time math must use the same clock.
- The solve may take 10–50 ms. By the time the solution is stored, the first
  `n_replay` setpoints may already be late. The replay index should skip forward:
  ```cpp
  double elapsed_ms = <wall_clock_now - solve_timestamp> in ms;
  // find k such that X[k](IDX_DT) >= elapsed_ms / 1000.0
  mpc_replay_idx_ = first_k_past_elapsed;
  ```
  This is the same logic as the cpp's `NEX` offset but done continuously.

---

### Step 2.3 — Fix `mpcReplayTick()` with correct re-basing

**What to do:**
Replace the existing fixed-index trajectory lookup with a time-elapsed IDX_DT lookup,
and apply the CORRECTED re-basing formula (BUG 1 fix):

```cpp
// 1. Find trajectory index by elapsed time
double elapsed = (steady_clock::now() - last_solve_.solve_timestamp).count() * 1e-9;
int k = 0;
while (k+1 < (int)last_solve_.X_traj_rel.size() &&
       last_solve_.X_traj_rel[k](IDX_DT) < elapsed) ++k;

// 2. Get current target state
Eigen::Vector3d tgt_pos_now, tgt_vel_now;
{ std::lock_guard<std::mutex> lk(target_mutex_);
  tgt_pos_now = target_state_.position;
  tgt_vel_now = target_state_.velocity; }

// 3. CORRECT re-basing: drone = rel + target  (NOT target - rel)
Eigen::VectorXd x_cmd_abs(13);
x_cmd_abs.segment(0, 3) = last_solve_.X_traj_rel[k].segment(0, 3) + tgt_pos_now;
x_cmd_abs.segment(3, 3) = last_solve_.X_traj_rel[k].segment(3, 3) + tgt_vel_now;
x_cmd_abs.segment(6, 7) = last_solve_.X_traj_rel[k].segment(6, 7);
```

**What to watch out for:**
- `IDX_DT = 13` is the cumulative time in the 14-dim state. The standard 13-dim
  `x_cmd_abs` does NOT include this index. Never send index 13 to `publishCommand()`.
- The replay timer fires at a fixed `ocp_dt` rate (e.g. TH_INIT = 0.1 s). But the
  actual timestep for index k is `U[k](IDX_THETA)`. If the drone firmware expects
  commands at exactly the timer rate, this is acceptable as an approximation; the
  trajectory is smooth enough. However do not use `k++` as the index increment —
  use the time-elapsed lookup above to avoid accumulating drift.
- When `k` reaches the end of the trajectory (horizon exhausted before next solve),
  hold the last valid command (do not zero or wrap). Set up a stale-trajectory safety
  monitor: if `elapsed > X_traj_rel.back()(IDX_DT) + 0.2s`, trigger a hover command.
- The terminal relative state from a good solve has `X[N].head(3).norm() < 0.1 m`.
  At the end of the horizon the drone is at (or near) the target. The last command
  in the buffer has `fz ≈ MASS * g` (hovering) so holding it is safe momentarily.

---

### Step 2.4 — Update logging in `planner_node.cpp`

**What to do:**
The CSV format must account for:
- Relative state columns (for debugging) alongside absolute command columns.
- `IDX_DT` (cumulative time) and `IDX_THETA` (per-step timestep) from the 14/5-dim
  trajectory.
- Target state snapshot columns (`tgt_x, tgt_y, tgt_z, tgt_vx, tgt_vy, tgt_vz`).

**What to watch out for:**
- Absolute position reconstruction for `all_solves.csv` must use the TARGET SNAPSHOT
  at solve time (`last_solve_.target_snapshot_pos`), NOT the current target position.
  The cpp comment at lines 503–508 is explicit:
  > "absolute reconstruction uses tgt.pos/vel at t_abs (the snapshot the solver used)"
  Using current target position in the log produces spurious velocity oscillations
  that do not reflect what the solver planned.
- Add a `cone_viol` column: `max(0, sqrt(rx²+ry²) - GS_TAN*rz)` where `rx,ry,rz`
  are relative position components. This matches the cpp's `writeCSV` and is essential
  for verifying glideslope compliance offline.

---

### Step 2.5 — Add launch file `stateswitch_landing.launch.py`

**What to do:**
Adapt the existing launch file. Key new parameters:

```python
'ocp_type':   'stateswitch',
'n_replay':   7,         # matches NEX=7 in the cpp
# Target topic remapping
'target_odom_topic':  '/target/odom',
'target_accel_topic': '/target/accel',
```

**What to watch out for:**
- `n_replay=7` with variable Theta means the ACTUAL time between solves varies between
  `7 × THL = 0.35s` and `7 × THH = 1.4s`. The solver must complete within 0.35 s
  (worst case all short steps). Profile the solve time on target hardware BEFORE
  setting `n_replay`. If avg solve time is >250 ms, reduce to `n_replay=5`.
- The replay timer is `TH_INIT = 0.1s` (10 Hz). This is deliberately conservative —
  the drone firmware can interpolate between commands at its own rate.

---

## Phase 3 — Simulation

### Step 3.1 — Gazebo target platform plugin

**What to do:**
Write a ROS2 Gazebo plugin (NOT ROS1 — the porting guide snippet uses `ros::NodeHandle`
which is ROS1 API). Publish `nav_msgs/msg/Odometry` with position and velocity, and a
separate `geometry_msgs/msg/AccelStamped` with centripetal acceleration.

**What to watch out for:**
- The porting guide's Gazebo plugin code is ROS1 (`ros::NodeHandle`, `ros::Time`,
  `odom_pub_.publish(msg)`) — rewrite using ROS2 `rclcpp::Node`.
- Publish BOTH the odom AND accel topics at ≥100 Hz. If the planner only receives
  odom and has to numerically differentiate for accel, the derivative will lag by
  one odom step — enough to degrade warm-start quality at higher target speeds.
- Test with `omega=0` (stationary target) first. The result should closely match
  the existing landing OCP behaviour.

---

### Step 3.2 — Simulation test sequence

Run these in order. Each must pass before proceeding:

1. **Stationary target, drone directly above** — matches cpp's deterministic init
   (`x0 = [0,0,5]`, target at `[0,0,1.5]`). Terminal position error < 0.1 m.

2. **Stationary target, drone offset laterally** — verify glideslope constraint keeps
   the drone inside the 60° cone throughout the trajectory.

3. **Slow circular target** (`R=2.0, omega=0.2`) — verify re-basing formula is correct
   by checking that `actual_state.csv` position tracks `commanded_state.csv` and both
   converge toward target position.

4. **Nominal circular target** (`R=2.0, omega=0.4`) — matches cpp default. Should
   achieve `d < 0.15 m` (the cpp's `CAP`) within `NRH=120` solve iterations.

5. **Edge case: solver timeout** — artificially limit `max_iter=10`. Verify the replay
   node gracefully holds last command and does not crash.

**What to watch out for:**
- The "perfect tracker" mode in the cpp (lines 788–835) forces the simulated drone
  onto the planned trajectory at each step. In Gazebo, the drone has real dynamics
  and will track with some error. Expect ~2–5 cm tracking error vs the cpp's near-zero
  residual. If error is much larger, check feedforward acceleration calculation.
- The cpp's `CAP=0.15 m` landing threshold is conservative. In hardware, landing
  detection should also require `vz_rel < 0.5 m/s` to prevent hard landings.

---

## Phase 4 — Hardware

### Step 4.1 — Target state source

**What to do:**
Decide whether to use MoCap for the target or an onboard estimator. For initial
hardware tests, MoCap for both drone and target is strongly recommended.

**What to watch out for:**
- Crazyswarm2 angular velocity is in deg/s and needs conversion (already handled in
  existing `planner_node.cpp`). Confirm the target's angular velocity (if published)
  does NOT need the same conversion — this depends on the target's driver.
- MoCap latency is typically 10–20 ms. If drone and target use the same MoCap system,
  their timestamps are synchronised and latency cancels in the relative state. If they
  use different systems or clocks, apply timestamp alignment.

---

### Step 4.2 — Safety checks before first flight

Before enabling `cmd_full_state` output, verify each of the following in a dry-run
(motors off, drone held by hand):

- [ ] `all_solves.csv` shows `pos_err` converging to < 0.2 m within 10 solve iterations
- [ ] `cone_viol` column is 0 throughout (drone stays inside 60° glideslope cone)
- [ ] `IDX_THETA` values in U trajectory stay within [0.05, 0.2] (THL, THH bounds)
- [ ] Re-basing: `commanded_state.csv` position converges toward measured target, not away
- [ ] Solve time consistently < `n_replay × THL = 0.35 s` (i.e., solver keeps up)
- [ ] No NaN or Inf in any trajectory column (indicates diverged solve)

**What to watch out for:**
- The ZFloor constraint (`-x(2) ≤ 0`) prevents the drone from going BELOW the target
  in the relative frame. If the drone starts below the target (e.g. takes off after
  the target has risen), the cold solve will be infeasible. Add a pre-flight check:
  `drone_z > target_z + 0.3 m` before enabling the planner.
- The terminal cost `RelTermCost` drives `vz_rel → vz_ref = -0.1 m/s` at landing.
  This is a soft landing in the target frame. If the target is ascending, the absolute
  drone descent rate may be near zero (gentle) or faster (target descending). Verify
  the absolute `vz_cmd` in `commanded_state.csv` stays within safe limits.

---

## Summary of Key Constants (from the cpp)

| Constant | Value | Location |
|---|---|---|
| `N` (horizon) | 30 | main(), not 100 like landing/hover |
| `NEX` (n_replay) | 7 | main() |
| `TH_INIT` | 0.1 s | main() |
| `THL` / `THH` | 0.05 / 0.2 s | main() |
| `NRH` (max RH steps) | 120 | main() |
| `CAP` (landing distance) | 0.15 m | main() |
| `FMIN` / `FMAX` | 0.08 / 0.6 N | global |
| `GS_DEG` | 60° | global |
| `VZ_LAND_MAX` | 2.5 m/s | global |
| `vz_ref` in terminal cost | -0.1 m/s | `RelTermCost` |
| `is_quaternion_in_state` | `false` | `makeParam()` |
| `NX_SS` / `NU_SS` | 14 / 5 | global |

Note: the horizon N=30 is significantly shorter than the existing OCPs (N=100).
This is intentional — time-optimal control with variable timestep can cover
equivalent physical time with fewer nodes.