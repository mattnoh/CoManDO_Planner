# Technical Audit: Existing Moving-Target Landing Presentation vs Current Code

This audit treats the current ALIPDDP examples and the current
`comando_planner` OCP registry as the source of truth. The existing slide
structure is not the problem. The problem is that the slides mix four different
things without clearly separating them:

- older standalone ALIPDDP examples
- current registered CoManDO OCPs
- external baselines from ALAN and CoNi-MPC
- future work and hardware plans

High-level verdict: the deck is partly outdated, not structurally wrong. Keep
the existing order, but correct the technical claims below.

## Current Source Of Truth

Current registered CoManDO OCP keys:

```text
hover
landing
stateswitch
tracking_circle_target
tracking_bodyrate_tf_noimu
tracking_bodyrate_tf_imu
tracking_bodyrate_bf_imu
tracking_bodyrate_bf_noimu
```

Important implications:

- `tracking_circle` is not a current CoManDO registry key.
- `tracking_circle_target` is open-loop only in the current node.
- Target-frame and body-frame body-rate OCPs are already integrated in CoManDO.
- `animate_poly_landing.py` is a visualization script for world-space logs, not
  an ALAN or CoNi-MPC benchmark implementation.

Primary local evidence:

- `CoManDO_planner/include/planner_core/ocp_registry.hpp`
- `CoManDO_planner/README.md`
- `CoManDO_planner/include/ocp/ocp_stateswitch.hpp`
- `CoManDO_planner/include/ocp/ocp_tracking_circle_target.hpp`
- `CoManDO_planner/include/ocp/ocp_tracking_bodyrate_tf_imu.hpp`
- `CoManDO_planner/include/ocp/ocp_tracking_bodyrate_bf_imu.hpp`
- `ALIPDDP-main/problem_examples/quad_cf_tracking_rh_landing.cpp`
- `ALIPDDP-main/benchmark/animate_poly_landing.py`
- `ALIPDDP-main/benchmark/papers/relative/ALAN/...`
- `ALIPDDP-main/benchmark/papers/relative/coni_mpc/...`

## Page 1

Current claim:

- Title slide: "Trajectory Planning for Landing on a Moving Target".

Status: accurate.

Correction:

- Only fix spelling: "seminnar" should be "seminar".

Code/paper evidence:

- No code dependency.

What to say in presentation:

- This talk is about the progression from world-frame moving-target landing to
  relative-frame and body-frame formulations.

## Page 2

Current claim:

- The problem can be organized by world/inertial frames vs relative/target
  non-inertial frames, and full-state nonlinear optimization vs
  geometric/convex drive.

Status: accurate but underdeveloped.

Correction:

- Keep the slide. Make the four categories explicit:
  world-frame full NLP, world-frame convex/SCP, target-frame NMPC, and
  target-frame geometric/flatness/QP.
- Use ALAN/Lo as the target-frame geometric/flatness example, not as an NMPC
  example.

Code/paper evidence:

- ALAN paper describes local-frame planning with flat outputs and Bezier/QP.
- CoNi-MPC code uses ACADO fixed-horizon NMPC with a non-inertial relative state.

What to say in presentation:

- "The literature splits mainly by frame choice and optimizer type. Our work
  moves through this map: first world-frame ALIPDDP, then shifted relative
  planning, then target-frame and body-frame body-rate OCPs."

## Page 3

Current claim:

- Chen 2025 is world-frame full nonlinear optimization.
- CoNi-MPC is target-frame non-inertial NMPC with ACADO.
- Shen 2024 is world-frame convexification/SCP.

Status: mostly accurate.

Correction:

- Keep CoNi as fixed-horizon ACADO NMPC. Do not imply it supports free-time
  landing optimization. In CoNi code, the horizon is fixed at 2.0 s with
  `dt = 0.1`, and the control is `[T, w_x, w_y, w_z]`.
- The wording about "pre-generated offline minimum-jerk/MINCO references" is
  plausible for the paper context, but keep it secondary. The stronger and safer
  statement is: CoNi tracks a reference window in a fixed-horizon NMPC rather
  than optimizing touchdown time with a `Theta` state/control pair.

Code/paper evidence:

- `coni_mpc/acado_model/quadrotor_model_thrustrates.cpp` defines 19 state
  variables, fixed `t_end = 2.0`, `dt = 0.1`, and controls `T, w_x, w_y, w_z`.
- `coni_mpc/src/coni_mpc/num_sim_mpc.cpp` converts world car/quad odometry into
  relative position, relative velocity, relative orientation, target acceleration,
  target angular velocity, and beta.

What to say in presentation:

- "CoNi is the clean target-frame NMPC reference. It gives us the correct
  non-inertial state structure. Our difference is that ALIPDDP also optimizes
  per-stage time and adds landing-specific cone and touchdown shaping."

## Page 4

Current claim:

- Additional references and example figures are shown.

Status: unclear.

Correction:

- Use this page to make ALAN precise. ALAN is not just "some more ideas"; it is
  the target-frame geometric/QP anchor.
- Add one concise ALAN description:
  nonrobocentric ground-sensor system, relative local frame, Bezier minimum-jerk
  trajectory, visual safe flight corridor, OSQP, PID plus feedforward tracking.

Code/paper evidence:

- ALAN paper says the trajectory is generated in local frame `N`, uses flat
  outputs, visual safe flight corridors, Bezier basis, and QP constraints.
- ALAN code `alan_landing_planning/src/traj_gen.cpp` builds Bernstein/Bezier
  constraints and solves through `osqpsolver`.

What to say in presentation:

- "ALAN is important because it shows a working nonrobocentric system, but its
  planner is a flatness/QP polynomial generator. Our work is a constrained
  nonlinear OCP with quadrotor dynamics and variable timing."

## Page 5

Current claim:

- World-frame global planning uses 14D augmented state
  `[p, v, q, omega, DT]` and 5D control `[f_z, M_x, M_y, M_z, Theta]`.
- Target prediction uses current measured target state
  `(p_meas, v_meas, a_meas)` with constant-acceleration extrapolation.

Status: partly outdated.

Correction:

- State/control and variable-time structure are accurate for the standalone
  world-frame example.
- The predictor claim is too narrow. Current `quad_cf_tracking_rh_landing.cpp`
  has a `ConstantSnapPredictor` with position, velocity, acceleration, jerk, and
  snap terms. If jerk and snap are set to zero, it reduces to constant
  acceleration, but the code is not limited to that.

Code/paper evidence:

- `quad_cf_tracking_rh_landing.cpp` defines `NX_SS = 14`, `IDX_DT = 13`, and
  target models with position, velocity, acceleration, jerk, and snap.
- `ConstantSnapPredictor` implements `predictPos`, `predictVel`,
  `predictAccel`, and `predictJerk`.

What to say in presentation:

- "This slide is the common world-frame OCP template. The key point is not the
  exact target model order; the key point is that the terminal target is
  evaluated at the optimized arrival time `DT_N`."

## Page 6

Current claim:

- `quad_cf_rh_landing.cpp` is world-frame planning.
- Terminal cost matches drone position and velocity to predicted target position
  and velocity at `DT_N`.
- Landing constraints are world-frame floor and SOC glideslope relative to the
  target.

Status: mostly accurate with one outdated detail.

Correction:

- Rename file reference to the actual standalone file if needed:
  `quad_cf_tracking_rh_landing.cpp`.
- Replace "higher derivatives set to zero -> constant-acceleration" with:
  "the current standalone predictor supports constant-snap prediction and can
  reduce to constant acceleration when jerk/snap are zero."
- Keep the terminal cost explanation.

Code/paper evidence:

- `TrackingTermCost` in `quad_cf_tracking_rh_landing.cpp` evaluates
  `pred_->predictPos(DT)` and `pred_->predictVel(DT)`.
- The terminal-cost gradient includes the `DT` chain rule through `dp_dt` and
  `dv_dt`.

What to say in presentation:

- "The optimizer is not chasing the current target position. It chooses a
  flight time and aims at where the target predictor says the target will be at
  that time."

## Page 7

Current claim:

- `quad_cf_rh_landing_shifted.cpp` places the target at the origin of planning
  coordinates and keeps the dynamics simple.

Status: outdated naming, concept mostly accurate.

Correction:

- If discussing the current CoManDO implementation, call this `stateswitch`.
- If discussing historical standalone work, name the actual historical example
  explicitly and label it as historical.
- The current CoManDO `stateswitch` does more than a pure frozen-origin shift:
  it subtracts target position and velocity, uses an OCP-owned target predictor,
  and samples predicted target acceleration at RK4 substeps.

Code/paper evidence:

- `ocp_stateswitch.hpp` transforms absolute state by subtracting target position
  and velocity.
- `Quad6DOFVarTimeRelativePred` samples target acceleration at RK4 k-points.
- The descriptor reconstructs world state by adding target position and velocity
  back for full-state command publication.

What to say in presentation:

- "This is the practical middle layer. It is not full non-inertial target-frame
  dynamics, but it is also not just a static shift anymore: the current
  state-switch OCP propagates target acceleration through the model."

## Page 8

Current claim:

- Shifted-global planning becomes stale unless replanned.
- It depends on replanning frequency and prediction quality.
- World-frame setpoints must be reconstructed for `cmd_full_state`.

Status: accurate but missing a current nuance.

Correction:

- Keep the weakness statement, but add that current `stateswitch` partially
  mitigates this by propagating target acceleration inside RK4.
- The remaining limitation is that this is still not a rotating non-inertial
  target-frame model: no target angular velocity/beta terms and no target-frame
  attitude transport.

Code/paper evidence:

- `stateswitch` has target acceleration prediction but no augmented
  `Omega_N`, `a_N`, or `beta_N` state.
- Target/body-frame OCPs add augmented IMU states.

What to say in presentation:

- "The shifted planner is good enough to test moving-target landing with current
  infrastructure. But if the target frame rotates or accelerates aggressively,
  the model is missing the real non-inertial frame terms."

## Page 9

Current claim:

- CoNi-MPC relative notation and augmented cooperative system state are shown.
- The slide includes copied menu/sidebar text and unclear equations.

Status: concept accurate, slide content unclear.

Correction:

- Keep the slide position, but clean it into a CoNi-MPC formulation audit:
  state is `[p_B^N, v_B^N, q_NB, a_N, Omega_N, beta_N]` in 19D.
- Remove equations that are malformed or copied without context.
- Do not say `a_N = 0`, `Omega_N = N beta_N`, `beta_N = 0` as general facts.
  Correct statement: in the CoNi model, `dot(a_N) = 0`,
  `dot(Omega_N) = beta_N`, and `dot(beta_N) = 0` over the prediction horizon.

Code/paper evidence:

- CoNi ACADO model defines relative position, velocity, orientation,
  `a_imu`, `omega_non`, and `beta_non` as differential states.
- CoNi dynamics set `dot(a_imu) = 0`, `dot(omega_non) = beta_non`, and
  `dot(beta_non) = 0`.

What to say in presentation:

- "This is the true target-frame non-inertial model. CoNi's important idea is
  not only subtracting target pose; it augments target IMU quantities so the MPC
  can account for frame acceleration and rotation."

## Page 10

Current claim:

- Comparison between CoNi-MPC and our target-frame implementation.
- Header says "Body Frame", but the table mostly describes target-frame
  notation.

Status: outdated and mixed.

Correction:

- Split the explanation verbally even if the slide stays in place:
  first compare CoNi with our target-frame OCP, then mention the body-frame
  version as the robocentric extension.
- Current CoManDO target-frame OCPs are:
  `tracking_bodyrate_tf_noimu`, `tracking_bodyrate_tf_imu`.
- Current CoManDO body-frame OCPs are:
  `tracking_bodyrate_bf_noimu`, `tracking_bodyrate_bf_imu`.
- Our current IMU variants use 19D physical state plus `DT`, so 20D total, and
  5D control `[T, omega_B, Theta]`.

Code/paper evidence:

- `ocp_tracking_bodyrate_tf_imu.hpp` documents 19D physical plus `DT`, target
  frame relative input, body-rate command mode, variable `DT`.
- `ocp_tracking_bodyrate_bf_imu.hpp` documents body-frame relative state
  `[p_T^B, v_rel^B, q_NB, Omega_N, a_T^B, beta_N, DT]`.

What to say in presentation:

- "Our target-frame version is closest to CoNi but adds ALIPDDP variable timing
  and landing constraints. Our body-frame version flips the relative state into
  the drone body frame so it matches body-relative sensing and `cmd_hover` style
  command output."

## Page 11

Current claim:

- Placeholder table with all categories at 100 percent.

Status: missing.

Correction:

- Keep the page slot, but replace content with a current OCP status table.

Suggested content:

| OCP | Role | CoManDO integrated | Execution | Evidence status |
| --- | --- | --- | --- | --- |
| `landing` | fixed/absolute landing baseline | yes | full-state | current registry |
| `stateswitch` | shifted relative RH planner | yes | full-state reconstructed to world | logs exist |
| `tracking_circle_target` | single-shot relative target solve | yes | open-loop full-state | open-loop only |
| `tracking_bodyrate_tf_noimu` | target-frame body-rate no-IMU baseline | yes | body-rate/cmd_hover | current registry |
| `tracking_bodyrate_tf_imu` | CoNi-style target-frame body-rate | yes | body-rate/cmd_hover | logs exist |
| `tracking_bodyrate_bf_noimu` | body-frame no-IMU baseline | yes | body-rate/cmd_hover | current registry |
| `tracking_bodyrate_bf_imu` | body-frame IMU formulation | yes | body-rate/cmd_hover | current registry |

Code/paper evidence:

- `ocp_registry.hpp` and `README.md`.

What to say in presentation:

- "This table separates implemented OCPs from historical examples and planned
  extensions. It prevents the rest of the talk from sounding like everything is
  at the same validation level."

## Page 12

Current claim:

- Another placeholder table with all categories at 100 percent.

Status: missing.

Correction:

- Keep the page slot, but replace it with the ALAN/CoNi/Ours comparison table.

Suggested content:

| Method | Frame | Optimizer | Timing | Dynamics | Command | Sensing |
| --- | --- | --- | --- | --- | --- | --- |
| ALAN | target/local non-inertial | Bezier QP/OSQP | sampled allocation | flatness, kinematic constraints | PID plus FF | ground camera/LED relative state |
| CoNi-MPC | target non-inertial | ACADO NMPC | fixed 2 s | non-inertial relative dynamics | thrust plus body rates | relative state plus target IMU |
| Ours world | world inertial | ALIPDDP | variable `Theta` | full quad dynamics | full-state | world target prediction |
| Ours stateswitch | shifted relative | ALIPDDP | variable `Theta` | quad dynamics plus target accel | full-state reconstructed | target odom/accel |
| Ours TF/BF | target/body relative | ALIPDDP | variable `Theta` | body-rate relative dynamics | cmd_hover/body-rate | relative odom plus target accel |

Code/paper evidence:

- ALAN paper/code for Bezier/QP/OSQP.
- CoNi ACADO model for fixed-horizon NMPC.
- CoManDO registry and OCP headers for current implementation.

What to say in presentation:

- "Our contribution is not simply choosing a relative frame. The difference is
  combining relative formulations with ALIPDDP variable-time constrained OCPs."

## Page 13

Current claim:

- CrazySim SITL simulations results, but content is placeholder/sidebar text.

Status: missing.

Correction:

- Keep as SITL result page.
- Use world-frame or shifted-world animation output, but label
  `animate_poly_landing.py` correctly as visualization only.
- If showing `worldplan`, `shiftedworld`, `target`, or `body` animations, say
  which OCP produced each one and whether it is standalone or CoManDO logs.

Code/paper evidence:

- `animate_poly_landing.py` loads `executed.csv` and `all_solves.csv`, then
  visualizes world-frame drone/target trajectories and solve timing.
- CoManDO logs include `stateswitch`, `tracking_circle_target`, and
  `tracking_bodyrate_tf_imu` directories.

What to say in presentation:

- "This is not an external benchmark. This is a visualization of our own solver
  output, showing executed trajectory, target trajectory, planned horizon, and
  solve time."

## Page 14

Current claim:

- CrazySim SITL simulations results, again placeholder/sidebar text.

Status: missing.

Correction:

- Use this page to compare at least two current formulations:
  `stateswitch` vs `tracking_bodyrate_tf_imu`, or world-frame vs shifted
  relative.
- If no clean metric plot is ready, make it a qualitative status slide rather
  than a fake numerical comparison.

Code/paper evidence:

- CoManDO logs include multiple stateswitch and target-frame IMU runs.

What to say in presentation:

- "The current evidence is implementation-level and simulation/log based. The
  comparison we still owe is a consistent metric table across the same target
  trajectory and same initial condition."

## Page 15

Current claim:

- Hardware Crazyflie results, but content is placeholder/sidebar text.

Status: unclear/missing.

Correction:

- Separate hardware evidence into three statuses:
  tested, replayed from bag/log, and planned.
- Do not imply body-frame hardware success unless there is a real hardware log
  or bag for that OCP.
- If the hardware bag is only for a subset such as `stateswitch`, say that.

Code/paper evidence:

- `CoManDO_planner/bags/comando_debug` exists.
- CoManDO logs include stateswitch and target-frame runs, but the slide needs to
  identify which logs are hardware vs SITL.

What to say in presentation:

- "Hardware status must be conservative. We have to say exactly which OCP was
  flown and which ones are still simulation/planned."

## Page 16

Current claim:

- Another hardware placeholder.

Status: missing.

Correction:

- Either remove this only if allowed, or keep the slot as "Limitations Before
  Hardware Generalization".
- Content should list sensing dependence, estimator assumptions, target accel
  availability, body-rate command path validation, and solver runtime.

Code/paper evidence:

- Body-rate OCPs require `/drone/body_relative_odom` or
  `/drone/target_frame_odom` plus `/target/odom` and `/target/accel`.

What to say in presentation:

- "The main missing hardware step is not another plot. It is proving that the
  relative-state and body-rate command contracts work without relying on hidden
  global-frame assumptions."

## Pages 17-18

Current claim:

- Polynomial representation and smoothness constraints.

Status: accurate for ALAN/flatness context, but disconnected from our ALIPDDP
story.

Correction:

- If these slides are meant to explain ALAN, label them explicitly as ALAN
  background.
- If they are meant to explain our solver, they are misleading because the
  current ALIPDDP OCPs are not Bezier polynomial QPs.

Code/paper evidence:

- ALAN paper uses Bezier basis, minimum jerk, affine constraints, and OSQP.
- Current CoManDO OCPs use ALIPDDP problem definitions, dynamics classes, and
  stage/terminal costs/constraints, not Bezier coefficients as decision
  variables.

What to say in presentation:

- "These polynomial slides belong to the ALAN comparison. They explain the
  baseline class we are not using, and why our OCP differs."

## Four Formulations That Must Be Kept Separate

| Formulation | Current implementation anchor | What it means | What it is not |
| --- | --- | --- | --- |
| World-frame future-target prediction | `quad_cf_tracking_rh_landing.cpp` | Optimize arrival time and target future state in world frame | Not relative-frame control |
| Shifted/state-switch relative planning | `stateswitch` | Subtract target pose/velocity, propagate target acceleration, reconstruct world commands | Not full rotating non-inertial dynamics |
| Target-frame non-inertial planning | `tracking_bodyrate_tf_imu` | CoNi-style target-frame state with target IMU augmentation and variable `Theta` | Not ALAN Bezier/QP |
| Body-frame robocentric planning | `tracking_bodyrate_bf_imu` | Express target in drone body frame for body-relative sensing and body-rate/cmd_hover output | Not yet proven hardware unless logs show it |

## What Is Missing From The Current Deck

- A current OCP registry/status table.
- A clean ALAN vs CoNi vs Ours comparison table.
- A clear validation-status table: standalone simulation, CoManDO integration,
  SITL/log replay, hardware.
- A precise statement that `animate_poly_landing.py` is visualization only.
- A conservative hardware-status slide that does not overclaim body-frame or
  target-frame results.
- A clearer distinction between shifted-world relative planning and true
  non-inertial target-frame dynamics.

## Recommended Speaker-Level Summary

Use this wording to keep the story tight without changing slide order:

"The deck follows the right progression, but some labels are stale. Page 5-6 is
world-frame future-target prediction. Page 7-8 is shifted/state-switch relative
planning, not full target-frame non-inertial dynamics. Page 9-10 should be the
CoNi-style target-frame model and our body-rate extensions. Pages 11-12 should
separate implementation status and ALAN/CoNi comparisons. Pages 13-15 should
only show results that are actually backed by logs, simulations, or bags."

