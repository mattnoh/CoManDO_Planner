# 05 — Solver Parameter Tuning Guide

---

## 5.1 Which Solver to Use

| Scenario | Recommended solver | Reason |
|----------|--------------------|--------|
| Research / new OCP development | `alipddp` | Full access to cost/constraint structure |
| Known OCP, need speed | `acados` | SQP-RTI is faster for fixed structure OCPs |
| Online replanning at >2 Hz | `acados` | ALIPDDP rarely converges in <200ms |
| Constraint-heavy problems | `alipddp` | Better handling of SOC constraints |

Set via ROS param: `solver: "alipddp"` or `solver: "acados"`

---

## 5.2 ALIPDDP Parameters

ALIPDDP (Augmented Lagrangian Interior-Point DDP) solves the OCP by iterating
a backward-forward DDP pass with an augmented Lagrangian outer loop for constraints.

All parameters live in the `Param` struct in `ocp_registry.hpp` and are set
per-OCP from constants defined in each `ocp_*.hpp` file.

### Parameter Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `reg1_min` | double | Minimum Hessian regularisation (Quu) |
| `reg2_min` | double | Minimum value function regularisation (Vxx) |
| `mu_mul` | double | Barrier parameter reduction multiplier per outer iteration |
| `rho` | double | Initial augmented Lagrangian penalty weight |
| `rho_mul` | double | Penalty weight growth multiplier |
| `rhoT` | double | Terminal constraint penalty weight (landing only) |
| `tolerance` | double | Convergence threshold on the Lagrangian gradient |
| `max_iter` | int | Maximum inner (DDP) iterations |

### Hover OCP tuning

```cpp
SOLVER_REG1_MIN  = 1e-6    // tight regularisation — well-conditioned problem
SOLVER_REG2_MIN  = 1.0     // value function stabilisation
SOLVER_MU_MUL    = 0.1     // aggressive barrier reduction
SOLVER_RHO       = 20.0    // strong initial penalty
SOLVER_RHO_MUL   = 9.0     // fast penalty growth
SOLVER_TOLERANCE = 1e-3    // tight convergence
SOLVER_MAX_ITER  = 200
```

**Rationale:** Hover is a well-conditioned quadratic-like problem. High `rho`
and fast growth (`rho_mul=9`) forces rapid constraint satisfaction. Tight
tolerance ensures the solution is close to optimal before dispatching.

### Landing OCP tuning

```cpp
SOLVER_REG1_MIN  = 1e-6
SOLVER_REG2_MIN  = 1e-2    // looser — more nonlinearity near landing
SOLVER_MU_MUL    = 0.1
SOLVER_RHO       = 10.0    // lower initial penalty (more constraints to satisfy)
SOLVER_RHO_MUL   = 10.0
SOLVER_RHOT      = 1.0     // terminal penalty
SOLVER_TOLERANCE = 0.05    // looser — acceptable for MPC replanning
SOLVER_MAX_ITER  = 500     // more iterations — problem is harder
```

**Rationale:** Landing involves SOC constraints and the velocity jerk constraint,
making the problem more nonlinear. Looser tolerance (`0.05`) trades optimality
for speed — in an MPC loop, a good-enough solution computed faster beats
a perfect solution that arrives too late.

### Debugging divergence

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Solver returns empty trajectory | Constraint infeasibility at x0 | Check initial state is inside feasible set |
| Large oscillations in X output | Poor warm-start | Reduce γ scale or disable warm-start |
| Very slow convergence | `rho` too low | Increase `SOLVER_RHO` |
| NaN in trajectory | Regularisation failure | Increase `reg2_min` |
| Solver hits `max_iter` every call | `tolerance` too tight | Loosen `SOLVER_TOLERANCE` |

---

## 5.3 Acados Parameters

Acados uses SQP-RTI (Sequential Quadratic Programming, Real-Time Iteration).
Each call to `solve()` performs one (or more) SQP iterations rather than
solving to full convergence. This gives fast, bounded solve times.

### Config fields (`AcadosMPC::Config`)

| Field | Default | Description |
|-------|---------|-------------|
| `rti_iterations` | 1 | Number of SQP iterations per solve call |
| `n_shift` | 1 | Warm-start shift steps |
| `dt` | 0.1 | OCP step size (must match generated code) |
| `mass` | 0.027 | Drone mass for hover reference computation |

### RTI iterations

With `rti_iterations = 1` (pure RTI): each solve call does exactly one
forward-backward SQP pass. Fastest possible solve, but solution quality
depends on the warm-start being close to the optimum.

With `rti_iterations > 1`: multiple passes per call, better convergence,
slower per call but potentially fewer replanning cycles needed.

**Recommended:** start with `rti_iterations = 1` and increase only if
trajectory quality is poor.

### Warm-start behaviour (Acados)

Acados retains the primal-dual iterate between calls internally (the SQP-RTI
method is designed around this). The manual warm-start shift in `shiftWarmStart()`
overwrites `nlp_out` directly before each solve, advancing the previous solution
by `n_shift` steps.

If `has_prev_solution_ = false` (first call or after `setTerminalState()`),
no shift is applied and Acados uses its internal zero-initialisation.

---

## 5.4 Solver Rate vs OCP dt Trade-offs

```
solver_rate ↑ → more frequent replanning → tracks disturbances better
                                         → less time per solve → may not converge

solver_rate ↓ → more time per solve → better convergence
                                    → stale trajectory between solves
                                    → larger n_shift needed

ocp_dt ↓ → finer trajectory discretisation → more accurate dynamics
          → longer real-time horizon for same N → better prediction
          → N steps require more computation

ocp_dt ↑ → coarser discretisation → less accurate
          → shorter horizon for same N
          → faster solve (fewer stages)
```

**Typical working points:**

| Setup | `solver_rate` | `ocp_dt` | `N` | Use case |
|-------|--------------|----------|-----|----------|
| Hover (ALIPDDP) | 2 Hz | 0.05s | 100 | Standard hover |
| Landing (ALIPDDP) | 1–2 Hz | 0.05s | 100 | Online replanning |
| Any (Acados) | 5–10 Hz | 0.05–0.1s | 50–100 | Fast RTI |

---

## 5.5 ROS Parameters Reference

All parameters are declared in `comando_planner_node.cpp` with defaults.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ocp_type` | `"hover"` | Which OCP to use: `"hover"` or `"landing"` |
| `solver` | `"alipddp"` | Which solver: `"alipddp"` or `"acados"` |
| `mode` | `"mpc"` | `"mpc"` for receding-horizon, `"open_loop"` for single solve |
| `drone_name` | `"cf_1"` | Used for topic namespacing and log folder naming |
| `enable_logging` | `true` | Write CSV logs to `./logs/` |
| `solver_rate` | `round(1/ocp_dt)` | Hz — how often to run the solver |
| `hover_target_x/y/z` | `0, 0, 1` | Target position in metres (ENU) |

**Bridge-specific parameters:**

| Parameter | Node | Default | Description |
|-----------|------|---------|-------------|
| `control_mode` | `px4_bridge` | `"rates"` | `"rates"` or `"trajectory"` |
| `mass` | `px4_bridge` | `0.027` | Drone mass for thrust normalisation |

---

## 5.6 Log Files

When `enable_logging: true`, logs are written to:

```
./logs/<drone_name>_<ocp_type>_<mode>_<solver>_<YYYYMMDD_HHMMSS>/
├── commanded_state.csv     # what the bridge sent to the platform
├── actual_state.csv        # what /mpc/state reported
├── all_solves.csv          # every solve's full trajectory
└── published_commands.csv  # platform-native commands (CF FullState)
```

**`all_solves.csv` columns:**
`solve_num, solve_time_ms, node, t, x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, fz, mx, my, mz`

**`commanded_state.csv` columns:**
`timestamp, x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, fz, mx, my, mz`

**`actual_state.csv` columns:**
`timestamp, x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz`