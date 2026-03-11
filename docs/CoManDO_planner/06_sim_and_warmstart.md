# 06 — Simulation Bridge and Warm-Start Fix

---

## 6.1 Standalone Simulation Testing

The `sim_bridge` node allows full closed-loop MPC testing with no hardware,
no Gazebo, and no lower-level attitude controller.
It integrates the same `Quad6DOF` dynamics model used by the OCP — so the
simulation is internally consistent by construction.

### What the sim bridge replaces

```
Normal flight:
  Hardware → crazyflie_bridge → /mpc/state → solver → /mpc/command → crazyflie_bridge → Hardware

Simulation:
  sim_bridge (integrates Quad6DOF) → /mpc/state → solver → /mpc/command → sim_bridge
```

The solver node is **unchanged**. Only the bridge is swapped.

### How it works

On every replay tick (`ocp_dt` Hz), `sendPlatformCommand(x_cmd, u_cmd)` is called
by the base class replay timer. The sim bridge:

1. **Ignores `x_cmd`** — it does not track a reference; it integrates physics
2. **Applies `u_cmd`** directly to the simulated state via one RK4 step
3. **Optionally adds Gaussian noise** to the resulting state
4. **Publishes** the (noisy) state to `/mpc/state`

There is no Mellinger loop, no motor mixer, no attitude PD.
`u_cmd = [fz, Mx, My, Mz]` goes straight into the dynamics equations.
This isolates MPC solver quality completely from tracking controller quality.

---

## 6.2 Noise Modes

Noise is injected **after integration**, on every state that gets published
to `/mpc/state`. This simulates sensor noise, not process noise — matching
how the Crazyflie's MoCap + IMU estimates would appear to the solver.

### Noise profiles

| Mode | `pos` σ | `vel` σ | `att` σ | `ω` σ | Represents |
|------|---------|---------|---------|-------|------------|
| `none` | 0 | 0 | 0 | 0 | Perfect sensing — test solver only |
| `moderate` | 5 mm | 10 mm/s | 0.005 rad (~0.3°) | 10 mrad/s | Vicon/OptiTrack + onboard IMU |
| `large` | 20 mm | 50 mm/s | 0.020 rad (~1.1°) | 50 mrad/s | Degraded sensing or windy conditions |

### How noise is applied

```cpp
// Position and velocity: independent Gaussian per axis
noisy(0..2) += N(0, σ_pos)
noisy(3..5) += N(0, σ_vel)

// Attitude: perturb quaternion vector part then renormalise
// Valid for small σ_att << 1 rad (small-angle approximation)
noisy(7..9) += N(0, σ_att)   // qx, qy, qz
noisy(6..9) /= ||noisy(6..9)||  // renormalise qw-first

// Angular rates: independent Gaussian per axis
noisy(10..12) += N(0, σ_omega)
```

Quaternion noise is added to the vector part `[qx, qy, qz]` and then
renormalised. This is a first-order approximation valid when `σ_att ≪ 1 rad`.
For `moderate` mode (`σ = 0.005 rad ≈ 0.3°`) this is well within the valid range.

---

## 6.3 Running the Simulation

### Launch both nodes manually

```bash
# Terminal 1 — solver node
ros2 run comando_planner comando_planner_node \
  --ros-args \
  -p ocp_type:=hover \
  -p solver:=alipddp \
  -p mode:=mpc \
  -p solver_rate:=2 \
  -p hover_target_x:=0.0 \
  -p hover_target_y:=0.0 \
  -p hover_target_z:=1.0 \
  -p drone_name:=cf_sim \
  -p enable_logging:=true

# Terminal 2 — sim bridge
ros2 run comando_planner sim_bridge \
  --ros-args \
  -p noise_mode:=moderate \
  -p ocp_dt:=0.05 \
  -p x0_x:=0.0 \
  -p x0_y:=0.0 \
  -p x0_z:=1.0 \
  -p drone_name:=cf_sim
```

### Noise mode comparison test

```bash
# Clean baseline
ros2 run comando_planner sim_bridge --ros-args -p noise_mode:=none ...

# Realistic noise
ros2 run comando_planner sim_bridge --ros-args -p noise_mode:=moderate ...

# Stress test
ros2 run comando_planner sim_bridge --ros-args -p noise_mode:=large ...
```

### Monitor what's happening

```bash
# Watch state stream
ros2 topic echo /mpc/state

# Check solver output
ros2 topic echo /mpc/command --no-arr   # omit large arrays

# Verify replay rate
ros2 topic hz /mpc/state
```

### What to look for

| Metric | Good | Investigate |
|--------|------|-------------|
| Solver publishes at solver_rate Hz | yes | solver not receiving `/mpc/state` |
| `/mpc/state` publishes at `1/ocp_dt` Hz | yes | sim_bridge not receiving `/mpc/command` |
| Position error stays bounded | < 0.1m | warm-start divergence or constraint violation |
| Solver `success: true` | consistent | OCP infeasibility at current x0 |
| Angular rates `wx,wy,wz` | < 0.1 rad/s | constraint not enforced |

---

## 6.4 Warm-Start Fix: Background

*Based on internal report: "Receding-Horizon DDP Warm-Start Fix", June 2025.*

### The problem

The receding-horizon MPC loop re-solves the N=100 step OCP at every `DT=0.1s`
interval, feeding the current drone state as `x0` and the shifted previous
control sequence as the warm-start. Before the fix, Mode 2 (receding-horizon)
diverged immediately from Mode 1 (open-loop reference):

| Step | Mode | Mx (N·m) | Limit (N·m) | Status |
|------|------|----------|-------------|--------|
| 1 | Open-loop (correct) | 0.01113 | 0.01288 | ✓ |
| 1 | RH warm-start (broken) | 0.19519 | 0.01288 | ✗ 17× over |
| 2 | RH warm-start (broken) | 0.24630 | 0.01288 | ✗ 22× over |

The `MaxMomentConstraint` (SOC ball on `[Mx, My, Mz]`) was present in the OCP.
It was not missing — it was **numerically invisible** to the IPM solver on
warm-start resolves due to two bugs in dual variable reinitialisation.

---

## 6.5 Root Cause Analysis

### Bug A — Full dual recompute on every warm-start

The original `warmStart()` called `initSlacksFromConstraints()` unconditionally
for all N time steps. This **discarded the converged dual variables `Y[k]`**
from the previous solve and replaced them with freshly computed values
derived only from `mu` and the current slack magnitude.

The converged `Y` values from the previous solve had grown large (tight IPM
barrier) because the solver had driven `mu` to a small value over many outer
iterations. Throwing them away and recomputing `Y = mu/s` with the warm-start
`mu` (~1e-2) produced **weak duals** — the barrier walls around the SOC moment
constraint were effectively invisible. The solver found moment-violating
controls in ~10ms before the barrier had any chance to rebuild.

> **Consensus principle violated:** "Shift and reuse slack variables s(t) across
> MPC steps... Enforce positivity on slacks and (for IP) duals."

### Bug B — Wrong Jordan algebra dual initialisation for SOC blocks

For slot-0 initialisation (the one slot that genuinely needs fresh duals
because `x0` changes), the code set:

```cpp
Y[k](idx) = param.mu;   // (mu, 0, 0, ..., 0) for all SOC blocks
```

This is only correct if the SOC slack sits exactly at the apex (`s_bar = 0`).
For a general interior point `S = (s0, s_bar)`, the Jordan algebra
complementarity condition `S ∘ Y = μ · e` requires:

```
Y(idx)        =  μ · s0  / (s0² - ||s_bar||²)
Y(idx+1:idx+dim) = -μ · s_bar / (s0² - ||s_bar||²)
```

Setting `Y = (μ, 0, ..., 0)` left a nonzero complementarity residual
`S ∘ Y - μ·e` at initialisation, giving the backward pass incorrect dual
information from the very first iteration of every warm-start solve.

> **Reference:** Alizadeh & Goldfarb (2003), *Second-order cone programming*.

---

## 6.6 The Fix

### Fix 1 — Shift S, Y, e forward (primary fix)

Instead of calling `initSlacksFromConstraints()` for all N slots,
`warmStart()` now shifts the converged `S[k]`, `Y[k]`, `e[k]` arrays forward
by one step, mirroring exactly how `U` is shifted:

```cpp
// Terminal slot gets previous S[N-1]
ST = S[N-1]; YT = Y[N-1]; eT = e[N-1];

// Shift interior slots forward
for (int k = N-1; k > 0; --k) {
    if (ocp.getDimC(k)) {
        S[k] = S[k-1]; Y[k] = Y[k-1]; e[k] = e[k-1];
    }
}
```

This **preserves the tight converged duals** across the warm-start boundary.
The barrier walls around the SOC moment constraints are immediately active
on the very first backward pass of the next solve. The solver cannot find a
moment-violating solution because the penalty for doing so is already large.

Only slot 0 is freshly initialised, because it corresponds to the new `x0` state.

### Fix 2 — Jordan inverse for SOC dual init at slot 0

Slot 0 initialisation uses the proper Jordan inverse for SOC blocks:

```cpp
double denom = s0*s0 - nt*nt;  // > 0 for well-interior slack

Y[0](idx)              = param.mu * s0 / denom;
Y[0].segment(idx+1, dim-1) = -param.mu * s_blk.tail(dim-1) / denom;
```

This ensures `S[0] ∘ Y[0] = μ·e` holds exactly at initialisation —
zero complementarity residual at the new `x0`.

**Fallback when slack is at the boundary** (`s0 ≤ nt + soc_floor`):

```cpp
S[0].segment(idx, dim).setZero();
S[0](idx) = 1.0;      // unit slack at apex
Y[0].segment(idx, dim).setZero();
Y[0](idx) = param.mu; // safe fallback
```

### Fix 3 — Floor μ rather than reset

Per consensus: re-set `mu` to a moderately small value — not the tiny converged
value (which causes ill-conditioning) and not the original large value (which
wastes all prior IPM progress):

```cpp
this->param.mu  = std::max(this->param.mu,  1e-4);
this->param.muT = std::max(this->param.muT, 1e-4);
```

`rho` and `rhoT` are reset to their initial values to prevent equality-penalty
blow-up from accumulating across solves.

`lambda` and `lambdaT` are zeroed because the trajectory shift changes which
constraint corresponds to which index — carrying them over would introduce
dual inconsistency.

---

## 6.7 Full warmStart() Implementation

```cpp
// solver.tpp
template <typename Scalar>
void ALIPDDP<Scalar>::warmStart(const Eigen::VectorXd& x0,
    const std::vector<Eigen::VectorXd>& u_warm)
{
    X[0] = x0;
    const int Nu = std::min(N, static_cast<int>(u_warm.size()));
    for (int k = 0; k < Nu; ++k) U[k] = u_warm[k];

    // ── Fix 1: Shift S, Y, e forward ─────────────────────────────────────────
    // DO NOT call initSlacksFromConstraints() here.
    // That discards tight converged duals and makes the SOC barriers invisible.
    if (ocp.getDimCT()) {
        ST = S[N-1]; YT = Y[N-1]; eT = e[N-1];
    }
    for (int k = N-1; k > 0; --k) {
        if (ocp.getDimC(k)) {
            S[k] = S[k-1]; Y[k] = Y[k-1]; e[k] = e[k-1];
        }
    }

    // ── Slot 0: fresh initialisation for new x0 ───────────────────────────────
    initialRoll();
    if (ocp.getDimC(0)) {
        const double no_floor  = 0.01;
        const double soc_floor = 0.1;
        const int dim_g = ocp.getDimG(0);

        // Inequality (non-SOC) blocks: Y = mu/s, positivity enforced
        for (int i = 0; i < dim_g; ++i) {
            S[0](i) = std::max(-C[0](i), no_floor);
            Y[0](i) = this->param.mu / S[0](i);
            e[0](i) = 1.0;
        }

        // ── Fix 2: SOC blocks — Jordan inverse ────────────────────────────────
        const auto& dim_hs     = ocp.getDimHs(0);
        const auto& dim_hs_top = ocp.getDimHsTop(0);
        for (size_t h = 0; h < dim_hs.size(); ++h) {
            const int idx = dim_hs_top[h];
            const int dim = dim_hs[h];
            Eigen::VectorXd s_blk = -C[0].segment(idx, dim);
            double s0 = s_blk(0);
            double nt = (dim > 1) ? s_blk.tail(dim - 1).norm() : 0.0;

            if (s0 > nt + soc_floor) {
                // Well interior: use Jordan inverse for exact complementarity
                S[0].segment(idx, dim) = s_blk;
                double denom = s0*s0 - nt*nt;
                Y[0](idx) = this->param.mu * s0 / denom;
                if (dim > 1)
                    Y[0].segment(idx+1, dim-1) =
                        -this->param.mu * s_blk.tail(dim-1) / denom;
            } else {
                // Near boundary: fallback to safe unit slack
                S[0].segment(idx, dim).setZero();
                S[0](idx) = 1.0;
                Y[0].segment(idx, dim).setZero();
                Y[0](idx) = this->param.mu;
            }
            e[0].segment(idx, dim).setZero();
            e[0](idx) = 1.0;
        }
    }

    // ── Fix 3: penalty/barrier parameter handling ─────────────────────────────
    this->param.rho  = param_init_.rho;   // reset rho — prevent blow-up
    this->param.rhoT = param_init_.rhoT;
    this->param.mu   = std::max(this->param.mu,  1e-4);  // floor mu, don't reset
    this->param.muT  = std::max(this->param.muT, 1e-4);

    // Zero equality multipliers — trajectory shift invalidates indices
    lambdaT.setZero();
    for (int k = 0; k < N; ++k)
        if (ocp.getDimEC(k)) lambda[k].setZero();

    resetCost(); resetError(); resetRegulation();
}
```

---

## 6.8 Verification Results

After the fix, the receding-horizon trajectory is numerically identical to the
open-loop reference across all 100 steps:

| Step | OL Mx (N·m) | RH Mx (N·m) | Limit (N·m) | Status |
|------|-------------|-------------|-------------|--------|
| 0 | 0.01109136 | 0.01109136 | 0.01288 | ✓ |
| 1 | 0.01112656 | 0.01112656 | 0.01288 | ✓ |
| 2 | 0.01116677 | 0.01116677 | 0.01288 | ✓ |
| 10 | 0.01185322 | 0.01185322 | 0.01288 | ✓ |
| 50 | 0.00914563 | 0.00914563 | 0.01288 | ✓ |
| 100 | (terminal) | (terminal) | 0.01288 | ✓ |

Trajectories match to 8 decimal places. Angular rates `wx < 0.016 rad/s`
versus the pre-fix `0.050+ rad/s` from accumulated constraint violations.

---

## 6.9 Consensus Principle Summary

| Change | Principle | Effect |
|--------|-----------|--------|
| Shift `S[k]`, `Y[k]` forward instead of recomputing | "Shift and reuse slack variables s(t) across MPC steps" | Preserves tight IPM barrier; constraint enforced from first warm-start iteration |
| Jordan inverse for SOC dual init at slot 0 | "Enforce positivity on slacks and (for IP) duals" — correct complementarity (Alizadeh & Goldfarb 2003) | `S[0] ∘ Y[0] = μ·e` holds exactly; zero complementarity residual at new x0 |
| Floor `μ` rather than reset | "Re-set μ to a moderately small value, not the very small value at convergence" | Avoids barrier invisibility from tiny μ AND ill-conditioning |
| Reset `rho/rhoT` to initial | "Penalty parameters... decreased if over-penalization stalls progress" | Prevents `V_xx` numerical blow-up from accumulated equality penalty |
| Zero `lambda/lambdaT` | "Trajectory shift changes which constraint corresponds to which index" | Avoids dual inconsistency from shifted equality constraint indices |

---

## 6.10 Remaining Open Item

The warm-start divergence is fixed in simulation. The remaining hardware
failure mode is the Mellinger tracking discrepancy:

> Actual `wx` reaches ~0.295 rad/s on hardware vs. the solver's prediction
> of ~0.02 rad/s — a 15× error.

This is **unrelated to the warm-start bug** and requires separate investigation:

- Mellinger gain tuning vs. actual quadrotor inertia
- Latency between the planner (~10ms solve) and the 500 Hz Mellinger loop
- Using the DDP feedback gain `K_t` directly as the high-rate inner-loop law
  (Dantec et al. 2022, Grandia et al. 2019) rather than a separate Mellinger tracker

The `sim_bridge` node is useful precisely here: running with `noise_mode:=moderate`
lets you validate that the warm-start and MPC logic are correct before the
Mellinger tracking gap introduces confounding errors.