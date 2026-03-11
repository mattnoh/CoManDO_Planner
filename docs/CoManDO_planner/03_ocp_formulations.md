# 03 — OCP Formulations

---

## 3.1 Shared Dynamics Model

Both OCPs use the same rigid-body 6-DOF quadrotor dynamics implemented in `Quad6DOF<Scalar>`.

**Continuous-time equations of motion:**

```
ṗ = v
v̇ = (1/m) R(q) [0, 0, fz]ᵀ + g
q̇ = (1/2) q ⊗ [0, ω]
ω̇ = J⁻¹ (M - ω × Jω)
```

Where:
- `p ∈ ℝ³` — position (ENU)
- `v ∈ ℝ³` — velocity (ENU)
- `q ∈ ℝ⁴` — unit quaternion `[qw, qx, qy, qz]` (qw-first convention)
- `ω ∈ ℝ³` — body angular rates (FLU)
- `R(q)` — rotation matrix from body to world frame
- `fz` — collective thrust along body-z axis (N)
- `M ∈ ℝ³` — body moments (Nm, inertia-scaled)
- `g = [0, 0, -9.81]` m/s²

**Discretisation:** 4th-order Runge-Kutta with fixed timestep `DT`.

**Physical constants (Crazyflie 2.x):**

| Parameter | Value |
|-----------|-------|
| Mass `m` | 0.027 kg |
| `Jxx = Jyy` | 1.66×10⁻⁵ kg·m² |
| `Jzz` | 2.92×10⁻⁵ kg·m² |
| `J_scale` | 1 / 1.66×10⁻⁵ ≈ 60240 |

---

## 3.2 Hover OCP (`ocp_hover.hpp`)

**Purpose:** stabilise the quadrotor at a fixed 3D target position.

**Horizon:** N = 100 steps, DT = 0.05s → 5 second lookahead

### Stage Cost

At each stage `k = 0, ..., N-1`:

```
q(x, u) = W_pos  · ‖p - p_target‖²
         + W_vel  · ‖v - v_target‖²
         + W_att  · (‖qv‖² + (1 - qw)²)
         + W_ω    · ‖ω‖²
         + W_fz   · fz²
         + W_M    · ‖M‖²
```

The attitude error `‖qv‖² + (1 - qw)²` is the geodesic distance from
the current quaternion to the upright quaternion `[1, 0, 0, 0]`.
It equals zero when the drone is perfectly level and increases monotonically
with tilt angle.

**Stage cost weights:**

| Weight | Value | Penalises |
|--------|-------|-----------|
| `W_pos` | 15.0 | position error |
| `W_vel` | 5.0 | velocity error |
| `W_att` | 2.0 | attitude error |
| `W_ω` | 5.0 | angular rates |
| `W_fz` | 0.1 | thrust magnitude |
| `W_M` | 0.1 | moment magnitude |

### Terminal Cost

Strong penalties on all states to enforce convergence:

```
p(x) = P_pos  · ‖p - p_target‖²
      + P_vel  · ‖v - v_target‖²
      + P_att  · (‖qv‖² + (1 - qw)²)
      + P_ω    · ‖ω‖²
```

| Weight | Value |
|--------|-------|
| `P_pos` | 100.0 |
| `P_vel` | 100.0 |
| `P_att` | 200.0 |
| `P_ω` | 100.0 |

### Constraints

**Max thrust:** `fz ≤ FMAX = 1.2 N`

Only the upper bound is enforced for hover. Minimum thrust is not constrained
because hover assumes continuous flight (never near ground).

---

## 3.3 Landing OCP (`ocp_landing.hpp`)

**Purpose:** guide the quadrotor from hover altitude to the landing target
`z_ref = 0.1m` (slightly above ground to prevent post-landing thrashing).

**Horizon:** N = 100 steps, DT = 0.05s → 5 second lookahead

### Stage Cost

```
q(x, u) = ‖x - x_ref‖²_Q
          + ‖u - u_ref‖²_R
          + ‖Δu‖²_S
          + taper_w · (W_pos · ‖p - p_prev‖² + W_vel · ‖v - v_prev‖²)
```

Where:
- `x_ref = [0, 0, 0.1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]` (land at z=0.1m, upright)
- `u_ref = [mg, 0, 0, 0]` (gravity-compensating hover thrust)
- `Δu = u - u_prev` (slew-rate penalty, anchored to previous solve's control)
- `p_prev, v_prev` — corresponding node from the previous solve's trajectory

**Q matrix (diagonal):**

| State indices | Weights | Notes |
|--------------|---------|-------|
| pos [0:3] | [2, 2, 2] | modest lateral pull |
| vel [3:6] | [3, 3, 4] | stronger on vz — descent-focused |
| qw [6] | 0.1 | |
| qv [7:10] | [0.1, 0.1, 0.1] | |
| ω [10:13] | [0.05, 0.05, 0.05] | light rate damping |

**R matrix (diagonal):**

| Control index | Weight |
|--------------|--------|
| fz [0] | 1×10⁻³ |
| M [1:4] | 1×10⁻⁴ / J_scale² |

**S matrix (slew-rate, diagonal):**

| Control index | Weight |
|--------------|--------|
| fz [0] | 5×10⁻³ |
| M [1:4] | 0.1 / J_scale² |

**Terminal cost P matrix (diagonal):**

| State indices | Weights |
|--------------|---------|
| pos [0:3] | 1000 |
| vel [3:6] | 500 |
| qw [6] | 500 |
| qv [7:10] | 500 |
| ω [10:13] | 200 |

### Trajectory-Consistency Penalty

Each solve independently minimises cost to the landing target, so two
consecutive solves from slightly different initial conditions can converge
to different local optima — producing discontinuous velocity commands.

The consistency penalty anchors the new solve's trajectory to the previous
solve's trajectory (already shifted by `n_shift`):

```
penalty = W_pos · ‖p[k] - p_prev[k]‖² + W_vel · ‖v[k] - v_prev[k]‖²
```

**Current tuning:** both `W_pos = W_vel = 0` (disabled — pure QR cost).
Enabling `W_vel` removes inter-solve velocity jumps without locking the
lateral path. `W_pos` is intentionally kept at zero because a nonzero
position weight anchors the solver to the first solve's lateral path
and prevents correction toward the landing target.

**Tapering:** for the last `TAPER_NODES = 40` steps, the consistency weight
is multiplied by `W_TRAJ_TAPER_FACTOR = 0.1`. This prevents an outdated
reference from blocking the terminal landing constraint near touchdown.

### Constraints (Landing OCP)

| Constraint | Type | Description |
|-----------|------|-------------|
| Max thrust | Inequality | `fz ≤ FMAX = 0.6 N` |
| Min thrust | Inequality | `fz ≥ FMIN = 0.08 N` |
| Tilt cone | SOC | `‖qv‖ ≤ tilt_limit`, `θ_max = 60°` |
| Max moment | SOC | `‖M‖ ≤ τ_max_scaled` |
| Velocity jerk | Inequality | `‖Δv‖ ≤ J_MAX · DT` per axis |
| Glideslope | SOC | `‖p_xy‖ ≤ tan(75°) · z` (currently disabled) |

**Tilt cone derivation:**
```
tilt_limit = sqrt((1 - cos(θ_max)) / 2)
           = sqrt((1 - cos(60°)) / 2)
           = sqrt(0.25) = 0.5
```

The SOC constraint `[tilt_limit; qx; qy]` prevents the drone from tilting
beyond 60° from vertical.

**Velocity jerk constraint derivation:**
```
Δv = dt · (R(q) · [0,0,fz]/m + g)
|Δv_i| ≤ J_MAX · DT   for i ∈ {x,y,z}
J_MAX = 0.05 / DT = 1.0 m/s²
```

This limits the per-step velocity change, preventing the OCP from commanding
physically unrealisable thrust-attitude sequences that the firmware would clip.

**Note on glideslope constraint:** currently disabled (`prob->addStageConstraint` is commented out).
When enabled, it enforces `‖p_xy‖ ≤ tan(75°) · z` — forcing the drone to stay
within a 75° cone above the landing pad. Disabled because near the cone boundary
the constraint becomes infeasible given the tilt cone constraint, causing ALIPDDP to diverge.

---

## 3.4 Adding a New OCP

See `04_platform_bridges.md` § 4.5 for the step-by-step guide.

---

## 3.5 Warm-Start Strategy

### Why warm-starting matters

ALIPDDP is a local solver. Without a good initial guess, it may converge to a poor
local optimum or fail to converge within the iteration budget. A good warm-start
dramatically reduces solve time and improves trajectory quality.

### Shift warm-start

On each subsequent solve, the previous solution is shifted forward by `n_shift` steps:

```
x_warm[k] = x_prev[min(k + n_shift, N)]
u_warm[k] = u_prev[min(k + n_shift, N-1)]
```

This initialises the new solve at the predicted state `n_shift` steps ahead —
matching the temporal alignment of the new OCP initial condition.

### γ-Blend warm-start

After shifting, the warm-start control sequence is blended toward a hover cold-start
based on how much the actual state diverges from the predicted state:

```
vel_err = ‖v_actual - v_predicted‖
γ = max(0, 1 - vel_err / VEL_ERR_SCALE)

u_warm = γ · u_shifted + (1-γ) · u_hover
```

Where `VEL_ERR_SCALE = 0.3 m/s` and `u_hover = [mg, 0, 0, 0]`.

**Motivation (Zhang et al. 2023, §III-B, Theorem 1):**
After shifting, the warm-start point may lie near the boundary of the new
feasible region. Injecting it directly can block the search direction and cause
divergence. Blending toward a well-centred cold-start (hover) ensures the initial
point has small constraint residuals.

**γ behaviour:**

| `vel_err` | γ | Effect |
|-----------|---|--------|
| 0 m/s | 1.0 | pure shifted warm-start (on-trajectory) |
| 0.15 m/s | 0.5 | 50/50 mix |
| ≥ 0.3 m/s | 0.0 | pure hover cold-start (large divergence) |

`VEL_ERR_SCALE = 0.3 m/s` is chosen from the jerk constraint:
max jerk limit gives `|Δvz| ≤ 0.133 m/s` per step, and two Mellinger lag steps
gives ~0.26 m/s maximum expected velocity error during spin-up.

### Angular rate zeroing before OCP

Before passing `current_state` to the OCP, the angular rate components are zeroed:

```cpp
Eigen::VectorXd x0_ocp = current_state;
x0_ocp.segment(10, 3).setZero();
```

**Reason:** Mellinger recomputes angular rates internally from attitude PD.
Its reactive corrections are not predicted by the OCP dynamics. Feeding the
measured angular rates back as `x0` creates a state mismatch between `x0_ocp`
and `prev_X_[1]` that destabilises the warm-start shift.