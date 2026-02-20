# Chapter 3: How the MPC Works

This chapter describes the internals of `QuadrotorMPC`: how the OCP is formulated in terms of `x[]` and `u[]`, how the initial trajectories are set, how warm-starting works, and how the solver is called and parameterised.

---

## 1. Problem Structure

The MPC solves an **Optimal Control Problem (OCP)** over a finite horizon of `N` steps. At each call it produces:

```
X = [ x[0], x[1], x[2], ..., x[N] ]       ← state trajectory  (N+1 nodes)
U = [ u[0], u[1], u[2], ..., u[N-1] ]     ← control trajectory (N nodes)
```

Each `x[k]` is the 13-dimensional state at step `k`:

```
x[k] = [ px, py, pz,          ← position       (0:2)
          vx, vy, vz,          ← velocity        (3:5)
          qw, qx, qy, qz,      ← quaternion      (6:9)
          wx, wy, wz ]         ← angular rate    (10:12)
```

Each `u[k]` is the 6-dimensional control input at step `k`:

```
u[k] = [ fx, fy, fz,          ← body-frame force    (0:2)
          mx, my, mz ]         ← body-frame torque   (3:5)
```

The dynamics at each step enforce:

```
x[k+1] = f( x[k], u[k] )
```

where `f` is the quadrotor rigid-body dynamics integrated over one OCP time step `dt`.

---

## 2. OCP Types

The OCP formulation is selected by the `ocp_type` parameter. Each type defines its own cost function, constraints, horizon length `N`, and time step `dt`.

| `ocp_type` | Class | Terminal condition |
|---|---|---|
| `"hover"` | `HoverOCP` | Tracks a user-supplied 13-dim `terminal_state` |
| `"landing"` | `LandingOCP` | Terminal state is embedded in the OCP |

The time step `dt` is a constant defined inside each OCP class and retrieved via:

```cpp
ocp_dt_ = mpc_->getOcpDt();
```

This `dt` is then used by the control loop for trajectory indexing (see Chapter 2).

---

## 3. Setting Up the Problem — `setupProblem(x_current)`

Every solver call begins by constructing a fresh OCP instance from the current state.

### 3.1 Cold Start (No Prior Solution)

On the first call, `has_prev_solution_ = false`. The OCP is created with `x_current` as the boundary condition. The solver initialises `X` and `U` internally (typically a zero or hover guess).

```
problem = OCP::create( x_current, terminal_state )
    → x[0] fixed to x_current
    → X[1..N], U[0..N-1] initialised by the OCP
```

### 3.2 Warm Start (Subsequent Calls)

When a prior solution exists, we shift it forward by one step to give the solver a good initial guess. This is done in two stages.

**Stage 1 — Shift `X` and `U` forward by one:**

```
X_warm[k]   = X_prev[k+1]    for k = 0, 1, ..., N-1
X_warm[N]   = X_prev[N]      ← hold last node

U_warm[k]   = U_prev[k+1]    for k = 0, 1, ..., N-2
U_warm[N-1] = U_prev[N-1]    ← hold last control
```

**Stage 2 — Correct the initial condition and rollout:**

The shifted trajectory still has `X_warm[0] = X_prev[1]`, which may no longer match the real drone state. We overwrite it and roll the dynamics forward to produce a dynamically consistent initial guess:

```
x[0]   = x_current                        ← real measured state

for k = 0 to N-1:
    x[k+1] = f( x[k], U_warm[k] )         ← forward rollout under shifted controls
```

This gives the solver a warm start where every state is physically reachable from the current position, rather than a trajectory that starts from a stale estimate.

---

## 4. Calling the Solver

After the problem is set up, the solver is instantiated and run:

```
solver = ALIPDDP( problem )
solver.init( solver_params )
solver.solve()

X_result = solver.getResX()   → [ x[0], x[1], ..., x[N] ]
U_result = solver.getResU()   → [ u[0], u[1], ..., u[N-1] ]
```

The solver is **recreated on every call**. There is no state carried over inside the solver object itself — all warm-start information is injected through the initial `X` and `U` set on the `problem` before `solve()` is called.

---

## 5. Using the Solution

On a successful solve, the result is stored for the control loop and for the next warm start:

```
X_prev ← X_result
U_prev ← U_result
has_prev_solution_ ← true
```

The control loop then indexes into `X_result` (see Chapter 2, Section 4.1). The first useful command point is `x[1]`, since `x[0]` is the initial condition (the current measured state).

If the solve fails, `has_prev_solution_` is set to `false` and the next call will run cold.


## 6. What the Solver Returns in Practice

For diagnostic purposes, the first four nodes of the state trajectory are printed after each solve:

```
X[0]: x_current          ← initial condition (what we measured)
X[1]: x predicted at t+dt
X[2]: x predicted at t+2·dt
X[3]: x predicted at t+3·dt
```

`X[0]` should always match `x_current` exactly, since it is set as a hard constraint. Any deviation in `X[1]` onward reflects the OCP's planned manoeuvre.

[Back to Chapter 2: Planner Node Setup](02_planner_node_setup.md)