# Chapter 3: How the MPC Works

This chapter describes the internals of `QuadrotorMPC`: how the OCP is structured in terms of x[] and u[], how initial guesses are set, how warm-starting works, and the reasoning behind each design decision.

---

## 1. Problem Structure

The MPC solves an Optimal Control Problem over a finite horizon of N steps, producing:

```
X = [ x[0], x[1], ..., x[N] ]       state trajectory   (N+1 nodes)
U = [ u[0], u[1], ..., u[N-1] ]     control trajectory (N nodes)
```

Each x[k] is the 13-dimensional state:

```
x[k] = [ px, py, pz,           position       (0:2)
          vx, vy, vz,           velocity        (3:5)
          qw, qx, qy, qz,       quaternion      (6:9)
          wx, wy, wz ]          angular rate    (10:12)
```

Each u[k] is the 6-dimensional control:

```
u[k] = [ fx, fy, fz,           body-frame force   (0:2)
          mx, my, mz ]          body-frame torque  (3:5)
```

The dynamics constraint at every step is:

```
x[k+1] = f( x[k], u[k] )
```

where f is the quadrotor rigid-body dynamics integrated over one OCP time step dt.

---

## 2. OCP Types

The OCP formulation is selected by the `ocp_type` parameter. Each type defines its own cost, constraints, horizon length N, and time step dt as static constants.

| `ocp_type` | Class | Terminal condition |
|---|---|---|
| `"hover"` | `HoverOCP` | Tracks a user-supplied 13-dim `terminal_state` |
| `"landing"` | `LandingOCP` | Terminal state embedded in the OCP |

Both N and dt are accessed from these constants directly — no runtime query of the problem object is needed.

---

## 3. What DDP Actually Reads

This is the most important thing to understand about how we set up the problem. DDP (Differential Dynamic Programming) has a specific contract with the caller:

- **x[0]** — you provide the fixed initial condition (the current measured state)
- **U = [u[0]..u[N-1]]** — you provide an initial guess for the control sequence
- **x[1..N]** — the solver computes these itself via a forward rollout before every backward pass

x[1..N] are never read from outside. Giving the solver pre-rolled states would have no effect — the first thing ALIPDDP does internally is recompute the entire state trajectory from x[0] and U.

---

## 4. Setting Up the Problem — `setupProblem(x_current)`

#### x[0] — always set to the real measured state

```cpp
problem_->setInitialState(0, current_state);
```

This is set unconditionally, every solve, before the warm/cold branching. It is the only state the solver reads from outside.

#### Warm start path

On every call after the first successful solve, `has_prev_solution_ = true`. We shift the previous control sequence forward by one step and pass it as the initial U:

```
U_warm[k]   = U_prev[k+1]    for k = 0, ..., N-2
U_warm[N-1] = U_prev[N-1]    (hold last control)
```

Then:

```cpp
for i in 0..N-1:
    problem_->setInitialControl(i, U_warm[i])
```

The solver's forward sweep will then compute x[1..N] from this U_warm and x[0].

**What was removed and why.** The original code also ran a manual forward rollout here:

```cpp
// OLD — removed:
Eigen::VectorXd x = current_state;
for (int i = 0; i < N; ++i) {
    x_next = problem_->getDynamics(i)->f(x, prev_U_[i]);
    problem_->setInitialState(i + 1, x_next);
    x = x_next;
}
```

This was redundant. DDP overwrites x[1..N] in its own forward sweep on every iteration before any backward pass runs. The manually set states had zero effect on the solution and were wasted computation. Removed.

#### Cold start path — rely on OCP defaults

On a cold start (`has_prev_solution_ == false`), `setupProblem()` does **not** overwrite controls.

This is intentional: both `HoverOCP::create()` and `LandingOCP::create()` already seed the initial control sequence with a gravity-compensating hover force based on the current quaternion, and that seed is better than a generic zero-force guess.

So the current flow is:

- create the OCP (`HoverOCP::create` or `LandingOCP::create`)
- set `x[0]` to the measured state
- only if warm-start is available, overwrite U with shifted `prev_U_`
- otherwise keep the OCP-provided control initialization

---

## 5. The prev_X_ Shift — Kept but Not Used by the Solver

`shiftWarmStart()` shifts both `prev_X_` and `prev_U_` forward by one step.

`prev_X_` is shifted but is not passed to the solver. As explained above, DDP does not accept an X seed — it always recomputes x[1..N] from U and x[0]. The shifted X is therefore kept in memory but has no effect on the current solution.

It is kept intentionally: if we later extend the solver backend or add a custom initialiser that does accept an X seed, the infrastructure to provide one is already in place.

---

## 6. Calling the Solver

```cpp
solver_.reset();
solver_ = make_shared<ALIPDDP<double>>(*problem_);
solver_->init(solver_params_);
solver_->solve();

X_result = solver_->getResX();   // [ x[0], x[1], ..., x[N] ]
U_result = solver_->getResU();   // [ u[0], u[1], ..., u[N-1] ]
```

The solver is recreated on every call. All warm-start information is injected through the initial U set on the problem object before `solve()` is called — no state is carried inside the solver itself between calls.

---

## 7. Caching the Result

```cpp
prev_X_ = X_result;   // kept for future use — not read by solver
prev_U_ = U_result;   // used as warm-start seed for the next solve
has_prev_solution_ = true;
```

The planner node uses X_result[1] as the published reference for Mellinger (see Chapter 2).

---

## 8. Terminal State Update

For the hover OCP, the terminal reference can be changed at runtime:

```cpp
mpc_->setTerminalState(new_terminal_13dim);
```

This resets `has_prev_solution_ = false` so the next solve runs cold from the hover seed toward the new target.

---

[Back to Chapter 2: Planner Node Setup](02_planner_node_setup.md)