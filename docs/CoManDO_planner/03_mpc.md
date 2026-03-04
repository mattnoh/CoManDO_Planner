# Chapter 3: How the MPC Works

This chapter describes the internals of `QuadrotorMPC`: state/control layout, how the receding horizon warm-start works, why the problem is reused rather than recreated, and the reasoning behind each design decision.

---

## 1. Problem Structure

The MPC solves an Optimal Control Problem over a finite horizon of N steps, producing:

```
X = [ x[0], x[1], ..., x[N] ]       state trajectory   (N+1 nodes)
U = [ u[0], u[1], ..., u[N-1] ]     control trajectory (N nodes)
```

Each `x[k]` is the 13-dimensional state:

```
x[k] = [ px, py, pz,           position       (0:2)
          vx, vy, vz,           velocity        (3:5)
          qw, qx, qy, qz,       quaternion      (6:9)
          wx, wy, wz ]          angular rate    (10:12)
```

Each `u[k]` is the 4-dimensional control:

```
u[k] = [ fz_B, body-z thrust (index 0)
Mx, My, Mz ] body-frame moments (indices 1:3)
```

> **Note:** Previous versions used 6 inputs `[fx, fy, fz, mx, my, mz]` representing full body-frame forces and torques. This was reduced to 4 inputs because quadrotor thrust acts only along the body-z axis — rotors cannot produce lateral forces directly. See [Chapter 4: Debugging Log](04_debugging_log.md#10-model-validation-via-open-loop-testing) for the rationale and validation process.
u[k] = [ fx, fy, fz,           body-frame force   (0:2)
          mx, my, mz ]          body-frame torque  (3:5)
```

The dynamics constraint at every step:

```
x[k+1] = f( x[k], u[k] )
```

where `f` is the quadrotor rigid-body dynamics integrated over one OCP time step `dt`.

---

## 2. OCP Types

| `ocp_type` | Class | Terminal condition |
|---|---|---|
| `"hover"` | `HoverOCP` | Tracks a user-supplied 13-dim `terminal_state` |
| `"landing"` | `LandingOCP` | Terminal constraint embedded in the OCP |

`dt` and `HORIZON` are static constants on each class (e.g. `HoverOCP::DT`, `HoverOCP::HORIZON`).

---

## 3. What DDP Actually Reads

DDP's contract with the caller:

- **x[0]** — fixed initial condition (current measured state). You provide this.
- **U = [u[0]..u[N-1]]** — initial guess for the control sequence. You provide this.
- **x[1..N]** — computed by the solver's own forward rollout. These are always overwritten before the first backward pass regardless of what you set.

x[1..N] passed from outside have no effect. The first thing ALIPDDP does is recompute the entire state trajectory from x[0] and U.

---

## 4. Receding Horizon: The Shift Count

At time `t`, the solver returns the optimal U for the window `[t, t+N*dt]`. The solver fires again at `t + solver_period`. Between those two ticks, the drone has consumed:

```
n_shift = round( solver_period / ocp_dt )
```

steps of the trajectory. The correct warm-start for the new window `[t+solver_period, t+solver_period+N*dt]` is U shifted forward by `n_shift`:

```
U_warm[k] = U_prev[k + n_shift]    for k = 0 .. N-1-n_shift
U_warm[k] = U_prev[N-1]            for k = N-n_shift .. N-1   (hold last)
```

**What went wrong before.** `shiftWarmStart()` always shifted by exactly 1 regardless of `solver_period`. If `solver_period = 100ms` and `ocp_dt = 50ms`, the correct shift is 2. Shifting by 1 left the warm-start one step behind the drone's actual position in the trajectory. After a few solves the reference drifted away from reality, forcing the solver to work increasingly hard to close the gap — and the published trajectory appeared jagged, as if solving from cold each time.

---

## 5. Problem Reuse vs Recreation

### What `create()` does

`HoverOCP::create(current_state, terminal_state)` and `LandingOCP::create(current_state)` both accept `current_state` as an argument. Inside, they:

1. Set `x[0] = current_state` on the problem
2. Seed U with a gravity-compensating hover from the current quaternion: `u0 = [q^{-1} * [0,0,m*g], 0,0,0]`
3. Possibly build a reference trajectory interpolated from `current_state` to `terminal_state` for running costs

Point 3 is the critical one. If `create()` re-interpolates the reference from the current measured state on every call, calling it every solve creates a **different cost landscape every tick**. The warm-start U that was optimal for the previous landscape is a poor (or irrelevant) seed for the new one. The solver effectively cold-starts every tick — which is exactly the "jagged path" symptom.

### What we do now

`create()` is called **only when the problem genuinely needs to change**:

- First solve ever (cold start, `problem_ == nullptr`)
- Terminal state changed (`setTerminalState()` sets `need_problem_rebuild_ = true`)

On all other solves we **reuse `problem_`** and only update:
1. `x[0]` via `setInitialState(0, current_state)` — the solver's initial condition
2. `U` via `setInitialControl(i, prev_U_[i])` after shifting — the warm-start seed

The cost landscape is identical to the previous solve, so the warm-start U is a valid and strong initial guess.

```
cold start (first call or terminal changed):
    problem_ = create(current_state, terminal_state)
    → U seeded by create() with gravity hover
    → x[0] set by create()

warm start (all subsequent calls):
    problem_->setInitialState(0, current_state)
    shift prev_U_ forward by n_shift
    problem_->setInitialControl(i, prev_U_[i]) for i in 0..N-1
    → cost landscape unchanged from previous solve
    → warm-start U is a good initial guess
```

### Solver recreation

Even though `problem_` is reused, the solver is still recreated on every call:

```cpp
solver_.reset();
solver_ = make_shared<ALIPDDP<double>>(*problem_);
solver_->init(solver_params_);
solver_->solve();
```

This is because ALIPDDP stores internal workspace allocated in its constructor. Recreating the solver but passing it the same (updated) `problem_` object is safe and ensures no stale workspace state from the previous iteration.

---

## 6. The `prev_X_` Shift — Kept, Not Used by Solver

`shiftWarmStart()` shifts both `prev_X_` and `prev_U_` forward by `n_shift` steps.

`prev_X_` is shifted but is **not passed to the solver**. DDP recomputes x[1..N] in its own forward rollout — any states set from outside are immediately overwritten. `prev_X_` is kept in memory for potential future use (e.g. a custom initialiser that does accept an X seed). It has no effect on the current solution.

---

## 7. Cold Start Path

On the first solve (or after `setTerminalState()`), `create()` seeds U with the gravity-compensating hover:

```
f0 = q_current^{-1} * [0, 0, m*g]   (rotate gravity to body frame)
u0 = [f0, 0, 0, 0]                   (no torques)
U = [u0, u0, ..., u0]                (N copies)
```

This is correct and already uses the real current quaternion — better than any seed we could construct externally. No explicit cold-start code is needed in `setupProblem()`.

---

## 8. Calling the Solver

```
solve(current_state, n_shift):
    1. setupProblem(current_state, n_shift)
       - if cold start: create() → problem_ set, U seeded by create()
       - if warm start: setInitialState(0, x_current)
                        shiftWarmStart(n_shift) → prev_U_ shifted
                        setInitialControl(i, prev_U_[i])
    2. solver_.reset()
       solver_ = ALIPDDP(*problem_)
       solver_.init(params)
       solver_.solve()
    3. X_result = solver_.getResX()   → [x[0], x[1], ..., x[N]]
       U_result = solver_.getResU()   → [u[0], u[1], ..., u[N-1]]
    4. cache prev_X_ = X_result  (for future use, not read by solver)
              prev_U_ = U_result  (warm-start seed for next solve)
              has_prev_solution_ = true
    5. return X_result[1] as next_state, full X and U in result
```

---

## 9. Terminal State Update

```cpp
mpc_->setTerminalState(new_terminal_13dim);
```

This sets `has_prev_solution_ = false` and `need_problem_rebuild_ = true`. The next solve calls `create()` with the new terminal, rebuilding the cost, then runs cold from the hover seed. Subsequent solves reuse the new problem.

---

## 10. Required Header Changes

`quadrotor_mpc.hpp` needs these additions:

```cpp
// In public:
Result solve(const Eigen::VectorXd& current_state, int n_shift = 1);

// In private:
void shiftWarmStart(int n_shift);
void setupProblem(const Eigen::VectorXd& current_state, int n_shift);
bool need_problem_rebuild_ = true;
```

---

[Back to Chapter 2: Planner Node Setup](02_planner_node_setup.md)