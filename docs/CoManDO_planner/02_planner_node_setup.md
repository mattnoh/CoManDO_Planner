# Chapter 2: Planner Node Setup

This chapter describes how `PlannerNode` orchestrates the MPC control loop: how the state is assembled, how the solver is called, and the reasoning behind the single-timer architecture.

---

## 1. Architecture — One Timer

The node runs a single timer:

```
solver_timer_  (solver_rate_ Hz)  →  solverLoop()
```

**What we had before and why we changed it.**
The original design had a `solver_timer_` that solved the OCP and a separate `control_timer_` that walked a trajectory index between solves:

```
idx = 1 + floor( elapsed_since_solve / ocp_dt )
cmd = x[idx]
```

This was removed because Mellinger is a full-state feedback controller running at ~500 Hz on the firmware — it continuously corrects toward the last reference we gave it. Walking the index from our side added mutex complexity (`traj_mutex_`, `last_solve_time_`, `traj_valid_`) with no practical gain. Mellinger's inner loop is the control timer. Removed members: `control_timer_`, `control_rate_`, `traj_mutex_`, `last_solve_time_`, `traj_valid_`, `state_traj_`, `control_traj_`.

---

## 2. The Shift Count — n_shift_

This is computed once at startup:

```cpp
n_shift_ = max(1, round( solver_period / ocp_dt_ ))
         = max(1, round( (1/solver_rate) / ocp_dt ))
```

**Why this matters.** Between two solver ticks, the drone has consumed `n_shift_` steps of the trajectory. The warm-start must shift U forward by exactly this many steps before injecting it as the initial guess. If we shift by 1 when we should shift by 2 (e.g. solver at 10 Hz, ocp_dt=50ms), the warm-start U is one step behind where the drone actually is. After several solves the reference drifts away from reality, the solver has to work harder each time to close the gap, and the published trajectory appears jagged — as if solving from cold each time. `n_shift_` is passed into `mpc_->solve()` on every call.

---

## 3. Terminal State — Why It Must Be Set Explicitly

For the hover OCP, the terminal target is a 13-dimensional state. If it is never set — or if it defaults to an all-zero vector — the `qw` component is 0, which is an invalid quaternion. `HoverOCP` receives a degenerate target and produces a zero-effort solution: the drone does not move regardless of where it starts. The fix:

```cpp
cfg.terminal_state = Eigen::VectorXd::Zero(13);
cfg.terminal_state(0) = tx;   // px
cfg.terminal_state(1) = ty;   // py
cfg.terminal_state(2) = tz;   // pz
cfg.terminal_state(6) = 1.0;  // qw = 1  (identity quaternion, upright hover)
```

The target position is exposed as ROS parameters `hover_target_x/y/z` (defaults: 0, 0, 1m).

---

## 4. State Assembly

`current_state_` is a 13-dim vector assembled from two independent callbacks. Each writes only the fields it owns:

```
index  source   field
─────────────────────────────────────────
0      /pose    position x
1      /pose    position y
2      /pose    position z
3      /odom    velocity vx  (Kalman, world frame)
4      /odom    velocity vy
5      /odom    velocity vz
6      /pose    quaternion qw
7      /pose    quaternion qx
8      /pose    quaternion qy
9      /pose    quaternion qz
10     /odom    angular rate wx  (gyro, body frame, deg/s → rad/s)
11     /odom    angular rate wy
12     /odom    angular rate wz
```

The solver is gated on both `pose_received_` and `odom_received_` being true. After that, `current_state_` is read at the moment the solver timer fires — both topics run at 100 Hz so the data is always fresh at ≤10 Hz solver rate.

---

## 5. Solver Loop

On every tick:

**Step 1 — Gate check:** return if pose or odom not yet received.

**Step 2 — Solve:**
```
result = mpc_->solve(current_state_, n_shift_)
    → result.state_trajectory   = [ x[0], x[1], ..., x[N] ]
    → result.control_trajectory = [ u[0], u[1], ..., u[N-1] ]
```

**Step 3 — Publish x[1]:** x[0] is the initial condition (current measured state). x[1] is the first planned future state. Published once per solve tick as the Mellinger reference.

**Acceleration feedforward:** currently disabled (zero). The formula `a_ff = (x[2][3:5] − x[1][3:5]) / ocp_dt` gives net world-frame acceleration including gravity, which double-counts gravity in Mellinger's thrust computation. Re-enable only after verifying the frame convention expected by the firmware.

**Step 4 — Visualisation and logging:** full trajectory X published on `/<drone>/planned_trajectory`; actual state, commanded state (x[1]), and u[0] logged to CSV.

---

## 6. Parameters

| ROS Parameter | Type | Default | Effect |
|---|---|---|---|
| `drone_name` | string | `"cf_1"` | Prefixes all topic names |
| `ocp_type` | string | `"hover"` | Selects the OCP formulation |
| `solver_rate` | int | 10 | Hz — solve and publish rate |
| `hover_target_x/y/z` | double | 0/0/1 | Hover terminal position (m) |
| `enable_logging` | bool | true | Enables CSV data logging |

`control_rate` was removed along with the control timer.

---

## 7. Required Header Changes

`quadrotor_mpc.hpp` needs:

```cpp
// Updated signatures
Result solve(const Eigen::VectorXd& current_state, int n_shift = 1);

// New member
bool need_problem_rebuild_ = true;   // set to true by setTerminalState()
```

`shiftWarmStart(int n_shift)` now takes a shift count argument instead of always shifting by 1.

---

[Next Chapter: How the MPC Works](03_mpc.md)