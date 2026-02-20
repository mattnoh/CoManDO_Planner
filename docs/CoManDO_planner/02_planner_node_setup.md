# Chapter 2: Planner Node Setup

This chapter describes how `PlannerNode` orchestrates the MPC control loop: how the two timers interact, how `x` is assembled and passed to the solver, and how the returned trajectory is used to generate commands at a rate independent of the solver.

---

## 1. The Two-Timer Architecture

The node runs two independent timers at configurable rates:

```
solver_timer_   (solver_rate_ Hz)   → solverLoop()
control_timer_  (control_rate_ Hz)  → controlLoop()
```

The key insight is that these two rates are **decoupled**. The control loop can (and typically should) run faster than the solver so that the drone receives fresh setpoints even while the solver is computing. The control loop does not wait for a new solution — it walks forward along the last stored trajectory.

```
time ──────────────────────────────────────────────────────────►

solver  [solve #1]          [solve #2]          [solve #3]
        │                   │                   │
        ▼                   ▼                   ▼
ctrl    ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●
        t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  ...
```

Each control tick picks the trajectory node that corresponds to *now* relative to the last solve time, rather than always replaying the same node.

---

## 2. State Assembly Before the Solver

Before each solve, the node reads `current_state_` — the 13-dimensional vector assembled continuously by the two subscriber callbacks (see Chapter 1). This vector is passed directly to the solver as the initial condition:

```
x_current = current_state_
           = [ pos(0:2) | vel(3:5) | quat(6:9) | omega(10:12) ]
```

The solver is only called once both `/pose` and `/odom` have been received, enforced by the flags `pose_received_` and `odom_received_`. Until both are true, `solverLoop()` returns early.

---

## 3. Solver Loop

`solverLoop()` runs at `solver_rate_` Hz and performs the following in order:

**Step 1 — Gate check**

```
if not (pose_received_ AND odom_received_):
    return   ← wait silently
```

**Step 2 — Solve**

Pass the current state to the MPC and get back a trajectory:

```
result = mpc_->solve(x_current)
    → result.state_trajectory   = [ x[0], x[1], ..., x[N] ]
    → result.control_trajectory = [ u[0], u[1], ..., u[N-1] ]
```

**Step 3 — Store under mutex**

```
lock(traj_mutex_)
    state_traj_      ← result.state_trajectory
    control_traj_    ← result.control_trajectory
    traj_valid_      ← true
    last_solve_time_ ← now()
unlock(traj_mutex_)
```

The timestamp `last_solve_time_` is critical — it lets the control loop compute which trajectory node corresponds to the current moment.

---

## 4. Control Loop — Time-Indexed Trajectory Walking

`controlLoop()` runs at `control_rate_` Hz. Its job is to pick the right node from the stored trajectory and send it as a command.

### 4.1 Index Calculation

Rather than always sending `x[1]`, the loop computes how many OCP time steps have elapsed since the last solve and advances the index accordingly:

```
elapsed = now() − last_solve_time_

k = 1 + floor( elapsed / dt_ocp )
k = clamp( k, 1, N )
```

Where `dt_ocp` is the OCP time step (fixed per OCP type). This means:

| Elapsed time | Index sent |
|---|---|
| `0 ≤ elapsed < dt_ocp` | `x[1]` |
| `dt_ocp ≤ elapsed < 2·dt_ocp` | `x[2]` |
| `2·dt_ocp ≤ elapsed < 3·dt_ocp` | `x[3]` |
| `elapsed ≥ (N−1)·dt_ocp` | `x[N]` (hold last) |

The drone therefore tracks the planned trajectory in *real time* rather than holding the first future waypoint for the entire inter-solve window.

### 4.2 Acceleration Feedforward

At index `k`, the feedforward acceleration is the finite difference of the velocity at the next node:

```
a_ff = ( x[k+1][3:5] − x[k][3:5] ) / dt_ocp
```

If `k` is the last node, `k+1` is clamped to `k` and `a_ff = 0`.

### 4.3 Command Assembly and Publication

The full command sent to Mellinger is:

```
cmd.pose.position     ← x[k][0:2]
cmd.twist.linear      ← x[k][3:5]
cmd.pose.orientation  ← x[k][6:9]
cmd.twist.angular     ← x[k][10:12]
cmd.acc               ← a_ff
```

Published on `/<drone_name>/cmd_full_state`.

---

## 5. Startup Sequence

```
Node constructs
    ↓
Parameters loaded (drone_name, ocp_type, control_rate, solver_rate)
    ↓
MPC object created — OCP type selected, solver settings initialised
    ↓
Subscribers and publishers created
    ↓
Both timers start immediately
    ↓
solverLoop fires → pose_received_=false or odom_received_=false → returns
    ↓  (repeated until both topics arrive)
    ↓
Both flags true → is_flying_ = true → logging initialised
    ↓
First solve: no prior solution → solver runs cold
    ↓
Trajectory stored → traj_valid_ = true
    ↓
controlLoop begins sending commands
    ↓
Subsequent solver ticks → warm-started from previous solution (see Chapter 3)
```

---

## 6. Thread Safety

The trajectory buffers (`state_traj_`, `control_traj_`, `traj_valid_`, `last_solve_time_`) are written by `solverLoop` and read by `controlLoop`. Both access them under `traj_mutex_`, so there is no data race between the two timers.

`current_state_` is written by the two subscriber callbacks and read by `solverLoop`. Under the default single-threaded ROS 2 executor, callbacks are not concurrent so this is safe. With a multi-threaded executor, a separate mutex would be needed for `current_state_`.

---

[Next Chapter: How the MPC Works](03_mpc.md)