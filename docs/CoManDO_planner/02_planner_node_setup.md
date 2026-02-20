# Chapter 2: Planner Node Setup

This chapter describes how `PlannerNode` orchestrates the MPC control loop: how the state is assembled and passed to the solver, how the returned trajectory is used to generate a command, and the reasoning behind the single-timer architecture.

---

## 1. Architecture — One Timer, Not Two

The node runs a single timer:

```
solver_timer_  (solver_rate_ Hz)  →  solverLoop()
```

**What we had before and why we changed it.**

The original design had two independent timers: a `solver_timer_` that re-solved the OCP, and a `control_timer_` that ran at a potentially higher rate and published setpoints to Mellinger between solves. The control timer walked a trajectory index forward using elapsed time since the last solve:

```
idx = 1 + floor( elapsed_since_solve / ocp_dt )
cmd = x[idx]
```

The intent was that Mellinger would receive a "fresh" reference even while the solver was computing, stepping through the planned path in real time rather than holding x[1] for the entire inter-solve window.

This logic was removed because it solved a problem we do not have. Mellinger is a full-state feedback controller running at ~500 Hz on the Crazyflie firmware. It does not hold a single setpoint — it continuously computes corrections toward whatever reference it last received. There is no benefit to us walking the trajectory index; Mellinger is already doing inner-loop interpolation far faster than our solver rate. The two-timer design added a mutex (`traj_mutex_`), a timestamp (`last_solve_time_`), and threading complexity with no practical gain.

The correct and simpler design: solve → publish x[1] → done.

---

## 2. State Assembly Before the Solver

Before each solve, the node reads `current_state_` — the 13-dimensional vector assembled continuously by the two subscriber callbacks. This vector is passed directly to the solver as the initial condition:

```
x_current = current_state_
           = [ pos(0:2) | vel(3:5) | quat(6:9) | omega(10:12) ]
```

The solver is only called once both `/pose` and `/odom` have been received, enforced by `pose_received_` and `odom_received_`. Until both are true, `solverLoop()` returns immediately.

---

## 3. Solver Loop

`solverLoop()` runs at `solver_rate_` Hz. On every tick:

**Step 1 — Gate check**

```
if not (pose_received_ AND odom_received_):
    warn and return
```

**Step 2 — Solve**

```
result = mpc_->solve(x_current)
    → result.state_trajectory   = [ x[0], x[1], ..., x[N] ]
    → result.control_trajectory = [ u[0], u[1], ..., u[N-1] ]
```

**Step 3 — Publish x[1] immediately**

x[0] is the initial condition — the current measured state. x[1] is the first planned future state. We publish x[1] directly as the reference for Mellinger. The acceleration feedforward is the finite difference of velocity between x[1] and x[2]:

```
a_ff = ( x[2][3:5] − x[1][3:5] ) / dt_ocp
```

If x[1] is the last node, a_ff is zero. The command is published once per solver tick.

**Step 4 — Visualisation and logging**

The full trajectory X is published on `/<drone>/planned_trajectory` as a `nav_msgs::msg::Path`. Logging writes the actual state, commanded state (x[1]), and u[0] to CSV.

---

## 4. Parameters

| ROS Parameter | Type | Default | Effect |
|---|---|---|---|
| `drone_name` | string | `"cf_1"` | Prefixes all topic names |
| `ocp_type` | string | `"hover"` | Selects the OCP formulation |
| `solver_rate` | int | 10 | Hz — how often the OCP is solved and a command is sent |
| `enable_logging` | bool | true | Enables CSV data logging |

`control_rate` was removed along with the control timer.

---

## 5. Startup Sequence

```
Node constructs → parameters loaded → MPC created → single solver_timer_ starts
    ↓
solverLoop fires → pose_received_=false or odom_received_=false → warns, returns
    ↓  (repeated until both topics arrive)
    ↓
Both flags true → is_flying_=true → logging initialised
    ↓
First solve (cold start — hover U seed) → publish x[1] → Mellinger begins tracking
    ↓
Subsequent solves — warm-started from previous U (see Chapter 3)
```

---

## 6. Removed Members

The following members existed to support the old two-timer design and are gone:

| Member | Was used for |
|---|---|
| `control_timer_` | Running `controlLoop()` at a separate rate |
| `control_rate_` | Parameter for control timer Hz |
| `traj_mutex_` | Protecting shared trajectory between two threads |
| `last_solve_time_` | Computing elapsed time for trajectory index walking |
| `traj_valid_` | Flag indicating a trajectory was ready to read |
| `state_traj_`, `control_traj_` | Shared buffers between solver and control loops |

---

[Next Chapter: How the MPC Works](03_mpc.md)