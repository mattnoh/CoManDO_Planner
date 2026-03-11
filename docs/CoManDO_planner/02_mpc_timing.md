# 02 — MPC Timing Architecture

---

## 2.1 The Core Problem

The MPC solver (ALIPDDP or Acados) is expensive — a single solve takes **50–300ms**
depending on the OCP and hardware. The Crazyflie needs commands at **20 Hz** (every 50ms).
These two rates are fundamentally incompatible if you try to run them in the same loop.

Additionally, if you publish `X[1]` immediately after each solve, the command corresponds
to the drone's predicted state **at the time you started solving**, not at the time
you actually send the command. After a 200ms solve, the drone has moved significantly —
publishing `X[1]` causes a **backward position jump** in the commanded trajectory.

CoManDO solves both problems with a dual-timer architecture.

---

## 2.2 Dual-Timer Architecture

Two completely independent timers run in separate callback groups (ROS 2 mutual exclusion):

```
┌──────────────────────────────────────────────────────────────────┐
│  comando_planner_node                                            │
│                                                                  │
│  solver_cb_group (MutuallyExclusive)                            │
│  └─ solver_timer_  @ solver_rate Hz (e.g. 1–2 Hz)              │
│       └─► callSolver(x0)                                        │
│       └─► publish MpcCommand  ──────────────────────────────►   │
│                                                               /mpc/command
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  crazyflie_bridge (or px4_bridge)                               │
│                                                                  │
│  replay_cb_group (MutuallyExclusive)                            │
│  └─ replay_timer_  @ (1/ocp_dt) Hz (e.g. 20 Hz)               │
│       └─► read mpc_traj_[replay_idx_]                           │
│       └─► apply transition blend                                │
│       └─► sendPlatformCommand(x_cmd, u_cmd)                     │
│       └─► replay_idx_++                                         │
│                                                                  │
│  sensor_cb_group (MutuallyExclusive)                            │
│  └─ pose_sub_, odom_sub_ (or px4_odom_sub_)                    │
│       └─► publishMpcState(state_13d)                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key property:** the replay timer in the bridge runs continuously regardless of
whether a new solve has arrived. It keeps sending the last known trajectory until
a fresher one replaces it, clamping at `X[N]` when the trajectory is exhausted.

---

## 2.3 Timer Rate Selection

### solver_rate

```
solver_rate  = 1 / solver_period
```

Default: `round(1.0 / ocp_dt)` — one solve per OCP step.
Recommended in practice: 1–2 Hz to give ALIPDDP enough time to converge.

Set via ROS param:
```yaml
solver_rate: 2   # Hz
```

### replay rate

```
replay_rate = 1 / ocp_dt
```

Fixed to match the OCP discretisation. For `ocp_dt = 0.05s` → 20 Hz.
This ensures that step `k` of the trajectory is dispatched exactly `k * ocp_dt`
seconds after the solve that produced it.

### n_shift

```
n_shift = round(solver_period / ocp_dt)
```

This is the number of OCP nodes that elapse between consecutive solves.
It tells the warm-start shift how many steps to advance the previous solution forward.

```
solver_rate = 2 Hz  →  solver_period = 0.5s
ocp_dt = 0.05s
n_shift = round(0.5 / 0.05) = 10
```

---

## 2.4 Solve Latency Compensation

### The Problem

```
t=0ms    Solver captures state x0, begins solving
t=0ms    ...ALIPDDP running...
t=200ms  Solver finishes. Publishes X[0..N], U[0..N-1].
t=200ms  Bridge receives MpcCommand.
```

If the bridge starts replay at `X[1]`, it commands the position the drone
**should have been at t=50ms** (one OCP step after capture), not where it is at t=200ms.
This is a 3-step backward jump in the commanded trajectory.

### The Fix: skip index

Measure the actual wall-clock solve duration and compute how many OCP steps elapsed:

```
skip = round(solve_duration_sec / ocp_dt)
skip = clamp(skip, 1, n_shift - 1)
```

Store this as `replay_start_idx` in the `MpcCommand`. The bridge starts replay
at `X[skip]` instead of `X[1]`.

**Example:**

```
ocp_dt = 0.05s
solve_duration = 0.21s
skip = round(0.21 / 0.05) = round(4.2) = 4
```

The bridge starts at `X[4]`, which is the predicted state 4×50ms = 200ms after
the capture time — matching actual elapsed time.

**Clamp rationale:**
- Lower bound `1`: never re-publish `X[0] = x0` (the drone has moved past it).
- Upper bound `n_shift - 1`: don't consume the entire new horizon before replay even starts.

### Implementation in `comando_planner_node.cpp`

```cpp
const auto t_solve_start = this->now();
SolverResult result = callSolver(x0);
const double solve_sec = (this->now() - t_solve_start).seconds();

const int skip = std::max(1,
    std::min(static_cast<int>(std::round(solve_sec / ocp_dt_)),
             std::max(1, n_shift_ - 1)));

// packed into MpcCommand:
msg->replay_start_idx = skip;
```

---

## 2.5 Timing Diagram

```
Time →   0    50   100  150  200  250  300  350  400  450  500ms
         |    |    |    |    |    |    |    |    |    |    |

Solver:  [====solving (200ms)====]           [====solving====]
         capture x0               publish    capture x0

Replay:  X[4] X[5] X[6] X[7] X[8] X[9] ... X[4] X[5] X[6]
         ↑                         ↑
         resume at skip=4          new traj arrives,
         (200ms / 50ms = 4)        blend starts (12 ticks)

CF cmd:  20Hz ─────────────────────────────────────────────►
```

**Without latency compensation:**
```
Replay:  X[1] X[2] X[3] X[4] X[5] ...
         ↑
         X[1] corresponds to t=50ms, but it's now t=200ms
         → backward jump of 3 OCP steps in commanded position
```

---

## 2.6 Open-Loop Mode

In `mode: open_loop`, the system:

1. Waits for the first state to arrive on `/mpc/state`
2. Runs exactly **one** solve from that initial state
3. Publishes the resulting `MpcCommand` once
4. The bridge replays the trajectory at `ocp_dt` Hz until exhausted, then holds `X[N]`

This mode is used for offline trajectory validation — fly the precomputed plan
without any replanning. Logging is identical to MPC mode.