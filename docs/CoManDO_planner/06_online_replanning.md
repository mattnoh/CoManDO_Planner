# Chapter 6: Online Replanning and Trajectory Replay

The planner separates solving from command replay. ALIPDDP can take longer than
one control period, so commands are sampled from the last accepted trajectory
while a new solve is computed and staged for a controlled handoff.

## Main Components

| Component | File | Role |
| --- | --- | --- |
| `PlannerNode` | `src/planner_node.cpp` | ROS timers, parameter switching, platform dispatch, logging |
| `PlannerCore` | `include/planner_core/planner_core.hpp` | ROS-free solve gating, acceptance, replay sampling, diagnostics |
| `QuadrotorMPC` | `include/planner_core/quadrotor_mpc.hpp`, `src/quadrotor_mpc.cpp` | ALIPDDP wrapper and warm-start state |
| `TrajectoryReplayer` | `include/trajectory_replayer.hpp` | Thread-safe active/pending trajectory buffer and interpolation |

## MPC Timing

In `mpc` mode:

| Timer | Period | Function |
| --- | --- | --- |
| Solver timer | 1 ms polling | Calls `solverLoop()`, which asks `PlannerCore` whether a solve may run |
| Replay timer | active OCP `dt` | Calls `mpcReplayTick()` and publishes the current replay sample |

The solver polling rate is not the solve acceptance rate. `PlannerCore` returns
`waiting_for_handoff_time` until the active plan has replayed the configured
handoff depth.

## Accepted Plan Handoff

When a solve is accepted, `PlannerCore` computes:

```text
last_replan_delay_sec = replayAdvanceTime(result.state_trajectory)
```

`replayAdvanceTime()` uses `n_replay` and the trajectory node timestamps. For
fixed-step OCPs this is approximately:

```text
n_replay * ocp_dt
```

For variable-DT OCPs it uses the cumulative DT state slot at the `n_replay`
node.

The first accepted solve becomes active immediately. Later accepted solves are
stored as pending plans. `TrajectoryReplayer::sample()` swaps the pending plan
into active use only when:

```text
activation_time_reached
and (active plan is stale or minimum replay time elapsed)
```

This avoids replacing a plan before the drone reaches the intended handoff
region.

## Interpolation

`TrajectoryReplayer` samples by elapsed wall-clock time:

1. Compute `elapsed = now - active_plan_origin_time`.
2. Find the bracketing trajectory nodes.
3. Linearly interpolate physical state and controls.
4. Renormalize the quaternion segment `[qw,qx,qy,qz]`.
5. Preserve the DT slot and trailing augmented states from the lower node.

For variable-DT OCPs, node times come from the cumulative DT state slot. For
fixed-step OCPs, node times are `k * ocp_dt`.

## Diagnostics

When a pending plan replaces the active plan, `TrajectoryReplayer` records a
handoff diagnostic:

| Value | Meaning |
| --- | --- |
| `previous_solve_num` | Replaced solve number |
| `new_solve_num` | Activated solve number |
| `active_plan_age_sec` | Age of old active plan at swap |
| `solve_latency_sec` | ALIPDDP solve duration |
| `state_jump_norm` | Norm between old and new command states at handoff |
| `control_jump_norm` | Norm between old and new controls at handoff |

The node logs these diagnostics and publishes RViz jump markers/text.

## Solve Acceptance

`PlannerCore::isSolveAcceptable()` rejects solves unless descriptor or runtime
configuration skips validation.

Generic checks:

| Check | Threshold |
| --- | --- |
| solver success and at least two trajectory states | required |
| `constraint_error` | <= `max_constraint_error` (`1.0` in default core config) |
| altitude | `z >= -0.05`, unless descriptor skips altitude validation |
| velocity norm | <= `20 m/s` |
| angular-rate norm | <= `50 rad/s` |
| thrust control | `-0.1 <= u[0] <= 50.0` |

Several relative OCPs skip generic validation because their raw state is not a
world-frame physical state.

## Stale Plan Handling

Replay samples are marked stale when:

```text
elapsed > horizon_end + 0.2
```

In MPC mode, the node warns and clamps to the terminal replay sample. After
three stale warnings, it enters hover hold and pauses the command sequence.

## Open-Loop Replay

Open-loop mode does not use the pending-plan replayer. The node:

1. Waits for required drone/target state.
2. Solves once.
3. Stores `ol_ref_X_` and `ol_ref_U_`.
4. Publishes one node per `ocp_dt`.
5. Transitions to hover hold at horizon end.

For absolute open-loop commands, optional divergence abort compares measured
world z and vz to the command. The check is skipped for relative-state modes
because raw `x[2]` is not world altitude there.

## Practical Tuning

| Symptom | Parameter or area to check |
| --- | --- |
| Handoffs are too frequent | Increase `n_replay` |
| Planner reacts too slowly | Decrease `n_replay` or shorten OCP solve time |
| Stale trajectory warnings | Check target freshness, solver success, and validation rejects |
| Visible handoff jumps | Inspect RViz handoff markers and `state_jump_norm` logs |
| Terminal solve chatter | Tune `terminal_freeze_enter_pos` and `terminal_freeze_exit_pos` |

[Back to Index](index.md)
