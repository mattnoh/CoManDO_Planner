# Chapter 2: Planner Node Setup and Runtime Lifecycle

`src/planner_node.cpp` owns the ROS 2 lifecycle around a ROS-free planner core.
It starts paused, waits for a runtime OCP profile, and only begins commanding
when `command_seq` increases.

## Callback Groups

The node uses separate callback groups so state updates, solving, and command
replay do not block each other unnecessarily.

| Group | Members | Role |
| --- | --- | --- |
| `sensor_cb_group_` | Platform and target subscriptions | Update `StateMonitor` with drone and target state |
| `solver_cb_group_` | MPC solver timer or open-loop startup/replay timers | Run ALIPDDP and open-loop sequencing |
| `replay_cb_group_` | MPC replay timer and open-loop hold timer | Stream commands and hover holds |

For `mpc` mode, `solverLoop()` is polled every 1 ms, but `PlannerCore` rejects
new solves until the previous active plan has replayed far enough for a
consistent handoff.

For `open_loop` mode, `openLoopStartupCheck()` waits for required state/target
streams, solves once, replays the plan at `ocp_dt`, and then enters hover hold.

## Startup Sequence

The default `planner_launch.py` starts the node with:

```text
start_paused = true
command_seq = 0
ocp_type = ""
mode = ""
```

If `ocp_type` and `mode` are empty, the node still creates platform and target
subscriptions plus visualization publishers, but it does not create solve/replay
timers.

The normal sequence is:

1. Start the planner infrastructure.
2. Set `ocp_type`, `mode`, `n_replay`, target parameters, and abort/freeze
   parameters.
3. Increment `command_seq`.
4. Planner rebuilds the OCP solver and resets replay state.
5. Planner waits for required drone and target state.
6. Planner begins `mpc` or `open_loop` execution.

`ocp_launch.py` staggers parameter sets and sends `command_seq` last to avoid
starting against a partially updated profile.

## Runtime OCP Switching

The parameter callback accepts updates to:

```text
ocp_type
mode
n_replay
hover_target_x/y/z
open_loop_abort_on_divergence
open_loop_abort_max_z_error_m
open_loop_abort_max_vz_error_mps
command_seq
```

When the profile changes, the node:

1. Validates the OCP key through `OCPRegistry`.
2. Rejects invalid modes and rejects `tracking_circle_target` in `mpc`.
3. Applies descriptor defaults for `dt`, mass, command mode, state dimension,
   and relative drone odometry mode.
4. Rebuilds Crazyflie state subscriptions if the descriptor switches into or out
   of body/target-frame relative input.
5. Rebuilds the ALIPDDP wrapper and `PlannerCore`.
6. Resets all active replay state and pauses until the next `command_seq`.

When `command_seq` increases, the node captures a hover-hold state when possible,
unpauses command execution, resets the planner state for a new command, and
starts the active profile.

## Mode Behavior

### MPC

`solverLoop()` checks:

1. Mode is `mpc`.
2. Commands are not paused.
3. OCP is not `tracking_circle_target`.
4. Drone state is available.
5. Required target state is fresh.

Then it creates `PlannerCoreInput`, calls `PlannerCore::trySolve()`, accepts or
rejects the solve, publishes the accepted path, and logs `all_solves.csv`.

`mpcReplayTick()` runs at the active OCP `dt`. It samples the active trajectory,
publishes the command through the platform adapter, publishes debug markers, and
logs command/actual state rows.

### Open Loop

`openLoopStartupCheck()` waits for state and target gates, then solves once with
`callSolver()`.

For `tracking_circle_target`, open-loop startup also requires a fresh
`/target/predicted_accel` buffer and reconstructs the world-frame command
trajectory from:

```text
solve-start /target/odom + /target/predicted_accel
```

`openLoopReplayTick()` publishes one trajectory node per `ocp_dt`. At horizon
end, the planner transitions to hover hold. If
`open_loop_abort_on_divergence:=true`, the node can abort absolute-frame open
loop when measured z or vz diverges too far from the command.

## Terminal Freeze

Terminal freeze is implemented in `PlannerCore`. When enabled, a primed MPC run
stops accepting new solves near the terminal target:

```text
position error < terminal_freeze_enter_pos
and, if enabled, velocity error < terminal_freeze_enter_vel
```

The freeze releases when position error exceeds `terminal_freeze_exit_pos`.
This suppresses solver work near a stable terminal condition while replay and
hover-hold behavior continue through the node.

## State and Target Ownership

`StateMonitor` owns:

| State | Source |
| --- | --- |
| Crazyflie 13D state | `platform::crazyflie` callbacks |
| MAVROS 13D state | `platform::mavros` callback |
| Target snapshot | `platform::target_tracker` callbacks |
| Predicted acceleration buffer | `/target/predicted_accel` callback |

The planner copies snapshots under mutex before passing data into
`PlannerCore` or direct open-loop solving.

[Next Chapter: MPC, OCP Registry, and Command Modes](03_mpc.md)
