# Stateswitch State Flow

This document explains how state is passed around for the `stateswitch` OCP in CoManDO_planner.

## State Vector Convention

Planner state uses 13 elements:

1. `x, y, z`
2. `vx, vy, vz`
3. `qw, qx, qy, qz`
4. `wx, wy, wz`

`stateswitch` internally uses an augmented 14-state with `dt` as index 13 during optimization, but planner I/O remains 13-state.

## Inputs and Topics

### Drone state input

Configured by planner parameters:

- `drone_state_is_relative` (bool)
- `drone_odom_topic` (string)

Behavior:

- If `drone_odom_topic` is empty (default), the planner reads platform-default drone state topics.
- If `drone_odom_topic` is non-empty (Crazyflie path), planner subscribes to that odometry topic directly and builds the 13-state from the odometry message.

### Target input

Target snapshot is always tracked by the target tracker:

- target odometry topic (position + velocity)
- target acceleration topic

Snapshot fields used by planner:

- `position`
- `velocity`
- `acceleration`
- `valid` freshness flag

## Solve Path (MPC Loop)

At each solve tick (`solverLoop`):

1. Planner reads current drone state `x0_abs` from `StateMonitor`.
2. Planner reads `TargetSnapshot`.
3. Planner decides whether to apply the registry transform:
   - `apply_registry_transform = (descriptor has transform_state) AND (drone_state_is_relative == false)`
4. If `apply_registry_transform` is true, registry converts to relative:
   - `x_rel.pos = x_drone.pos - x_target.pos`
   - `x_rel.vel = x_drone.vel - x_target.vel`
   - attitude/rates copied through
5. If `apply_registry_transform` is false, planner passes state through unchanged.
6. Planner sets target acceleration sent to OCP:
   - if target snapshot is valid: use target acceleration
   - else: use zero vector
7. `QuadrotorMPC` calls `OCPRegistry::create(...)` and then `StateswitchOCP::create(...)` with that state.

## One-Switch Meaning

`drone_state_is_relative` is the switch controlling subtraction location:

- `false`: planner assumes drone state is absolute and performs subtraction in registry transform.
- `true`: planner assumes drone state is already relative and does not subtract.

So OCP factory contract is stable: it receives the state as already-prepared for relative dynamics.

## Replay Path (Command Publish)

After solve:

1. Solver trajectory is stored in `TrajectoryReplayer` with `is_relative_plan` metadata.
2. On replay tick:
   - if `is_relative_plan` is true, planner adds live target position/velocity back to relative command state before publishing command
   - otherwise, command state is used directly

For `stateswitch`, `is_relative_plan` is set from target snapshot validity in registry post-processing.

## Direct Relative Odometry Testing

For test environments without an external relative-state system:

`target_publisher` can publish `/drone/relative_odometry` by:

1. subscribing to drone odometry (`drone_odom_topic` parameter)
2. generating circular target state
3. publishing relative odometry:
   - `p_rel = p_drone - p_target`
   - `v_rel = v_drone - v_target`

This allows running planner with:

- `drone_state_is_relative=true`
- `drone_odom_topic=/drone/relative_odometry`

and testing direct-relative input mode end-to-end.

## Practical Summary

There are two valid pipelines for `stateswitch`:

1. Absolute input pipeline:
   - planner receives absolute drone state
   - registry subtracts target
   - OCP receives relative state

2. Direct-relative input pipeline:
   - planner receives already-relative drone state
   - registry pass-through
   - OCP receives relative state

Both pipelines are intended to feed the same effective initial relative state into `stateswitch`.
