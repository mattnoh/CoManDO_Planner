# Chapter 3: MPC, OCP Registry, and Command Modes

The planner uses ALIPDDP through `QuadrotorMPC` and OCP descriptors. Runtime
code chooses an OCP by string key, then uses descriptor fields and callbacks to
configure state frames, target validation, command dispatch, logging headers,
and solve construction.

## Core State and Control Families

### 13D Full-State OCPs

Used by `hover`, `landing`, `stateswitch`, and `tracking_circle_target`.

```text
x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
u = [fz, mx, my, mz]
```

`fz` is body-z thrust in Newtons for full-state Crazyflie command conversion.
Moments are OCP controls and are logged, but the Crazyflie `FullState` command
uses position, velocity, attitude, angular velocity, and acceleration
feedforward.

### Body-Rate OCPs Without Augmentation

Used by `tracking_bodyrate_bf_noimu` and `tracking_bodyrate_tf_noimu`.

Physical state is 10D plus a cumulative DT slot used by variable-time dynamics:

```text
target-frame: [p_B^N, v_B^N, q_NB, DT]
body-frame:   [p_T^B, v_rel^B, q_NB, DT]
u = [T_ms2, omega_x, omega_y, omega_z, Theta]
```

`T_ms2` is specific thrust. `Theta` is the optimized time segment duration.

### Body-Rate OCPs With Augmentation

Used by `tracking_bodyrate_bf_imu` and `tracking_bodyrate_tf_imu`.

Physical state is 19D plus DT:

```text
base 10D state
+ target angular velocity Omega_N
+ target linear acceleration estimate
+ target angular acceleration beta_N
+ DT
```

The exact acceleration-state interpretation differs by body-frame versus
target-frame OCP headers, but both carry target kinematics through the solve.

## OCP Registry

Source: `include/planner_core/ocp_registry.hpp`

Each OCP exposes a `descriptor()` function. The registry table maps keys to
descriptors:

```cpp
{"hover", HoverOCP::descriptor()}
{"landing", LandingOCP::descriptor()}
{"stateswitch", StateswitchOCP::descriptor()}
{"tracking_circle_target", TrackingCircleTargetOCP::descriptor()}
{"tracking_bodyrate_tf_noimu", TrackingBodyrateTfNoImuOCP::descriptor()}
{"tracking_bodyrate_tf_imu", TrackingBodyrateTfImuOCP::descriptor()}
{"tracking_bodyrate_bf_imu", TrackingBodyrateBfImuOCP::descriptor()}
{"tracking_bodyrate_bf_noimu", TrackingBodyrateBfNoImuOCP::descriptor()}
```

The node and `PlannerCore` use the descriptor rather than switching on OCP names
for most behavior.

## Descriptor Fields That Affect Runtime

| Field | Runtime effect |
| --- | --- |
| `dt` | Replay timer period and nominal OCP step |
| `default_n_replay` | Default replan handoff spacing |
| `default_mass_kg` | Command conversion and hover-control mass |
| `command_mode` | `CmdFullState` or `CmdBodyRate` |
| `drone_odom_mode` | Selects absolute, body-relative, or target-frame state input |
| `variable_dt` | Reads cumulative node time from state index `state_dim` |
| `needs_target_trajectory` | Requires fresh predicted acceleration before open-loop solve |
| `skip_trajectory_validation` | Bypasses generic physical bound checks for OCPs whose raw state is not world absolute |
| `transform_state` | Converts sensor state to OCP state before solving |
| `validate_target` | OCP-specific target freshness gate |
| `post_process_result` | Adds relative-plan metadata and target reconstruction data after solving |
| `prepare_extra` | Builds OCP-specific extra data passed to `create()` |
| `reconstruct_world_state` | Converts replay state to world state for full-state command publication |

## Current OCP Behavior Summary

| OCP | Important descriptor behavior |
| --- | --- |
| `hover` | Absolute 13D, full-state command, fixed `dt=0.05`, default mass `0.0282 kg` |
| `landing` | Absolute 13D, full-state command, fixed `dt=0.05`, default mass `0.027 kg` |
| `stateswitch` | Absolute sensor input, subtracts target position/velocity in `transform_state`, variable DT, full-state command reconstructed to world |
| `tracking_circle_target` | Absolute sensor input, subtracts target odom, needs predicted acceleration, open-loop only in node |
| `tracking_bodyrate_bf_*` | Reads `/drone/body_relative_odom`, variable DT, command mode `CmdBodyRate` |
| `tracking_bodyrate_tf_*` | Reads `/drone/target_frame_odom`, variable DT, command mode `CmdBodyRate` |

## MPC Solve Path

In `mpc` mode:

1. `planner_node` gathers current drone state and `TargetSnapshot`.
2. `PlannerCore::trySolve()` validates the frame contract.
3. Descriptor `transform_state` prepares the OCP state.
4. Warm-start data from the previous accepted solve is selected.
5. Descriptor `prepare_extra` builds OCP-specific data.
6. `QuadrotorMPC::solve()` creates the concrete OCP and calls ALIPDDP.
7. Descriptor `post_process_result` adds target-relative metadata.
8. Generic acceptance checks validate success, constraint error, and physical
   bounds unless skipped by configuration or descriptor.
9. Accepted plans are handed to `TrajectoryReplayer` and logged.

## Warm Starting and Handoff

`PlannerCore` maintains the last accepted plan and uses the replayer to avoid
command gaps. New solves do not instantly replace the active plan. They are
stored as pending plans and swapped only after:

```text
now >= pending_earliest_activation_time
and active plan has replayed at least n_replay steps
```

For variable-DT OCPs, node time comes from the cumulative DT state slot instead
of `k * dt`.

## Command Dispatch

`planner_core/adapters.hpp` converts replay samples to `PlannerCommand`.

| Descriptor command mode | Platform | ROS command |
| --- | --- | --- |
| `CmdFullState` | Crazyflie | `FullState` on `/{drone}/cmd_full_state` |
| `CmdFullState` | MAVROS | `PositionTarget` on `/mavros/setpoint_raw/local` |
| `CmdBodyRate` | Crazyflie | Converted hover command on `/{drone}/cmd_hover` |
| `CmdBodyRate` | MAVROS | `AttitudeTarget` on `/mavros/setpoint_raw/attitude` |

This is why body-rate OCPs can share one OCP output while using different
platform command APIs.

[Next Chapter: Logging, RViz, Bags, and Debugging](04_debugging_log.md)
