# CoManDO Planner ROS Interface

This document describes the current public ROS interface of the
`comando_planner` package. The executable node is `comando_planner`; the helper
target node is `target_publisher`.

## Executables

| Executable | Source | Purpose |
| --- | --- | --- |
| `comando_planner` | `src/planner_node.cpp` | Runtime planner node, OCP switching, ALIPDDP solve/replay, command publishing |
| `target_publisher` | `src/target_publisher.cpp` | Gazebo-style synthetic pose or mocap target estimator and relative-odometry helper |
| `test_bodyrate_cmd.py` | `scripts/test_bodyrate_cmd.py` | Manual body-rate command publisher for Crazyflie command-path checks |

## Launch Model

`planner_launch.py` starts the planner infrastructure only. It sets static
platform and topic parameters, starts the node paused, and initializes
`command_seq` to zero.

`ocp_launch.py` does not start a node. It sets runtime OCP parameters on an
already-running `/comando_planner` node and sets `command_seq` last. The planner
starts or restarts a command only when `command_seq` increases.

```bash
ros2 launch comando_planner planner_launch.py drone_name:=cf_1 platform:=crazyflie

ros2 launch comando_planner ocp_launch.py \
  ocp_type:=hover mode:=mpc hover_target_z:=1.0 command_seq:=1
```

## Planner Parameters

### Infrastructure Parameters

| Parameter | Default | Used by | Description |
| --- | --- | --- | --- |
| `drone_name` | `cf_1` | planner launch | Namespace used for Crazyflie state/command topics and planner visualization topics |
| `platform` | `crazyflie` | planner launch | `crazyflie` or `mavros` |
| `solver` | `alipddp` | planner launch | Solver backend; only `alipddp` is implemented |
| `enable_logging` | `true` | planner launch | Enables CSV logs under `./logs` |
| `hover_thrust` | `0.3` | MAVROS node param | Normalized MAVROS hover thrust used to scale specific thrust commands; declared by the node constructor but not exposed by the current `planner_launch.py` |
| `skip_trajectory_validation` | `false` | node param | Disables generic solve trajectory validation when true |
| `body_relative_odom_topic` | `/drone/body_relative_odom` | planner launch | State input for `tracking_bodyrate_bf_*` OCPs |
| `target_odom_topic` | `/target/odom` | planner launch | Target odometry input |
| `target_accel_topic` | `/target/accel` | planner launch | Target acceleration input |
| `target_predicted_accel_topic` | `/target/predicted_accel` | planner launch | Future target acceleration buffer input |
| `debug_body_relative_trace` | `false` | node param | Extra throttled logging for body-relative state plumbing; declared by runtime config but not exposed by the current `planner_launch.py` |
| `record_bag` | `true` | planner launch | Starts `ros2 bag record` with planner/target/platform topics |
| `bag_output` | empty | planner launch | Optional override for `ros2 bag record -o`; default is the active run log folder's `bags/comando_debug` |

### Runtime Profile Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `ocp_type` | empty in node, `landing` in trigger launch | Registered OCP key |
| `mode` | empty in node, `mpc` in trigger launch | `mpc` or `open_loop` |
| `n_replay` | OCP default when configured, `7` in trigger launch | Minimum active-plan replay depth before an MPC pending plan may replace it |
| `hover_target_x` | `0.0` | Terminal x for full-state absolute OCPs |
| `hover_target_y` | `0.0` | Terminal y for full-state absolute OCPs |
| `hover_target_z` | `0.0` | Terminal z for full-state absolute OCPs |
| `command_seq` | `0` | Increment to start the currently configured profile |
| `open_loop_abort_on_divergence` | `false` | In open loop, abort if measured world z or vz diverges from command beyond thresholds |
| `open_loop_abort_max_z_error_m` | `0.50` | Open-loop z divergence threshold |
| `open_loop_abort_max_vz_error_mps` | `1.00` | Open-loop vertical-velocity divergence threshold |

`tracking_circle_target` is intentionally open-loop only in the current node. A
request to run it in `mpc` mode is rejected.

### Terminal Freeze Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `enable_terminal_freeze` | `true` | Enables solve suppression near the terminal target |
| `terminal_freeze_enter_pos` | `0.20` | Position-error threshold to engage freeze |
| `terminal_freeze_enter_vel` | `0.10` | Velocity-error threshold when velocity gating is enabled |
| `terminal_freeze_require_vel` | `false` | Requires velocity threshold before engaging freeze |
| `terminal_freeze_exit_pos` | `0.20` | Position-error threshold to release freeze |

## Planner Subscriptions

### Crazyflie Platform

| Topic | Type | Notes |
| --- | --- | --- |
| `/{drone_name}/pose` | `geometry_msgs/msg/PoseStamped` | Position and quaternion for absolute OCPs |
| `/{drone_name}/odom` | `nav_msgs/msg/Odometry` | Linear velocity and angular velocity for absolute OCPs |
| `body_relative_odom_topic` | `nav_msgs/msg/Odometry` | Replaces pose/odom subscriptions for body-frame OCPs |
| `/drone/target_frame_odom` | `nav_msgs/msg/Odometry` | Replaces pose/odom subscriptions for target-frame OCPs |

For absolute Crazyflie input, angular rates from Crazyswarm2 are converted from
degrees per second to radians per second. Relative odometry override topics are
expected to already be SI/rad units.

### MAVROS Platform

| Topic | Type | Notes |
| --- | --- | --- |
| `/mavros/local_position/odom` | `nav_msgs/msg/Odometry` | ENU 13D state input |

The MAVROS adapter does not use the Crazyflie relative-odometry resubscribe path.

### Target Inputs

These subscriptions are always created, regardless of active OCP, so the planner
can switch OCPs at runtime without restarting:

| Topic parameter | Default topic | Type | Notes |
| --- | --- | --- | --- |
| `target_odom_topic` | `/target/odom` | `nav_msgs/msg/Odometry` | Target pose, velocity, orientation, angular velocity |
| `target_accel_topic` | `/target/accel` | `geometry_msgs/msg/AccelStamped` | Target linear acceleration and optional angular acceleration |
| `target_predicted_accel_topic` | `/target/predicted_accel` | `trajectory_msgs/msg/MultiDOFJointTrajectory` | Future acceleration samples for `tracking_circle_target` |

Freshness rules are OCP-dependent. Body-rate OCPs use the `TargetSnapshot`
valid flag, which requires fresh target odom and accel. `tracking_circle_target`
requires fresh target odom and separately requires a predicted-acceleration
buffer not older than two seconds.

## Planner Publications

### Crazyflie Commands

| OCP command mode | Topic | Type | Notes |
| --- | --- | --- | --- |
| `CmdFullState` | `/{drone_name}/cmd_full_state` | `crazyflie_interfaces/msg/FullState` | Position, velocity, quaternion, angular rate, acceleration feedforward |
| `CmdBodyRate` | `/{drone_name}/cmd_hover` | `crazyflie_interfaces/msg/Hover` | Planner converts OCP body-rate output to `[vx_body, vy_body, z_distance, yaw_rate]` |

For full-state commands, acceleration feedforward is computed from thrust and
attitude:

```text
a_world = R(q) * [0, 0, fz / mass] + [0, 0, -9.81]
```

### MAVROS Commands

| OCP command mode | Topic | Type | Notes |
| --- | --- | --- | --- |
| `CmdFullState` | `/mavros/setpoint_raw/local` | `mavros_msgs/msg/PositionTarget` | Position, velocity, and yaw |
| `CmdBodyRate` | `/mavros/setpoint_raw/attitude` | `mavros_msgs/msg/AttitudeTarget` | Body rates plus normalized thrust |

For body-rate output, the OCP control `T_ms2` is scaled as:

```text
thrust_norm = clamp(T_ms2 * hover_thrust / 9.81, 0, 1)
```

### Visualization

| Topic | Type | Contents |
| --- | --- | --- |
| `/{drone_name}/planned_trajectory` | `nav_msgs/msg/Path` | Last accepted planned path in world coordinates when reconstructable |
| `/{drone_name}/planner_debug_markers` | `visualization_msgs/msg/MarkerArray` | Target marker/trail, active command point, handoff jump diagnostics |

## OCP Registry Contract

Each OCP is registered in `include/planner_core/ocp_registry.hpp` as an
`OCPDescriptor`.

| Descriptor field | Effect |
| --- | --- |
| `dt` | Nominal replay/sample period |
| `default_n_replay` | Default runtime `n_replay` when none is provided |
| `default_mass_kg` | Mass used by command conversion and hover controls |
| `command_mode` | Selects full-state versus body-rate command dispatch |
| `drone_odom_mode` | Selects absolute, body-relative, or target-frame drone input |
| `variable_dt` | Interprets state index `state_dim` as cumulative time |
| `needs_target_trajectory` | Gates open-loop startup on fresh `/target/predicted_accel` |
| `validate_target` | OCP-specific target freshness/validity gate |
| `transform_state` | Converts sensor state to OCP state before solving |
| `post_process_result` | Marks/reconstructs relative target metadata after solving |
| `prepare_extra` | Builds OCP-specific data passed to the OCP factory |
| `reconstruct_world_state` | Converts replay state to world state for full-state command publication |

Current registry keys are:

```text
hover
landing
stateswitch
tracking_circle_target
tracking_bodyrate_tf_noimu
tracking_bodyrate_tf_imu
tracking_bodyrate_bf_imu
tracking_bodyrate_bf_noimu
```

## Logging Interface

When enabled, logs are created in:

```text
./logs/{drone}_{ocp}_{mode}_{solver}_{YYYYMMDD_HHMMSS}/
```

| File | Written when | Contents |
| --- | --- | --- |
| `all_solves.csv` | Solver success | Full OCP state/control trajectory, timing, target snapshot, target node reconstruction when available |
| `commanded_state.csv` | `CmdFullState` commands | Commanded state, wrench control, acceleration feedforward |
| `commanded_hover_state.csv` | Crazyflie `CmdBodyRate` dispatch | `[vx, vy, z_distance, yaw_rate]` |
| `commanded_bodyrate_state.csv` | MAVROS `CmdBodyRate` dispatch | `[T_ms2, omega_x, omega_y, omega_z]` |
| `actual_state.csv` | Replay/open-loop ticks | Raw measured state labelled by coordinate mode |

Coordinate labels are `absolute`, `absolute_shifted`, `body_relative`, and
`target_frame_relative`.
