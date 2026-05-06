# Chapter 4: Logging, RViz, Bags, and Debugging

The current planner has three main debugging surfaces: CSV logs, RViz
visualization topics, and optional rosbag recording from `planner_launch.py`.

## CSV Logs

Enable logging with:

```bash
ros2 launch comando_planner planner_launch.py enable_logging:=true
```

The logger writes under the current working directory of the node:

```text
./logs/{drone}_{ocp}_{mode}_{solver}_{YYYYMMDD_HHMMSS}/
```

| File | Purpose |
| --- | --- |
| `all_solves.csv` | One row per solve node; includes solve metadata, OCP state/control, node time, `theta`, and target snapshot/reconstruction columns |
| `commanded_state.csv` | Full-state command rows; includes command state, `[fz,mx,my,mz]`, and computed acceleration feedforward |
| `commanded_hover_state.csv` | Crazyflie body-rate OCP dispatch after conversion to `cmd_hover` |
| `commanded_bodyrate_state.csv` | MAVROS body-rate OCP dispatch as specific thrust and angular rates |
| `actual_state.csv` | Raw measured state rows with a coordinate-mode label |

The coordinate-mode column indicates how to interpret actual-state rows:

| Label | Meaning |
| --- | --- |
| `absolute` | World-frame drone state |
| `absolute_shifted` | Absolute drone input for an OCP that solves target-relative internally |
| `body_relative` | Body-frame relative odometry input |
| `target_frame_relative` | Target-frame relative odometry input |

`all_solves.csv` includes target columns even for absolute OCPs. For relative
plans, target node positions and velocities are either reconstructed by the OCP
post-processing path or integrated from the target snapshot.

## RViz Topics

The planner publishes:

| Topic | Type | Notes |
| --- | --- | --- |
| `/{drone_name}/planned_trajectory` | `nav_msgs/msg/Path` | Last accepted horizon, reconstructed in world frame when possible |
| `/{drone_name}/planner_debug_markers` | `visualization_msgs/msg/MarkerArray` | Target sphere/trail, active command marker, handoff jump marker/text |

Open the bundled RViz layout from the workspace root after build:

```bash
rviz2 -d install/comando_planner/share/comando_planner/rviz/comando_debug.rviz
```

## Bag Recording

`planner_launch.py` can start `ros2 bag record` automatically:

```bash
ros2 launch comando_planner planner_launch.py \
  drone_name:=cf_1 platform:=crazyflie \
  record_bag:=true
```

It records TF, planner visualization, target streams, Crazyflie command/state
topics, and MAVROS local-position/setpoint topics. Set `record_bag:=false` for
normal development if you do not want bag files created every launch. By
default, the bag is saved inside the matching run log folder as
`bags/comando_debug`.

Manual replay:

```bash
ros2 bag play logs/RUN_FOLDER/bags/comando_debug --clock
rviz2 -d install/comando_planner/share/comando_planner/rviz/comando_debug.rviz
```

## Common Runtime Warnings

| Message fragment | Meaning | Usual fix |
| --- | --- | --- |
| `Planner started UNCONFIGURED` | Node has no `ocp_type` and `mode` yet | Run `ocp_launch.py` with a profile |
| `Waiting for state` | Required drone state has not arrived | Check platform, drone namespace, relative odom mode, and topic names |
| `Waiting for fresh target state` | OCP target gate failed | Check `/target/odom` and `/target/accel` timestamps/rates |
| `waiting for /target/predicted_accel` | `tracking_circle_target` has no fresh future acceleration buffer | Publish `MultiDOFJointTrajectory` predicted acceleration |
| `frame_contract` | Descriptor expected a different active odom mode/topic | Check active `ocp_type` and target_publisher `drone_odom_mode` |
| `Trajectory stale` | Replay sampled beyond accepted horizon | Check solver failures, target freshness, and constraint/validation rejects |
| `tracking_circle_target is open_loop-only` | Runtime requested `mpc` for that OCP | Use `mode:=open_loop` |

## Body-Relative Debugging

Enable target-publisher conversion logs:

```bash
ros2 run comando_planner target_publisher --ros-args \
  -p drone_odom_mode:=body_frame \
  -p debug_body_relative_trace:=true
```

If upstream drone odometry angular velocity is already in rad/s, pass:

```bash
-p body_relative_input_angular_unit:=rad_s
```

The default is `deg_s`, matching the Crazyflie/Crazyswarm2 convention used by
the helper publisher. Planner-side `debug_body_relative_trace` exists in runtime
config, but the current `planner_launch.py` does not expose it as a launch
argument.

## Validation Targets

After building, run installed validation executables with:

```bash
ros2 run comando_planner test_poly_ocp
ros2 run comando_planner test_trajectory_replayer
ros2 run comando_planner test_target_frame_warm_start
ros2 run comando_planner test_planner_core
ros2 run comando_planner test_stateswitch_predictor
```

For a direct Crazyflie command-path check:

```bash
ros2 run comando_planner test_bodyrate_cmd.py --ros-args \
  -p drone_name:=cf_1 -p thrust:=9.81 -p wz:=0.52 -p duration:=3.0
```

[Next Chapter: Frames, Relative Dynamics, and Tracking Math](05_mathematical_tutorial.md)
