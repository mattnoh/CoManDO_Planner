# CoManDO Planner (ROS 1 Noetic, MAVROS)

`comando_planner` is an MPC planner for MAVROS/PX4 quadrotors. It wraps
ALIPDDP optimal-control problems behind a registry, keeps drone-state and
target tracking in ROS adapters, and streams either full-state or body-rate
setpoints depending on the active OCP. On this branch only
`platform:=mavros` is implemented.

The workflow is two-step: start the planner infrastructure paused, then push
an OCP profile onto the rosparam server and bump `command_seq` to trigger it.

## Package Layout

| Path | Purpose |
| --- | --- |
| `src/planner_node.cpp` | ROS 1 node: parameter polling, timers, logging, platform dispatch |
| `include/planner_core/` | ROS-free planner contracts, OCP registry, command adapters, replay orchestration |
| `include/ocp/` | Registered ALIPDDP OCP formulations |
| `include/platform/` | MAVROS adapter, state monitor, target tracker |
| `src/target_publisher.cpp` | Synthetic target publisher plus optional relative odometry |
| `launch/*.launch` | ROS 1 launch files (the `.py` files are ROS 2 leftovers — do not use) |
| `config/profile_*.yaml` | OCP profiles for `apply_profile.sh` |
| `logs/*.py` | Post-flight analysis and figure scripts; run folders and bags also land here |
| `rviz/comando_sitl.rviz` | RViz layout for SITL (drone, target, planned trajectory) |
| `docs/` | Interface reference, architecture notes, SITL bring-up guide |

## Requirements and Build

| Component | Notes |
| --- | --- |
| Ubuntu 20.04, ROS 1 Noetic | Base environment |
| ALIPDDP | Sibling directory `../ALIPDDP-main`, pulled in via `add_subdirectory` |
| `mavros`, `mavros_msgs` | Hard dependency on this branch |
| Eigen3 | Required by planner, OCPs, and target helpers |

```bash
source /opt/ros/noetic/setup.zsh
cd /home/lab/mavros_ws
catkin build comando_planner
source devel/setup.zsh
```

## Quick Start (PX4 SITL)

Full bring-up guide with troubleshooting: **`docs/MAVROS_SITL.md`**.

```bash
# Terminal 1 — PX4 SITL (drop HEADLESS=1 for the Gazebo GUI)
cd /home/lab/projects/px4/PX4-Autopilot
HEADLESS=1 make px4_sitl gz_x500
# first run after a PX4 clean: rosrun mavros mavparam set NAV_DLL_ACT 0

# Terminal 2 — MAVROS (must be un-namespaced; adapter hard-codes /mavros/...)
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"

# Terminal 3 — target publisher (needed for target-relative OCPs)
roslaunch comando_planner target_launch.launch target_trajectory:=circle \
  drone_odom_topic:=/mavros/local_position/odom \
  drone_pose_topic:=/mavros/local_position/pose

# Terminal 4 — planner: engages OFFBOARD+ARM, then waits paused.
# Also auto-records a flight bag to logs/flight_<date>.bag (record_bag:=false to disable).
roslaunch comando_planner planner_launch.launch platform:=mavros \
  drone_name:=px4_drone terminal_freeze_enter_vel:=0.30

# Trigger OCPs, one at a time:
rosrun comando_planner apply_profile.sh   # hover (open_loop) to the staging point
rosrun comando_planner apply_profile.sh /comando_planner \
  $(rospack find comando_planner)/config/profile_stc_landing_sitl.yaml   # land on the target
```

There is **no `ocp_launch.launch`** — profiles are plain rosparams on
`/comando_planner/`, and `apply_profile.sh` loads a yaml and bumps
`command_seq`. Setting params by hand works too:

```bash
rosparam set /comando_planner/ocp_type hover
rosparam set /comando_planner/mode open_loop
rosparam set /comando_planner/hover_target_z 1.8
rosparam set /comando_planner/command_seq 1     # bump by +1 to trigger
```

The default hover profile runs **open_loop** (one solve, replay, hold at the
plan's end) — the transit to the staging point needs no replanning. Landing
touchdown handling is set per profile via `land_action`
(`force_disarm` / `auto_land` / `disarm` / `none`); see `docs/MAVROS_SITL.md`.

## Registered OCPs

`include/planner_core/ocp_registry.hpp`:

```text
hover
landing
stateswitch            (alias: state_switch)
stc_landing            (alias: rh_stc — CT-cSTC landing, SZMUK_* env tunables)
tracking_circle_target (alias: circle_target; open_loop only)
tracking_bodyrate_tf_noimu / tf_noimu
tracking_bodyrate_tf_imu   / tf_imu
tracking_bodyrate_bf_imu   / bf_imu
tracking_bodyrate_bf_noimu / bf_noimu
```

Full-state OCPs (`hover`, `landing`, `stateswitch`, `stc_landing`) publish to
`/mavros/setpoint_raw/local` and PX4's own position controller tracks them.
Body-rate OCPs publish to `/mavros/setpoint_raw/attitude` and use cf-scale
constants — retune before using them on a larger airframe
(`docs/MAVROS_SITL.md` §7).

## Target Publisher

`target_launch.launch` publishes `/target/odom` (`nav_msgs/Odometry`) and
`/target/accel` (`geometry_msgs/AccelStamped`). Key args:

| Arg | Default | Notes |
| --- | --- | --- |
| `target_trajectory` | `circle` | Synthetic trajectory shape |
| `planning_frame` | `world` | `world` for `stateswitch`/`stc_landing`; `body_frame`/`target_frame` add relative drone odometry for the body-rate OCPs |
| `drone_odom_topic` / `drone_pose_topic` | `/mavros/local_position/...` | Drone state inputs |
| `publish_hz` | `100.0` | Publish rate |

## Planner Launch Arguments

`planner_launch.launch` highlights (see the file for the full list):

| Arg | Default | Notes |
| --- | --- | --- |
| `platform` | `mavros` | Only implemented platform on this branch |
| `drone_name` | `px4_drone` | Names `/{drone}/planned_trajectory`; logging label |
| `log_dir` | `$(find comando_planner)/logs` | Where planner CSV run folders are created |
| `record_bag` | `true` | Auto-record `logs/flight_<date>.bag` alongside the planner |
| `terminal_freeze_enter_pos/vel` | `0.20` / `0.10` | Touchdown latch gates; use `vel:=0.30` against moving targets |
| `hover_thrust` | `0.3` | Body-rate OCPs only |
| `skip_trajectory_validation` | `false` | Debug: accept plans that fail the physical-bounds check |
| `stc_z_stage`, `stc_los_alt_trig` | `1.3` | stc_landing staging/trigger altitudes (target-relative metres) |

## Logging, Analysis, RViz

Each planner process writes one run folder
`logs/{drone}_{ocp}_{mode}_{solver}_{timestamp}/` with `all_solves.csv` and
`solver_events.csv` (a hover-then-land session logs both phases into the one
folder, named after the first profile). The auto-recorded bag lands next to
it as `logs/flight_<date>.bag` and finalizes when the planner launch is
Ctrl-C'd.

Post-flight, one command builds the standard run folder (bag + CSVs +
executed trajectory + constraint bounds + figures), trimmed automatically
from the **landing** OCP activation to touchdown so the hover transit stays
out of the plots:

```bash
cd $(rospack find comando_planner)/logs
python3 analyze_flight.py   # newest bag → the run's own folder, all figures
# → 3D + states, planned-vs-executed overlay, GIF, executed CSVs
# (--bag <bag> / --name <folder> to override)
```

See `docs/MAVROS_SITL.md` §9 for the tool → output-file table.

Live RViz (shows the virtual target, which the Gazebo GUI cannot):

```bash
roslaunch comando_planner rviz.launch                 # live view
roslaunch comando_planner rviz.launch replay:=true    # + rosbag play --clock <bag>
```

## Validation

Standalone test executables (no ROS master needed):

```bash
rosrun comando_planner test_poly_ocp
rosrun comando_planner test_trajectory_replayer
rosrun comando_planner test_target_frame_warm_start
rosrun comando_planner test_planner_core
rosrun comando_planner test_stateswitch_predictor
rosrun comando_planner test_stc_landing
```

`scripts/test_bodyrate_cmd.py` manually exercises the MAVROS body-rate path.

## More Documentation

| File | Purpose |
| --- | --- |
| `docs/MAVROS_SITL.md` | Full PX4 SITL bring-up, landing recipes, post-flight analysis |
| `docs/COMANDO_PLANNER_INTERFACE.md` | Full ROS topics, parameters, registry, logging |
| `docs/CoManDO_planner/` | Chapter-style architecture notes |
