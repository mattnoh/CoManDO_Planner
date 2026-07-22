# CLAUDE.md

This file gives coding-agent guidance for the current `comando_planner`
package.

## Build

Build from the ROS workspace root:

```bash
source /opt/ros/humble/setup.zsh
cd /home/lab/projects/cf/online/ros_ws
colcon build --symlink-install --packages-select comando_planner
source install/setup.zsh
```

`CMakeLists.txt` adds ALIPDDP from the sibling package path
`../ALIPDDP-main`. Optional dependencies:

| Dependency | Build behavior |
| --- | --- |
| `mavros_msgs` | Enables MAVROS platform support and defines `HAS_MAVROS_MSGS` |
| `mocap4r2_msgs` | Enables mocap target mode and defines `HAS_MOCAP4R2_MSGS` |

The package still builds without those optional message packages, but related
runtime paths are unavailable or fall back.

## Common Run Commands

Two-step launch:

```bash
ros2 launch comando_planner planner_launch.py \
  drone_name:=cf_1 platform:=crazyflie record_bag:=false

ros2 launch comando_planner ocp_launch.py \
  ocp_type:=hover mode:=mpc n_replay:=4 \
  hover_target_x:=0.0 hover_target_y:=0.0 hover_target_z:=1.0 \
  command_seq:=1
```

Target publisher for body-frame OCPs:

```bash
ros2 run comando_planner target_publisher --ros-args \
  -p target_mode:=gazebo_circle \
  -p drone_odom_mode:=body_frame \
  -p drone_odom_topic:=/cf_1/odom \
  -p drone_pose_topic:=/cf_1/pose
```

Target publisher for target-frame OCPs:

```bash
ros2 run comando_planner target_publisher --ros-args \
  -p target_mode:=gazebo_circle \
  -p drone_odom_mode:=target_frame \
  -p drone_odom_topic:=/cf_1/odom \
  -p drone_pose_topic:=/cf_1/pose
```

## CrazySim SITL (validated stc_landing_noaug landing, 2026-07-22)

CrazySim lives at `../../../CrazySim` (firmware SITL prebuilt in
`crazyflie-firmware/sitl_make/build/cf2`). Full sequence:

```bash
# 1. firmware + gazebo (UDP 19950 firmware / 19850 cflib — matches crazyflies.yaml cf_1)
cd <CrazySim>/crazyflie-firmware
bash tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_singleagent.sh -m crazyflie -x 0 -y 0

# 2. crazyswarm2 — cflib backend, NOT backend:=sim (that is the python
#    integrator: no firmware, no /cf_1/pose)
ros2 launch crazyflie launch.py backend:=cflib gui:=False mocap:=False

# 3. planner — record_bag:=false is MANDATORY (bag recording OOM-kills the node)
ros2 launch comando_planner planner_launch.py drone_name:=cf_1 platform:=crazyflie record_bag:=false

# 4. target
ros2 launch comando_planner target_launch.py target_mode:=gazebo_circle \
  planning_frame:=world drone_odom_topic:=/cf_1/odom drone_pose_topic:=/cf_1/pose

# 5. position with the crazyswarm2 high-level commander, NOT the hover OCP
#    (planner_node relies on goto holding the drone until the first solve):
ros2 service call /cf_1/arm crazyflie_interfaces/srv/Arm "{arm: true}"
ros2 service call /cf_1/takeoff crazyflie_interfaces/srv/Takeoff "{height: 1.0, duration: {sec: 3}}"
ros2 service call /cf_1/go_to crazyflie_interfaces/srv/GoTo \
  "{relative: false, goal: {x: 2.0, y: 2.0, z: 1.8}, duration: {sec: 5}}"

# 6. trigger the landing OCP (bump command_seq by +1 each trigger)
ros2 launch comando_planner ocp_launch.py ocp_type:=stc_landing_noaug \
  mode:=mpc n_replay:=7 command_seq:=1
```

Hard-won gotchas:

- **`SZMUK_Z_STAGE=1.3` / `SZMUK_LOS_ALT_TRIG=1.3`** are required for the stc
  landing OCPs to converge from the documented staging geometry; the compiled
  defaults (1.8/1.8) stall the IPM (constraint ~30-60, "Outer Max/Min").
  `planner_launch.py` injects them via `additional_env` (args `stc_z_stage`,
  `stc_los_alt_trig`), mirroring the ROS1 `planner_launch.launch`. Any
  standalone solve harness must export them too.
- ROS1-style flat profile YAMLs do **not** load with `ros2 param load`
  (needs `/comando_planner:\n  ros__parameters:`) and fail silently —
  `command_seq` stays 0 and the OCP never starts. Use `ocp_launch.py`.
- `test_stc_landing_noaug` validates against an **R=2.0 / v=0.8** target; the
  SITL circle is R=1.0 / v=0.4. The unit test passing does not prove SITL
  convergence — check offline with the real target values + launch env.
- Known gaps vs the ROS1 arm: `land_action` is not a declared parameter (no
  touchdown disarm — the drone holds the terminal setpoint on the ground),
  and in the validated landing all post-activation replans were rejected
  (the flight rode the activation plan; MAVROS re-accepts ~13 solves).

## Current Architecture

The planner is split into:

| Area | Files | Notes |
| --- | --- | --- |
| ROS node | `src/planner_node.cpp` | Parameters, timers, state snapshots, logging, command publication |
| ROS-free core | `include/planner_core/` | OCP descriptors, frame/command adapters, solve/replay orchestration |
| OCPs | `include/ocp/` | Concrete ALIPDDP problem definitions and descriptors |
| Platforms | `include/platform/` | Crazyflie, MAVROS, target tracking, state monitor |
| Replay | `include/trajectory_replayer.hpp` | Thread-safe active/pending plan interpolation |

## OCP Registry

Registry source: `include/planner_core/ocp_registry.hpp`.

To add an OCP:

1. Add `include/ocp/ocp_mynew.hpp`.
2. Implement `MyNewOCP::descriptor()` returning `OCPDescriptor`.
3. Include the header in `ocp_registry.hpp`.
4. Add one table entry: `{"mynew", MyNewOCP::descriptor()}`.

Current keys:

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

There is no current `tracking_circle` registry key.

## Runtime Contracts

| OCP family | Drone input | Target input | Command output |
| --- | --- | --- | --- |
| `hover`, `landing` | absolute Crazyflie or MAVROS odom | none | full state |
| `stateswitch` | absolute drone state | `/target/odom`, `/target/accel` | full state reconstructed to world |
| `tracking_circle_target` | absolute drone state | `/target/odom`, `/target/predicted_accel`; open-loop only | full state reconstructed to world |
| `tracking_bodyrate_bf_*` | `/drone/body_relative_odom` | `/target/odom`, `/target/accel` | Crazyflie `cmd_hover` or MAVROS `AttitudeTarget` |
| `tracking_bodyrate_tf_*` | `/drone/target_frame_odom` | `/target/odom`, `/target/accel` | Crazyflie `cmd_hover` or MAVROS `AttitudeTarget` |

## Validation Commands

After building:

```bash
ros2 run comando_planner test_poly_ocp
ros2 run comando_planner test_trajectory_replayer
ros2 run comando_planner test_target_frame_warm_start
ros2 run comando_planner test_planner_core
ros2 run comando_planner test_stateswitch_predictor
```

Manual Crazyflie body-rate path check:

```bash
ros2 run comando_planner test_bodyrate_cmd.py --ros-args \
  -p drone_name:=cf_1 -p thrust:=9.81 -p wz:=0.52 -p duration:=3.0
```

## Documentation

Primary docs:

| File | Purpose |
| --- | --- |
| `README.md` | Quick start, OCP catalog, workflows |
| `docs/COMANDO_PLANNER_INTERFACE.md` | Full ROS topics, parameters, registry, logging |
| `docs/dynamic_target_hardware_integration.md` | External target estimator/predictor contract |
| `docs/CoManDO_planner/` | Chapter-style architecture notes |
