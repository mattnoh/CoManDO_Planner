# CLAUDE.md

This file gives coding-agent guidance for the `comando_planner` package on the
**ROS 1 Noetic branch** (a git worktree of the ROS 2 repo). The README's
`ros2 launch` commands belong to the ROS 2 branch; everything below is what
actually works here.

## Branch facts

- Build system: **catkin** (`catkin build comando_planner`), not colcon.
- **Only `platform:=mavros` is implemented.** `include/platform/crazyflie.hpp`
  is a compile-only stub (State struct so `state_monitor.hpp` compiles);
  `planner_node.cpp` throws for any other platform.
- `mavros_msgs` is a **hard** dependency on this branch (CMakeLists
  `CATKIN_DEPENDS`), not optional.
- Launch files: use the `.launch` XML files. The `.py` files in `launch/`
  are ROS 2 (`launch`/`launch_ros`) and do not work under Noetic. There is
  **no `ocp_launch.launch`** — OCPs are triggered via rosparam (below).

## Build

```bash
source /opt/ros/noetic/setup.zsh
cd /home/lab/mavros_ws
catkin build comando_planner
source devel/setup.zsh
```

ALIPDDP is pulled in via `add_subdirectory(../ALIPDDP-main ...)`; it is never
built by catkin directly (root-level `CATKIN_IGNORE`).

## Common Run Commands

Planner infrastructure (starts paused, unconfigured):

```bash
roslaunch comando_planner planner_launch.launch platform:=mavros drone_name:=px4_drone
```

Trigger an OCP profile (params live under the node's private namespace
`/comando_planner/`; bump `command_seq` to trigger):

```bash
rosrun comando_planner apply_profile.sh                          # config/profile_hover_sitl.yaml
rosrun comando_planner apply_profile.sh /comando_planner \
  $(rospack find comando_planner)/config/profile_stc_landing_sitl.yaml
# or by hand:
rosparam set /comando_planner/ocp_type hover
rosparam set /comando_planner/mode mpc
rosparam set /comando_planner/hover_target_z 1.5
rosparam set /comando_planner/command_seq 1
```

Profile gotchas (see the comments in `config/*.yaml`):

- **Params are sticky** on the rosparam server — always set `land_action`
  explicitly in every profile so a previous landing profile can't leak into
  hover. `land_action: force_disarm` is the verified choice for moving
  platforms; `auto_land` is too slow (target drives away during descent).
- The SITL landing sequence is hover to the staging height first
  (`profile_hover_sitl.yaml`, world 1.8 m), wait for settle, then apply the
  landing profile.

Target publisher (needed by target-relative OCPs):

```bash
roslaunch comando_planner target_launch.launch \
  target_trajectory:=circle \
  drone_odom_topic:=/mavros/local_position/odom \
  drone_pose_topic:=/mavros/local_position/pose
```

Simulated relative sensing (drives `/target/odom` from a body-frame relative
measurement instead of ground truth; ground truth moves to `/target/odom_gt`):

```bash
roslaunch comando_planner target_launch.launch \
  target_source:=relative_estimate sim_relative_estimate:=true pad_z:=0.2
```

The two stages talk only over `/drone/relative_target_estimate`, so a real
estimator swaps in with `sim_relative_estimate:=false`. Contract and math:
`docs/dynamic_target_hardware_integration.md`.

Full PX4 SITL bring-up (PX4 + gz + mavros + planner): **`docs/MAVROS_SITL.md`**.

## Current Architecture

| Area | Files | Notes |
| --- | --- | --- |
| ROS node | `src/planner_node.cpp` | Parameters (10 Hz rosparam polling), timers, state snapshots, logging, command publication, solve-acceptance gates |
| ROS-free core | `include/planner_core/` + `src/quadrotor_mpc.cpp` | OCP descriptors, frame/command adapters, solve/replay orchestration |
| OCPs | `include/ocp/` | Concrete ALIPDDP problem definitions and descriptors |
| Platforms | `include/platform/` | MAVROS (real), Crazyflie (stub), target tracking, state monitor |
| Target publisher | `src/target_publisher.cpp` + `include/target/` | Synthetic circle/figure-8 target on `/target/odom` + `/target/accel`, plus optional relative-odom outputs (`planning_frame` param) and simulated body-frame relative sensing (`target_source`/`sim_relative_estimate`) |
| Replay | `include/trajectory_replayer.hpp` | Thread-safe active/pending plan interpolation |

Solve-acceptance / start gates in `planner_node.cpp` (do not weaken to "make
it fly"):

- `max_constraint_error` (launch arg, default 1.0) rejects unconverged solves.
  Converged solves report ~1e-15; infeasible ones (e.g. a landing triggered
  from below the STC staging height) report 1e2–1e4 and fly wild trajectories
  if accepted. If solves are rejected en masse, fix the initial condition
  (start the landing from the settled staging hover), don't open the gate.
- Target-relative OCPs hold the command (seq not consumed) until
  `/target/odom` is fresh, so launching the planner before `target_launch`
  is safe — it auto-starts once target data arrives.

### MAVROS adapter (`include/platform/mavros.hpp`)

Hard-coded global topics (`/mavros/...`) — launch mavros **un-namespaced**.
A 10 Hz heartbeat streams setpoints continuously (warmup hold → last command),
requests OFFBOARD then ARM with ≤1 Hz retries, and drives its arm state from
`/mavros/state` truth. OFFBOARD/ARM is only auto-requested during initial
engagement — after a later disarm/mode change it logs and stands down.
ENU data in/out; mavros converts ENU→NED internally (do not add conversions).

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
stateswitch            (alias: state_switch)
stc_landing            (alias: rh_stc)
stc_landing_noaug      (alias: rh_stc_noaug)
tracking_circle_target (alias: circle_target)
tracking_bodyrate_tf_noimu / tf_noimu
tracking_bodyrate_tf_imu   / tf_imu
tracking_bodyrate_bf_imu   / bf_imu
tracking_bodyrate_bf_noimu / bf_noimu
```

`stc_landing_noaug` is the no-aug interval CT-cSTC arm ported from
`ALIPDDP-main/problem_examples/STC/quad_single_horizon_noaug_stc.cpp` onto the
same RH machinery: 14-dim solver state (no accumulator), per-stage hard
inequality `integral(H dt) <= SZMUK_CTCS_STEP_EPS` (SZMUK_CTCS_MODE ignored —
the interval path is hard-wired). Defaults are the noaug_basin champion
(EPS=1.6e-3, Y_SCALE=10). Its cold seed aims at (0,0,Z_STAGE) until laterally
captured — a seed that dives below the staging floor stalls the IPM on the
hard interval constraint. Activate it from a settled staging hover with a
general lateral offset (`test_stc_landing_noaug` checks exactly that gate).

`stc_landing` is the CT-cSTC landing OCP ported from
`ALIPDDP-main/problem_examples/STC/quad_landing_rh_clean_stc.cpp`. It carries
a 15-dim solver state (13 physical + cumulative time + constraint
accumulator), rebases the time/accumulator rows inside `create()` on each
warm-started solve, and keeps all of the benchmark's `SZMUK_*` env-var
tunables (export before launching to retune without rebuilding).

## Runtime Contracts

| OCP family | Drone input | Target input | Command output |
| --- | --- | --- | --- |
| `hover`, `landing` | absolute MAVROS odom | none | full state |
| `stateswitch`, `stc_landing`, `stc_landing_noaug` | absolute drone state | `/target/odom`, `/target/accel` | full state reconstructed to world |
| `tracking_circle_target` | absolute drone state | `/target/odom`, `/target/predicted_accel`; open-loop only | full state reconstructed to world |
| `tracking_bodyrate_bf_*` | `/drone/body_relative_odom` | `/target/odom`, `/target/accel` | MAVROS `AttitudeTarget` |
| `tracking_bodyrate_tf_*` | `/drone/target_frame_odom` | `/target/odom`, `/target/accel` | MAVROS `AttitudeTarget` |

Note: all OCPs use Crazyflie-scale constants (mass 0.027 kg). Full-state OCPs
are fine on larger PX4 airframes (PX4 tracks the setpoints); body-rate OCPs
need retuning first (see `docs/MAVROS_SITL.md` §7).

## Validation Commands

After building (standalone, no ROS master needed):

```bash
rosrun comando_planner test_poly_ocp
rosrun comando_planner test_trajectory_replayer
rosrun comando_planner test_target_frame_warm_start
rosrun comando_planner test_planner_core
rosrun comando_planner test_stateswitch_predictor
rosrun comando_planner test_stc_landing
rosrun comando_planner test_stc_landing_noaug
```

Manual MAVROS body-rate path check:

```bash
rosrun comando_planner test_bodyrate_cmd.py \
  _drone_name:=drone _thrust:=9.81 _wz:=0.52 _duration:=3.0
```

## Post-Flight Analysis (`logs/`)

`planner_launch.launch` auto-records `logs/flight_<date>.bag` (disable with
`record_bag:=false`); Ctrl-C the launch to finalize it. The planner's own
CSVs (`all_solves.csv`, `solver_events.csv`) land in a per-run folder under
`logs/`. One command collects and renders everything for the newest run:

```bash
cd $(rospack find comando_planner)/logs
python3 analyze_flight.py     # newest bag → run folder: executed CSVs,
                              # 3D + states figures, planned-vs-executed
                              # overlay, GIF (--bag/--name to override)
```

The plot window is auto-trimmed from landing-OCP activation to touchdown.
Sub-tools (`bag_to_stc_csv.py`, `plot_stc_paper.py`,
`plot_sitl_planned_overlay.py`, `animate_stc_paper.py`) can be run
individually — see `docs/MAVROS_SITL.md` §9 for the tool → output table.

RViz diagnostics: `roslaunch comando_planner rviz.launch` (add `replay:=true`
when viewing a bag — it sets `/use_sim_time`, which the launch always writes
explicitly because a leftover `true` from a previous replay freezes every
node started afterwards).

## Documentation

| File | Purpose |
| --- | --- |
| `README.md` | Quick start (ROS 2-centric; see the banner for this branch) |
| `docs/MAVROS_SITL.md` | PX4 SITL + MAVROS bring-up guide for this branch |
| `docs/COMANDO_PLANNER_INTERFACE.md` | Full ROS topics, parameters, registry, logging |
| `docs/dynamic_target_hardware_integration.md` | External target estimator/predictor contract |
| `docs/CoManDO_planner/` | Chapter-style architecture notes |
