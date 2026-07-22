# Running CoManDO on PX4 SITL (MAVROS, ROS 1 Noetic)

This is the bring-up guide for the **MAVROS platform** of this (ROS 1 Noetic)
branch against a PX4 software-in-the-loop simulation. Unlike the Crazyflie
path (CrazySim, ROS 2 branch), there is no packaged simulator launch — the
pipeline is assembled from three pieces: PX4 SITL + Gazebo, MAVROS, and the
planner.

Machine-specific paths in this guide:

| Piece | Location |
| --- | --- |
| PX4-Autopilot (already built) | `/home/lab/projects/px4/PX4-Autopilot` |
| Gazebo Sim 8 (Harmonic, `gz`) | system install (`gz sim`) |
| This workspace | `/home/lab/mavros_ws` |

## TL;DR (three terminals + trigger)

```bash
# Terminal 1 — PX4 SITL with a gz x500 quad
cd /home/lab/projects/px4/PX4-Autopilot
make px4_sitl gz_x500                    # HEADLESS=1 make px4_sitl gz_x500 for no GUI

# Terminal 2 — MAVROS (MUST be un-namespaced: the adapter hard-codes /mavros/...)
source /opt/ros/noetic/setup.zsh && source /home/lab/mavros_ws/devel/setup.zsh
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"

# Terminal 3 — planner (infrastructure only, starts paused).
# Also auto-starts a rosbag recorder: the bag lands in
# CoManDO_planner/logs/flight_<date>.bag (disable with record_bag:=false).
source /opt/ros/noetic/setup.zsh && source /home/lab/mavros_ws/devel/setup.zsh
roslaunch comando_planner planner_launch.launch platform:=mavros drone_name:=px4_drone

# Trigger an OCP (there is NO ocp_launch.launch on this branch — see below)
rosrun comando_planner apply_profile.sh   # loads config/profile_hover_sitl.yaml + bumps command_seq
```

After the planner starts you should see the MAVROS warmup (2 s of hold
setpoints), then `OFFBOARD + armed confirmed by /mavros/state`. Once the hover
profile is applied, the drone flies to the hover target. The profile runs the
hover OCP in **open_loop** mode (one solve from the current state, replay,
hold at the plan's end) — going to the staging point needs no replanning, and
open loop avoids MPC handoff churn entirely. Set `mode: mpc` in the profile
if the transit must react to disturbances.

## 1. PX4 SITL

PX4 is already cloned and built (`build/px4_sitl_default` exists). Useful
variants:

```bash
make px4_sitl gz_x500                # x500 quad, default world, gz GUI
HEADLESS=1 make px4_sitl gz_x500     # no gz GUI (fine for smoke tests)
```

PX4 SITL exposes MAVLink for offboard APIs on **UDP 14540**. In the `pxh>`
shell you can inspect state (`commander status`), see rejection reasons for
OFFBOARD/arming, and read parameters (`param show MPC_THR_HOVER`).

**Required for arming without QGroundControl** (verified on this machine's
PX4 build): the gz_x500 airframe sets `NAV_DLL_ACT=2` (GCS-loss failsafe),
which makes the "No connection to the GCS" preflight check block arming —
mavros heartbeats as an onboard controller, not a GCS. Disable it once:

```bash
rosrun mavros mavparam set NAV_DLL_ACT 0     # or: pxh> param set NAV_DLL_ACT 0
```

Other no-RC niceties (usually already defaulted in recent PX4):

```
pxh> param set COM_RCL_EXCEPT 4     # ignore RC-loss failsafe in offboard
```

## 2. MAVROS

```bash
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```

Two things matter:

- **No namespace.** The planner's MAVROS adapter
  (`include/platform/mavros.hpp`) subscribes/publishes hard-coded global
  names (`/mavros/local_position/odom`, `/mavros/setpoint_raw/*`,
  `/mavros/state`, `/mavros/set_mode`, `/mavros/cmd/arming`). Launching
  mavros under a group/namespace breaks the wiring. The planner's
  `drone_name` arg is only used for logging and `/px4_drone/planned_trajectory`.
- **Frames are handled by mavros.** The adapter reads odom as ENU and
  publishes ENU data; mavros's `setpoint_raw` plugin converts ENU→NED
  internally. Do not add manual frame conversions.

Sanity check:

```bash
rostopic echo -n1 /mavros/state              # connected: True
rostopic hz /mavros/local_position/odom      # ~30 Hz once EKF has a local position
```

## 3. Planner

```bash
roslaunch comando_planner planner_launch.launch platform:=mavros drone_name:=px4_drone
```

On startup the MAVROS adapter runs a 10 Hz heartbeat that:

1. streams position-hold setpoints for a 2 s warmup (PX4 requires a >2 Hz
   setpoint stream before accepting OFFBOARD),
2. requests OFFBOARD, then ARM, retrying at ≤1 Hz until `/mavros/state`
   confirms both,
3. keeps re-streaming the last command (or the hold) forever after, so PX4
   never leaves OFFBOARD due to setpoint starvation — including while the
   planner is paused waiting for a `command_seq`.

Safety behavior: OFFBOARD/ARM is only auto-requested during the *initial*
engagement. If the FCU later disarms or changes mode (pilot takeover,
failsafe, auto-disarm after landing), the adapter logs it and does **not**
fight back.

`hover_thrust` (default 0.3) only scales thrust for **body-rate** OCPs
(`tracking_bodyrate_*`); full-state OCPs (`hover`, `landing`, `stateswitch`,
`stc_landing`) send position/velocity/yaw targets that PX4's own position
controller tracks, so `hover_thrust` is irrelevant for them.

## 4. Triggering OCPs

There is **no `ocp_launch.launch` on this branch** (the `ocp_launch.py` in
`launch/` is a ROS 2 file and does not work under Noetic). OCP profiles are
plain parameters on the node's private namespace `/comando_planner/`,
triggered by bumping `command_seq`:

```bash
rosparam set /comando_planner/ocp_type hover
rosparam set /comando_planner/mode mpc
rosparam set /comando_planner/n_replay 4
rosparam set /comando_planner/hover_target_x 0.0
rosparam set /comando_planner/hover_target_y 0.0
rosparam set /comando_planner/hover_target_z 1.8
rosparam set /comando_planner/command_seq 1     # bump by +1 to trigger
```

or use the helper (loads a yaml + bumps `command_seq` automatically):

```bash
rosrun comando_planner apply_profile.sh                          # hover profile
rosrun comando_planner apply_profile.sh /comando_planner my.yaml # custom profile
```

The default profile is `config/profile_hover_sitl.yaml`.

### Target-relative OCPs (`stateswitch`, `stc_landing`)

These need a target on `/target/odom` + `/target/accel`. For a simulated
target use the target publisher (a static or circular virtual target is
enough for smoke tests):

```bash
roslaunch comando_planner target_launch.launch \
  target_trajectory:=circle \
  drone_odom_topic:=/mavros/local_position/odom \
  drone_pose_topic:=/mavros/local_position/pose
```

then:

```bash
rosrun comando_planner apply_profile.sh /comando_planner \
  $(rospack find comando_planner)/config/profile_stc_landing_sitl.yaml
```

For the no-aug interval arm (`stc_landing_noaug`, 14-state, hard per-node
`integral(H dt) <= eps` instead of the terminal `y_N = 0` equality) use
`config/profile_stc_landing_noaug_sitl.yaml` instead. Its compiled-in
defaults are the tuned SITL setting (noaug_basin champion
`SZMUK_CTCS_STEP_EPS=1.6e-3`, `SZMUK_CTCS_Y_SCALE=10`; validated run:
`logs/stc_landing_noaug_sitl_circle/`). Two rules for this arm:

- **Engage from a general lateral offset** — hover to a staging point with a
  real lateral distance from the platform first (never from directly above,
  never from spawn). The hard interval constraint makes the activation cold
  solve the critical one; from a settled offset hover it converges
  (`test_stc_landing_noaug` checks exactly this gate).
- Its cold seed aims at `(0,0,Z_STAGE)` until laterally captured; do not
  weaken `max_constraint_error` if activation solves are rejected — fix the
  engagement geometry instead (same rule as `stc_landing`).

### Stopping after touchdown (`land_action`)

By default (`land_action: none`) terminal freeze holds the last setpoint at
the touchdown point forever — the drone hovers glued to the target. For a real
landing set **`land_action`** in the profile:

- `force_disarm` (recommended for moving platforms): immediate motor cut at
  the freeze latch (`MAV_CMD_COMPONENT_ARM_DISARM` with the force magic).
  Verified: ~1 s from touchdown to motors-off, so the platform only moves
  ~0.4 m. This is what deck-landing on a real moving platform needs.
- `auto_land`: planner stops commanding and switches PX4 to `AUTO.LAND`; PX4
  descends slowly and auto-disarms (`COM_DISARM_LAND`). Gentle, but takes ~5 s
  — a 0.4 m/s platform is 2 m away by then, so in RViz/Gazebo it looks like a
  miss (the intercept itself was on target; check the bag numbers).
- `disarm`: normal disarm request; PX4 accepts only once its land detector
  agrees (use when the drone physically rests on something).

`land_action` is a sticky rosparam — always state it explicitly in every
profile yaml (the hover profile sets `none`) so a previous landing profile
can't leak into the next command. After a completed landing, bumping
`command_seq` again while disarmed re-engages OFFBOARD+ARM for the next
flight automatically.

**Important for moving targets:** the touchdown trigger is terminal freeze,
whose default velocity gate (0.10 m/s relative) is tighter than the
benchmark's capture criterion (`SZMUK_RH_CAP_VEL` = 0.30). Against a moving
platform the intercept can pass through without latching, after which the
target drives away and re-solves from near the ground are correctly rejected
(the staging STC demands re-climbing to `Z_STAGE`, so the terminal
`y_N = 0` is infeasible from there). Launch the planner with the
benchmark-matched gate:

```bash
roslaunch comando_planner planner_launch.launch platform:=mavros \
  terminal_freeze_enter_vel:=0.30
```

Verified sequence in SITL with this gate: descent → intercept at the moving
target → freeze latch → `AUTO.LAND` → PX4 auto-disarm within ~4 s.

`stc_landing` keeps all of its benchmark `SZMUK_*` env-var tunables (horizon,
cone angles, trigger altitudes, weights…). Export them **in the planner's
terminal before `roslaunch`**, e.g.:

```bash
export SZMUK_GS_CONE_DEG=20 SZMUK_ALT_TRIG=1.0
roslaunch comando_planner planner_launch.launch platform:=mavros
```

## 5. What to watch

```bash
rostopic echo /mavros/state                  # mode: OFFBOARD, armed: True
rostopic hz /mavros/setpoint_raw/local      # ≥10 Hz always (heartbeat), ~1/ocp_dt when replaying
rostopic echo /px4_drone/planned_trajectory  # nav_msgs/Path of the active plan
```

RViz (drone + target odometry + planned trajectory; the Gazebo GUI cannot show
the virtual platform, RViz can):

```bash
roslaunch comando_planner rviz.launch        # config: rviz/comando_sitl.rviz
```

The launch also publishes the static identity transforms `odom`→`world` and
`world`→`map` that the displays need: mavros odometry is stamped `map`
(child `base_link`), the planner/target topics are stamped `world`, and the
RViz fixed frame is `odom` — without the chain, whichever frame is
disconnected renders nothing (the classic symptom: target visible, quad
missing).

### Where everything gets written

- **Planner CSVs** (`all_solves.csv`, `solver_events.csv`): the node's
  `log_dir` param defaults to `$(find comando_planner)/logs` in
  `planner_launch.launch`, so run folders appear directly in
  `CoManDO_planner/logs/<drone>_<ocp>_<mode>_<solver>_<timestamp>/`.
  (The path used to be cwd-relative — roslaunch nodes run in `~/.ros`, which
  is where old runs ended up.) The logger initializes once per planner
  process and names the folder after the FIRST profile applied, so a
  hover-then-land session logs both phases into one `*_hover_*` folder.
- **Flight bag**: recorded automatically by `planner_launch.launch`
  (`record_bag:=false` to disable) as `logs/flight_<date>.bag`; it stops when
  you Ctrl-C the planner launch. All topics `analyze_flight.py` needs are
  included.

## 8. Replaying a recorded rosbag in RViz

```bash
roscore
roslaunch comando_planner rviz.launch replay:=true   # sets /use_sim_time + TF bridge
rosbag play --clock --loop <run>/stc_landing_sitl_2m.bag
```

## 9. Post-flight analysis: paper plots from a flight bag

Runs are collected in **`CoManDO_planner/logs/<run_name>/`** (same layout as
the existing `logs/stc_landing_sitl_2m/` and `logs/hardware/` results): the
raw `flight.bag`, the planner's own `all_solves.csv` + `solver_events.csv`,
the executed-trajectory `moving_executed.csv` + `constraints.csv`, and the
figures.

The tools live in `logs/` itself — what makes which file:

| Tool | Produces |
| --- | --- |
| `analyze_flight.py` (one-shot driver) | the whole run folder: copies the bag + planner CSVs, then runs the four below |
| `bag_to_stc_csv.py` | `moving_executed.csv` + `constraints.csv` (bag → executed trajectory + constraint bounds, STC trigger signals recomputed from the `ocp_stc_landing.hpp` formulas and any `SZMUK_*` env overrides) |
| `plot_stc_paper.py` | `stc_flight_3d.png` (world + target-relative 3D) and `stc_flight_states.png` (constraint/state timelines) |
| `animate_stc_paper.py` | `stc_flight.gif` |
| `plot_sitl_planned_overlay.py` | `stc_flight_planned_overlay.png` — executed path vs the planned horizons from `/px4_drone/planned_trajectory` in the bag |
| `animate.py`, `animate_benchmark.py` | older all_solves.csv animators |

Full recipe for an STC landing run (recording is automatic — see §4):

```bash
# 1. Fly: hover profile, wait for settle, landing profile, wait for disarm,
#    then Ctrl-C the planner launch (that finalizes logs/flight_<date>.bag).

# 2. Analyze — fully automatic: picks the newest logs/flight_*.bag, moves it
#    into the planner's own run folder (which already has the CSVs), and
#    renders everything there — executed CSVs, 3D + states figures,
#    planned-vs-executed overlay, GIF. The window is trimmed automatically
#    from the LANDING OCP activation to touchdown — the hover-to-staging
#    phase is excluded from the plots.
cd $(rospack find comando_planner)/logs
python3 analyze_flight.py                     # --bag/--name to override
```

Useful `analyze_flight.py` flags: `--name` (run folder name; defaults to the
planner run folder's timestamped name), `--planner-run <dir>` to target a
specific planner log folder (auto-detection prefers `logs/` here, falling
back to `~/.ros/logs` for old runs), `--t0/--t1` to override the automatic
trim (seconds from the first odom message), `--no-gif` to skip the slow
animation, `--dpi`. The same recipe works for hardware bags — the only
inputs are `/mavros/local_position/odom` and `/target/odom` (plus
`/mavros/state` and `/rosout` for auto-trim).

## 6. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| OFFBOARD rejected (`pxh>` shows "Offboard … rejected") | Setpoint stream < 2 Hz or started after the request. The heartbeat handles this; if it persists check `rostopic hz /mavros/setpoint_raw/local` and that mavros is connected (`/mavros/state connected: True`). |
| Arming denied ("Resolve system health failures first") | Most likely the GCS-connection preflight check: `rosrun mavros mavparam set NAV_DLL_ACT 0` (see §1). Otherwise wait for EKF local position (`rostopic hz /mavros/local_position/odom`) and check `pxh> commander arm` output for the reason; for no-RC SITL set `COM_RCL_EXCEPT 4`. |
| Drone drops out of OFFBOARD mid-run | Setpoint gap. Should not happen with the heartbeat; if it does, look for planner crashes/pauses and check the adapter warning `FCU left OFFBOARD/armed`. |
| Odom never arrives | PX4 EKF has no local position yet (give the sim a few seconds) or mavros launched in a namespace (topics not at `/mavros/...`). |
| Planner solves but drone doesn't move | Not armed / not OFFBOARD (`/mavros/state`), or commands are body-rate with a bad `hover_thrust`. |
| Time weirdness | Everything runs on wall clock. Do **not** set `/use_sim_time` — PX4 SITL runs lockstep but mavros stamps in real time. |

## 7. Scaling caveat (Crazyflie-sized OCPs on an x500)

All OCP formulations in this package use **Crazyflie-scale physical
constants** (`MASS ≈ 0.027 kg`, cf thrust/moment bounds). Consequences on a
~2 kg x500:

- **Full-state OCPs work**: PX4's position controller does the actual
  tracking, and the planned position/velocity profiles are dynamically mild.
  The planned agility just reflects cf-scale thrust bounds.
- **Body-rate OCPs (`tracking_bodyrate_*`) are NOT usable without retuning**:
  the thrust command is `u(0)·hover_thrust/9.81` with cf-scale `u(0)`, and the
  moment bounds are cf-scale. Retune the OCP constants (and `hover_thrust`
  to the x500's hover throttle, `pxh> param show MPC_THR_HOVER`, ≈0.5)
  before trying.
- `mass_kg` is currently taken from the OCP descriptor (not a ROS param);
  it only affects logging and hover-hold guards, not the dynamics.
