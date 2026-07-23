# Running CoManDO on CrazySim SITL (crazyflie arm, ROS 2 Humble)

Validated baseline: 2026-07-23, `stc_landing_noaug` intercept of the R=1 m
circling target — **17/17 solves accepted, 0 rejected**, lateral error
3.79 m → 0.04 m over ~4 s of monotonic descent. Reference CSVs:
`logs/crazysim_stc_landing_noaug_baseline/` (this repo). This is the CrazySim
counterpart of the ROS 1 guide `docs/MAVROS_SITL.md` and its
`logs/stc_landing_noaug_sitl_circle` reference run (13/13).

Written to be executable by a human across five terminals **or by an agent
from a single shell** — if you are an agent, read
[Agent runbook (single shell)](#agent-runbook-single-shell) first and use the
five-terminal section as the reference for what each stage means.

## Agent runbook (single shell)

Rules that make this work unattended; each was learned from an actual failure:

1. **Shell**: run every ROS 2 command inside `bash -lc '...'` (interactive
   shells here may be zsh, where `setup.bash` fails with
   `no such file: .../setup.sh`). If you use `set -u`, wrap the two `source`
   lines in `set +u` … `set -u` — ROS setup scripts are not nounset-clean.
2. **One environment prelude** in every shell you spawn:
   `export ROS_DOMAIN_ID=77; source /opt/ros/humble/setup.bash; source /home/lab/projects/cf/online/ros_ws/install/setup.bash`.
3. **Background each stage, redirect its output to a log file you keep**, and
   gate on a readiness condition — never on sleep alone:
   - CrazySim ready: `pgrep -x cf2` returns a pid (≤30 s).
   - crazyswarm2 ready: `/cf_1/pose` in `ros2 topic list` (≤40 s).
   - planner ready: `comando_planner` in `ros2 node list` (≤30 s).
   - target ready: `/target/odom` **and** `/target/accel` present (≤20 s).
4. **Trigger with explicit param sets, not `ocp_launch.py`.** Its sequential
   `ros2 param set` subprocesses can partially fail under slow DDS, bumping
   `command_seq` while `ocp_type` is still empty — the planner then aborts
   (`Unknown OCP type:`). Do instead:
   ```bash
   ros2 param set /comando_planner ocp_type stc_landing_noaug
   ros2 param set /comando_planner mode mpc
   ros2 param set /comando_planner n_replay 4
   # VERIFY before bumping — refuse to trigger otherwise:
   ros2 param get /comando_planner ocp_type      # must print stc_landing_noaug
   ros2 param set /comando_planner command_seq 1 # strictly increasing per session
   ```
5. **Watch the descent ~24 s**, polling
   `ros2 topic echo --once --field pose.position.z /cf_1/pose` (note: `--once`,
   not `-n1`), and each poll also check the planner is still alive
   (`pgrep -f "comando_planner --ros-args"`) — it dies silently on OOM.
6. **Evaluate the pass criteria table below from the planner log/CSVs** before
   claiming success. "The drone is on the floor" is NOT success — a run with
   1 accept / 64 rejects also ends on the floor (open-loop ride +
   uncommanded descent).
7. **Clean up on every exit path**, including failure: kill `cf2`, the
   CrazySim `gz sim` (`pgrep -f crazysim_default`), `crazyflie_server`,
   `comando_planner`, `target_publisher`, **and their `ros2 launch` parents**
   (they respawn children). Verify with
   `pgrep -af "crazysim_default|comando_planner|crazyflie_server|cf2$"`.
   Never kill other `gz sim` instances (e.g. a PX4 session may be running).
8. On failure, jump to the [failure catalogue](#failure-catalogue-each-observed-each-individually-fatal)
   — every known symptom is listed with its cause; do not re-tune the OCP
   before checking it.

## Prerequisites

| Component | Location | Notes |
| --- | --- | --- |
| CrazySim | `/home/lab/projects/cf/online/CrazySim` | firmware SITL prebuilt at `crazyflie-firmware/sitl_make/build/cf2` |
| This workspace | `/home/lab/projects/cf/online/ros_ws` | `colcon build --packages-select comando_planner`, then `source install/setup.bash` |
| crazyswarm2 | in this workspace | `crazyflies.yaml` `cf_1: uri: udp://0.0.0.0:19850` (CrazySim's cflib port) |

Shell note: ROS 2 `setup.bash` breaks under zsh (`no such file or directory:
.../setup.sh`). Use a bash shell (or wrap commands in `bash -lc '...'`).

Every terminal below assumes:

```bash
export ROS_DOMAIN_ID=77          # any consistent value; isolates from other ROS sessions
source /opt/ros/humble/setup.bash
source /home/lab/projects/cf/online/ros_ws/install/setup.bash
```

## The five terminals

### 1. CrazySim firmware + Gazebo

```bash
cd /home/lab/projects/cf/online/CrazySim/crazyflie-firmware
bash tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_singleagent.sh -m crazyflie -x 0 -y 0
```

Ready when `pgrep -x cf2` returns a pid (~10 s). The script kills prior `cf2`
instances itself. Firmware UDP 19950, cflib UDP 19850 (`ss -lun | grep 198`).

### 2. crazyswarm2 — cflib backend

```bash
ros2 launch crazyflie launch.py backend:=cflib gui:=False mocap:=False
```

**Not `backend:=sim`** — that is the pure-python integrator: no firmware, no
`/cf_1/pose`. Ready when `ros2 topic hz /cf_1/pose` shows ~100 Hz.
(`'SyncCrazyflie' object has no attribute 'status'` spam is a cflib version
mismatch; harmless for this flow.)

### 3. Planner

```bash
ros2 launch comando_planner planner_launch.py \
  drone_name:=cf_1 platform:=crazyflie record_bag:=false
```

- **`record_bag:=false` is mandatory** — bag recording OOM-kills the node
  (exit -9, no error message).
- The launch file injects `SZMUK_Z_STAGE` / `SZMUK_LOS_ALT_TRIG` via
  `additional_env` (args `stc_z_stage` / `stc_los_alt_trig`); without them the
  stc OCPs stall from the documented staging geometry (constraint ~30–60,
  endless "Outer Max/Min").
- Starts paused/unconfigured; note the `Logging to: ./logs/...` line — that
  folder (relative to the launch cwd) is where `all_solves.csv` /
  `solver_events.csv` land.

### 4. Target publisher

```bash
ros2 launch comando_planner target_launch.py \
  target_mode:=gazebo_circle planning_frame:=world \
  drone_odom_topic:=/cf_1/odom drone_pose_topic:=/cf_1/pose
```

Check **both** `/target/odom` and `/target/accel` at ~100 Hz — the tracker's
freshness gate needs both.

### 5. Position, then trigger

Position with the crazyswarm2 high-level commander, **not** the hover OCP —
`planner_node.cpp` relies on the goto holding the drone until the first
landing solve arrives:

```bash
ros2 service call /cf_1/arm crazyflie_interfaces/srv/Arm "{arm: true}"
sleep 1
ros2 service call /cf_1/takeoff crazyflie_interfaces/srv/Takeoff \
  "{group_mask: 0, height: 1.0, duration: {sec: 3, nanosec: 0}}"
sleep 5
ros2 service call /cf_1/go_to crazyflie_interfaces/srv/GoTo \
  "{group_mask: 0, relative: false, goal: {x: 2.0, y: 2.0, z: 1.8}, yaw: 0.0, duration: {sec: 6, nanosec: 0}}"
sleep 9
ros2 topic echo --once --field pose.position.z /cf_1/pose    # expect ~1.8
```

Then the landing:

```bash
ros2 launch comando_planner ocp_launch.py \
  ocp_type:=stc_landing_noaug mode:=mpc n_replay:=4 command_seq:=1
```

- **`n_replay:=4`.** See the failure table below for why 7 kills it.
- `command_seq` must strictly increase per planner session — use 2, 3, … on
  re-triggers. If nothing happens, `ros2 param get /comando_planner
  command_seq` and check the planner actually saw the bump.
- ROS 1-style flat profile YAMLs do **not** work with `ros2 param load`
  (needs `/comando_planner:\n  ros__parameters:`) and fail silently —
  `config/profile_stc_landing_noaug_sitl.yaml` here is already in ROS 2 form.

## Pass criteria (what "working" means)

From the planner stdout / the run's CSV folder:

| Check | Baseline value |
| --- | --- |
| `solve_accepted` count | ~17 (activation + every replan) |
| `solve_rejected` count during descent | **0** (activation-phase geometry waits are OK) |
| `replan_delay_sec / theta` in `solver_events.csv` | **= 4** (= n_replay) |
| solve iters during descent | 150–180 (full effort, not 27-iter bails) |
| node-0 lateral error over solves | monotonic 3.79 → 0.04 m |
| rel z over solves | monotonic 1.60 → 0.16 m |

Baseline CSVs to diff against: `logs/crazysim_stc_landing_noaug_baseline/`.

An earlier run that "landed" with **1 accept / 64 rejects** was NOT working:
it rode the activation plan open-loop, that plan ended 0.47 m above the pad
with 0.45 m lateral error, and the final descent was uncommanded. Accept/reject
counts are the test — not whether the drone ends up on the floor.

## Failure catalogue (each observed, each individually fatal)

| Mistake | Symptom |
| --- | --- |
| `n_replay:=7` (descriptor NEX) | 1 accept / 64+ rejects; constraint ~5e5; solver bails at 27 iters. Warm start shifts 7·0.12=0.84 s but x0 leads only ~0.26 s → ~0.6 s desync. |
| `record_bag` left true | planner dies exit -9 (OOM) mid-run |
| `backend:=sim` | no `/cf_1/pose`, planner never gets state |
| positioning via hover OCP | solves rejected / drone never moves; use arm→takeoff→go_to |
| missing `SZMUK_*` env (launch defaults removed) | activation stalls, constraint 30–60, "Outer Max/Min" forever |
| flat ROS 1 profile YAML via `ros2 param load` | silent no-op; `command_seq` stays 0; "the OCP is not starting" |
| `ocp_launch.py` under slow DDS (automation) | partial param application — `command_seq` bumped with `ocp_type` empty → planner aborts `Unknown OCP type:`. Fine interactively; agents use explicit verified param sets (runbook rule 4). |
| staging below the relative floor (e.g. world 1.8 with `Z_STAGE=1.8` rel) | activation rejected `constraint_error` ~58; raise staging or use the launch-arg floors |
| MAVROS horizon tuning `THH=0.16` on CF | re-breaks n_replay=4 (4·0.16=0.64 s shift vs 0.23 s lead → 1 accept/91 reject). THH/RH_N are per-platform; CF stays on compiled defaults. |

## Known gap

`land_action` is not a declared parameter on the ROS 2 node — there is no
touchdown disarm. At terminal freeze the drone holds a hover at pad height and
the (virtual) platform circles out from under it. The intercept itself is
correct; only the motor-cut is missing. Port from the ROS 1 planner when
needed.

## Cleanup

Kill, in any order: the `cf2` process, the CrazySim `gz sim` server
(`pgrep -f crazysim_default`), both crazyswarm2 servers, the planner and
target_publisher — and their `ros2 launch` parents, which otherwise respawn
the nodes. Verify nothing is left with:

```bash
pgrep -af "crazysim_default|comando_planner|crazyflie_server|cf2$"
```
