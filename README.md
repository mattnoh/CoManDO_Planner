# CoManDO Planner

**CoManDO** (COnic-COnstrained Manifold Dynamic Optimizer) is a ROS2 receding-horizon MPC planner for quadrotor control. It solves aerial manipulation and dynamic target-tracking problems using the **ALIPDDP** solver (Augmented Lagrangian Interior Point Differential Dynamic Programming).

---

## Prerequisites

| Component | Notes |
|-----------|-------|
| Ubuntu 22.04 + ROS2 Humble | Base requirement |
| ALIPDDP | Clone as sibling directory: `../ALIPDDP-main` |
| **CrazySim** + **Crazyswarm2** | For Crazyflie SITL workflow |
| **PX4-Autopilot** + **MAVROS2** | For PX4 SITL + MAVROS workflow |

### Build

```bash
source /opt/ros/humble/setup.zsh
source ~/ros_ws/install/setup.zsh
cd ~/ros_ws
colcon build --symlink-install --packages-select comando_planner
```

MAVROS support is detected automatically at build time (`find_package(mavros_msgs QUIET)`). The package compiles without it; MAVROS features are guarded by `HAS_MAVROS_MSGS`.

---

## OCP Catalog

| OCP | Platform | `drone_odom_mode` | Command | Notes |
|-----|----------|-------------------|---------|-------|
| `hover` | any | `none` | CmdFullState | Fixed-point stabilization |
| `landing` | any | `none` | CmdFullState | Vertical descent |
| `stateswitch` | any | `none` | CmdFullState | Moving-target interception |
| `tracking_circle_target` | any | `none` | CmdFullState | Receding-horizon circle via predicted trajectory |
| `tracking_bodyrate_bf_noimu` | crazyflie / mavros | `body_frame` | CmdBodyRate | 10D body-frame; CF→cmd_hover, MAVROS→AttitudeTarget |
| `tracking_bodyrate_bf_imu` | crazyflie / mavros | `body_frame` | CmdBodyRate | 20D augmented (IMU states) |
| `tracking_bodyrate_tf_noimu` | crazyflie / mavros | `target_frame` | CmdBodyRate | 10D target-frame (CoNi-MPC) |
| `tracking_bodyrate_tf_imu` | crazyflie / mavros | `target_frame` | CmdBodyRate | 20D augmented (IMU states) |

---

## CrazySim SITL Workflow

```bash
# Terminal A — CrazySim (CF firmware SITL + Gazebo)
cd ~/crazysim
bash launch_crazysim.sh

# Terminal B — Crazyswarm2 (ROS2 ↔ CrazySim bridge)
ros2 launch crazyflie launch.py

# Terminal C — CoManDO planner
ros2 launch comando_planner planner_launch.py \
    drone_name:=cf_1 platform:=crazyflie

# Terminal D — Target publisher (required for tracking OCPs)
ros2 run comando_planner target_publisher --ros-args \
    -p target_mode:=circle \
    -p drone_odom_mode:=body_frame \
    -p center_z:=0.4 -p radius:=0.8 -p omega:=0.3

# Terminal E — Trigger an OCP
#   Hover at 1 m:
ros2 launch comando_planner ocp_launch.py \
    ocp_type:=hover mode:=mpc \
    hover_target_x:=0 hover_target_y:=0 hover_target_z:=1.0 command_seq:=1

#   Body-frame circular tracking (no IMU augmentation):
ros2 launch comando_planner ocp_launch.py \
    ocp_type:=tracking_bodyrate_bf_noimu mode:=mpc command_seq:=1

#   Target-frame tracking (start target_publisher with drone_odom_mode:=target_frame):
ros2 launch comando_planner ocp_launch.py \
    ocp_type:=tracking_bodyrate_tf_noimu mode:=mpc command_seq:=1
```

**`drone_odom_mode` values for `target_publisher`:**

| Value | Topic published | Used with |
|-------|----------------|-----------|
| `none` | — | absolute OCPs (hover, landing, …) |
| `shifted_world` | `/drone/relative_odometry` | `stateswitch`, `tracking_circle_target` |
| `body_frame` | `/drone/body_relative_odom` | `tracking_bodyrate_bf_*` |
| `target_frame` | `/drone/target_frame_odom` | `tracking_bodyrate_tf_*` |

---

## PX4 SITL + MAVROS Workflow

```bash
# Terminal A — PX4 SITL + Gazebo
cd ~/PX4-Autopilot
make px4_sitl gazebo-classic_iris

# Terminal B — MAVROS2 bridge
ros2 launch mavros px4.launch.py \
    fcu_url:="udp://:14540@127.0.0.1:14557" \
    gcs_url:="udp://:14550@127.0.0.1:14551"

# Terminal C — Read PX4's calibrated hover thrust (once, after first flight)
ros2 service call /mavros/param/get mavros_msgs/srv/ParamGet \
    "{param_id: MPC_THR_HOVER}"
# Note the returned value (e.g. 0.35) — pass it as hover_thrust below.
# PX4 learns this from flight history; setting it correctly is critical
# for accurate thrust normalization (T_ms2 → [0,1] throttle).

# Terminal D — CoManDO planner
ros2 launch comando_planner planner_launch.py \
    drone_name:=drone platform:=mavros \
    hover_thrust:=0.35

# Terminal E — Target publisher + OCP (same commands as CrazySim section)
```

**Pre-arm sequence (automatic):** When `platform:=mavros`, the planner publishes 2 s of neutral setpoints before switching to OFFBOARD mode and sending the ARM command. No manual steps needed after Terminal D starts.

---

## Analysis & Visualization

```bash
cd src/CoManDO_planner/logs

# Static PDF report — works with any OCP (13D or 10D/20D bodyrate)
python plot_circle.py --dir <log_dir>

# Animated body-frame view
python animate_body.py --dir <log_dir>
```

Both scripts auto-detect the column schema from the CSV headers. No hardcoded column lists.

---

## Architecture

Two-step launch: infrastructure first, OCP hot-switch second.

```
planner_node ──► StateMonitor  (thread-safe drone + target state)
             ──► OCPRegistry   (factory pattern — add OCP without touching planner_node)
             ──► QuadrotorMPC  (ALIPDDP wrapper, warmstart management)
             ──► TrajectoryReplayer  (streams commands between solves at OCP DT rate)
```

**Solve-while-replay pattern:** Solver fires every `n_replay` cycles; replayer fires at OCP DT (e.g. 50 Hz). Command output stays consistent across 100 ms+ solve times.

**Platform adapters** (`include/platform/`):

| Platform | Input topic | Output topic | Frame |
|----------|-------------|--------------|-------|
| `crazyflie` | `/{name}/pose` + `/{name}/odom` | `/{name}/cmd_full_state` or `/{name}/cmd_hover` | ENU |
| `px4` | `/fmu/out/vehicle_odometry` | `/fmu/in/trajectory_setpoint` | NED (auto-converted) |
| `mavros` | `/mavros/local_position/odom` | `/mavros/setpoint_raw/attitude` or `/mavros/setpoint_raw/local` | ENU |

**CmdBodyRate dispatch:** The OCP declares `command_mode = CmdBodyRate`. The planner dispatches by platform:
- **crazyflie** → `extract_hover_cmd` callback converts body-rate output to `[vx, vy, z, yaw_rate]` → `cmd_hover`
- **mavros** → `AttitudeTarget` (body rates + normalized thrust)
- **generic** → `TwistStamped` fallback

---

## Logging

When `enable_logging:=true`, logs go to `./logs/{drone}_{ocp}_{mode}_{solver}_{timestamp}/`:

| File | Contents |
|------|---------|
| `all_solves.csv` | Per-solve metadata + full state/control trajectories |
| `commanded_state.csv` | Per-command output (CmdFullState only) |
| `commanded_hover_state.csv` | Per-command `[vx, vy, z, yaw_rate]` (CmdBodyRate + crazyflie) |
| `commanded_bodyrate_state.csv` | Per-command `[T_ms2, ωx, ωy, ωz]` (CmdBodyRate + mavros/generic) |
| `actual_state.csv` | Raw sensor measurements |
