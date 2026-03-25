# CoManDO Planner ROS Interface Documentation

## Overview

CoManDO Planner is a platform-agnostic MPC planner for quadrotor control. This document describes the ROS interface for the **Crazyflie platform** using the **ALIPDDP solver**.

**Node name:** `comando_planner`  
**Package:** `comando_planner`

### Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           comando_planner               │
                    │                                         │
  /{drone}/pose ───►│  Sensor Callback Group                  │
                    │    └─► State mutex ──► current_state_   │
  /{drone}/odom ───►│                                         │
                    │                                         │
                    │  Solver Callback Group                  │
                    │    └─► solverLoop() ──► ALIPDDP solve   │
                    │                                         │
                    │  Replay Callback Group                  │
                    │    └─► mpcReplayTick() ──► publish      │
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                          /{drone}/cmd_full_state
```

---

## 1. ROS Parameters

### Launch Parameters

Defined in `launch/planner_launch.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drone_name` | string | `"cf_1"` | Drone namespace for topic prefixing |
| `enable_logging` | bool | `true` | Enable CSV logging to `./logs/` |
| `ocp_type` | string | `"landing"` | OCP formulation: `"landing"` or `"hover"` |
| `n_replay` | int | `4` | Setpoints sent per solve cycle |
| `platform` | string | `"crazyflie"` | Hardware platform |
| `solver` | string | `"alipddp"` | Solver backend |
| `mode` | string | `"mpc"` | Execution mode: `"mpc"` or `"openloop"` |
| `hover_target_x` | double | `0.0` | Target x position [m] |
| `hover_target_y` | double | `0.0` | Target y position [m] |
| `hover_target_z` | double | `0.0` | Target z position [m] (0.0 = land) |

### Runtime Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mass_kg` | `0.027` | Quadrotor mass [kg] |
| `ocp_dt` | `0.05` | Time step [s], set by OCP type |

### n_replay Timing

The solve interval is determined by: `n_replay × ocp_dt`

| n_replay | ocp_dt | Solve Interval |
|----------|--------|----------------|
| 4 | 50ms | 200ms |
| 10 | 50ms | 500ms |

---

## 2. Inputs (Subscriptions)

### State Estimation Topics

| Topic | Message Type | Rate | Purpose |
|-------|--------------|------|---------|
| `/{drone_name}/pose` | `geometry_msgs/msg/PoseStamped` | ~30Hz | Position + attitude |
| `/{drone_name}/odom` | `nav_msgs/msg/Odometry` | ~30Hz | Linear + angular velocity |

### Pose Message Fields

```
geometry_msgs/PoseStamped
├── header
│   ├── stamp (time)
│   └── frame_id (string)
└── pose
    ├── position
    │   ├── x (float64)  → state[0]
    │   ├── y (float64)  → state[1]
    │   └── z (float64)  → state[2]
    └── orientation
        ├── w (float64)  → state[6]
        ├── x (float64)  → state[7]
        ├── y (float64)  → state[8]
        └── z (float64)  → state[9]
```

### Odometry Message Fields

```
nav_msgs/Odometry
├── header
│   ├── stamp (time)
│   └── frame_id (string)
└── twist
    └── twist
        ├── linear
        │   ├── x (float64)  → state[3]
        │   ├── y (float64)  → state[4]
        │   └── z (float64)  → state[5]
        └── angular
            ├── x (float64)  → state[10] (deg/s → rad/s)
            ├── y (float64)  → state[11] (deg/s → rad/s)
            └── z (float64)  → state[12] (deg/s → rad/s)
```

**Note:** Crazyswarm2 publishes angular velocity in deg/s. The planner converts to rad/s:
```cpp
state(10) = msg->twist.twist.angular.x * (M_PI / 180.0);
state(11) = msg->twist.twist.angular.y * (M_PI / 180.0);
state(12) = msg->twist.twist.angular.z * (M_PI / 180.0);
```

### State Vector (13-dimensional)

| Index | Symbol | Unit | Description |
|-------|--------|------|-------------|
| 0 | x | m | Position x (ENU frame) |
| 1 | y | m | Position y |
| 2 | z | m | Position z (up positive) |
| 3 | vx | m/s | Linear velocity x |
| 4 | vy | m/s | Linear velocity y |
| 5 | vz | m/s | Linear velocity z |
| 6 | qw | - | Quaternion w (scalar) |
| 7 | qx | - | Quaternion x |
| 8 | qy | - | Quaternion y |
| 9 | qz | - | Quaternion z |
| 10 | wx | rad/s | Angular velocity x |
| 11 | wy | rad/s | Angular velocity y |
| 12 | wz | rad/s | Angular velocity z |

---

## 3. Outputs (Publications)

### Command Topic

| Topic | Message Type | Rate | Purpose |
|-------|--------------|------|---------|
| `/{drone_name}/cmd_full_state` | `crazyflie_interfaces/msg/FullState` | ocp_dt (default 50ms) | Full state command |

### FullState Message Fields

```
crazyflie_interfaces/FullState
├── header
│   ├── stamp (time)
│   └── frame_id ("world")
├── pose
│   ├── position
│   │   ├── x (float32)  ← commanded state[0]
│   │   ├── y (float32)  ← commanded state[1]
│   │   └── z (float32)  ← commanded state[2]
│   └── orientation
│       ├── w (float32)  ← commanded state[6]
│       ├── x (float32)  ← commanded state[7]
│       ├── y (float32)  ← commanded state[8]
│       └── z (float32)  ← commanded state[9]
├── twist
│   ├── linear
│   │   ├── x (float32)  ← commanded state[3]
│   │   ├── y (float32)  ← commanded state[4]
│   │   └── z (float32)  ← commanded state[5]
│   └── angular
│       ├── x (float32)  ← commanded state[10]
│       ├── y (float32)  ← commanded state[11]
│       └── z (float32)  ← commanded state[12]
└── acc
    ├── x (float32)  ← feedforward acceleration
    ├── y (float32)  ← feedforward acceleration
    └── z (float32)  ← feedforward acceleration
```

### Acceleration Feedforward

Feedforward acceleration is computed from thrust command:

```cpp
// a_world = R(q) * [0, 0, fz/m] + [0, 0, -g]
Eigen::Vector3d acc = q.toRotationMatrix() * Eigen::Vector3d(0, 0, fz / mass);
acc(2) -= 9.81;  // gravity
```

**Important:** The lower-level controller requires acceleration feedforward for good tracking. Zeroing this field causes tracking degradation.

### Visualization Topic

| Topic | Message Type | Purpose |
|-------|--------------|---------|
| `/{drone_name}/planned_trajectory` | `nav_msgs/msg/Path` | Planned trajectory for RViz |

---

## 4. Solver Interface

### Input

| Field | Type | Description |
|-------|------|-------------|
| `current_state` | `Eigen::VectorXd(13)` | Current state estimate |

### Output (QuadrotorMPC::Result)

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Solve succeeded |
| `state_trajectory` | `std::vector<Eigen::VectorXd>` | N+1 states, each 13-dim |
| `control_trajectory` | `std::vector<Eigen::VectorXd>` | N controls, each 4-dim |
| `solve_time_ms` | `double` | Solve duration [ms] |
| `solve_iters` | `int` | ALIPDDP iterations |

### Control Vector (4-dimensional)

| Index | Symbol | Unit | Description |
|-------|--------|------|-------------|
| 0 | fz_B | N | Body-z thrust force |
| 1 | Mx | N·m | Body-x moment |
| 2 | My | N·m | Body-y moment |
| 3 | Mz | N·m | Body-z moment |

---

## 5. Timing Model

### Thread Architecture

Three callback groups prevent blocking:

1. **sensor_cb_group_**: Updates state from pose/odom callbacks
2. **solver_cb_group_**: Runs MPC solver (1ms poll)
3. **replay_cb_group_**: Streams commands (ocp_dt interval)

### Solver Loop

```
Timer: 1ms poll
  └─► if (replay_ticks_since_solve_ >= n_replay)
        ├─► getCurrentState()
        ├─► solver.solve(state)
        ├─► Store trajectory in mpc_traj_
        ├─► mpc_replay_idx_ = 1 + k (k = solve_time_ms / ocp_dt)
        └─► replay_ticks_since_solve_ = 0
```

### Replay Loop

```
Timer: ocp_dt interval (default 50ms)
  └─► x_cmd = mpc_traj_[mpc_replay_idx_]
      u_cmd = mpc_ctrl_[mpc_replay_idx_]
      publishCommand(x_cmd, u_cmd)
      mpc_replay_idx_++
      replay_ticks_since_solve_++
```

### Solve-Command Relationship

```
Time:    |----|----|----|----|----|----|----|----|----|
Solve:   S1        ───────────────►  S2
         │                          │
         ├─► cmd[1]                 ├─► cmd[k+1]
         ├─► cmd[2]                 ├─► cmd[k+2]
         ├─► cmd[3]                 ├─► ...
         └─► cmd[4] (n_replay=4)    └─► cmd[k+n_replay]

k = round(solve_time_ms / ocp_dt_ms)
```

---

## 6. OCP Formulations (Reference)

### Common Structure

| Element | Dimension | Description |
|---------|-----------|-------------|
| State `x` | 13 | Position, velocity, quaternion, angular velocity |
| Control `u` | 4 | Thrust + 3 moments |
| Horizon `N` | 100 | Number of stages |

### Cost Function Structure

```
J = Σ [ stage_cost(x_k, u_k) ] + terminal_cost(x_N)
    k=0..N-1
```

**Stage Cost Components:**
- State tracking: `||x - x_ref||²_Q`
- Control effort: `||u - u_ref||²_R`
- Control slew rate: `||u - u_prev||²_S`
- Trajectory consistency (optional): `||x - x_prev||²_W`

**Terminal Cost:**
- Terminal state penalty: `||x_N - x_ref||²_P`

### Constraint Types

| Type | Symbol | Description |
|------|--------|-------------|
| Inequality | NO | Thrust limits (FMIN, FMAX) |
| Second-order cone | SOC | Glideslope, tilt cone, moment limits |

### OCP Parameters

**Landing OCP** (`ocp_landing.hpp`):
- Horizon: N = 100
- dt = 0.05s
- FMIN = 0.08 N, FMAX = 0.6 N
- Glideslope = 60°
- Tilt cone = 60°

**Hover OCP** (`ocp_hover.hpp`):
- Horizon: N = 100
- dt = 0.05s
- FMAX = 1.2 N

### Weight Matrices (Landing OCP)

```cpp
// State cost Q (13-dim diagonal)
Q_DIAG = [2.0, 2.0, 2.0,      // position
          3.0, 3.0, 4.0,      // velocity
          0.1, 0.1, 0.1, 0.1, // quaternion
          0.05, 0.05, 0.05]   // angular velocity

// Control cost R (4-dim diagonal)
R_DIAG = [1e-3, 1e-4/J², 1e-4/J², 1e-4/J²]

// Slew rate cost S (4-dim diagonal)
S_DIAG = [5e-3, 1e-1/J², 1e-1/J², 1e-1/J²]

// Terminal cost P (13-dim diagonal)
P_DIAG = [1000, 1000, 1000,  // position
          500, 500, 500,      // velocity
          500, 500, 500, 500, // quaternion
          200, 200, 200]      // angular velocity
```

---

## 7. Logging Format

### Log Directory Structure

```
./logs/{drone_name}_{ocp_type}_{mode}_{solver}_{YYYYMMDD_HHMMSS}/
├── all_solves.csv
├── commanded_state.csv
└── actual_state.csv
```

### all_solves.csv

Solver trajectory output for each solve.

| Column | Type | Description |
|--------|------|-------------|
| solve_num | int | Solve index |
| solve_time_ms | float | Solve duration |
| solve_iters | int | ALIPDDP iterations |
| node | int | Trajectory node index (0..N) |
| t | float | Time from solve start (node × dt) |
| x, y, z | float | Position [m] |
| vx, vy, vz | float | Velocity [m/s] |
| qw, qx, qy, qz | float | Quaternion |
| wx, wy, wz | float | Angular velocity [rad/s] |
| fz, mx, my, mz | float | Control (thrust, moments) |

### commanded_state.csv

Commands sent to hardware.

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | Wall clock time [s] |
| solve_num | int | Active solve index |
| [state 13] | float | Commanded state |
| [control 4] | float | Commanded control |
| acc_x, acc_y, acc_z | float | Feedforward acceleration |

### actual_state.csv

Measured state for comparison.

| Column | Type | Description |
|--------|------|-------------|
| timestamp | float | Wall clock time [s] |
| solve_num | int | Active solve index |
| [state 13] | float | Measured state |

---

## 8. File Structure

### Core Source Files

| File | Purpose |
|------|---------|
| `src/planner_node.cpp` | Main ROS node, timer callbacks, logging |
| `src/quadrotor_mpc.cpp` | ALIPDDP solver wrapper, warm-start logic |

### Header Files

| File | Purpose |
|------|---------|
| `include/quadrotor_mpc.hpp` | MPC class interface |
| `include/ocp_registry.hpp` | OCP factory and parameter lookup |
| `include/ocp_landing.hpp` | Landing problem formulation |
| `include/ocp_hover.hpp` | Hover problem formulation |
| `include/platform/crazyflie.hpp` | Crazyflie I/O abstraction |

### Launch Files

| File | Purpose |
|------|---------|
| `launch/planner_launch.py` | ROS2 launch file with parameters |

---

## 9. Integration Checklist

### Adding a New OCP Type

1. **Create OCP header** (`include/ocp_new.hpp`):
   - Define constants (HORIZON, DT, MASS, etc.)
   - Implement stage cost class
   - Implement terminal cost class
   - Implement constraint classes
   - Implement `create()` factory function

2. **Register in OCPRegistry** (`include/ocp_registry.hpp`):
   - Add to `getDT()` switch
   - Add to `getSolverParams()` switch
   - Add to `create()` factory

3. **Add solver parameters**:
   - Define SOLVER_REG1_MIN, SOLVER_REG2_MIN
   - Define SOLVER_RHO, SOLVER_RHO_MUL
   - Define SOLVER_TOLERANCE, SOLVER_MAX_ITER

4. **Update launch file** (`launch/planner_launch.py`):
   - Update ocp_type default if needed

### Testing New OCP

1. Run with logging enabled: `enable_logging:=true`
2. Check all_solves.csv for convergence
3. Compare commanded_state.csv vs actual_state.csv
4. Monitor solve_time_ms for real-time feasibility

---

## 10. Known Issues

### Angular Velocity Units

Crazyswarm2 publishes angular velocity in **deg/s**, not rad/s. The planner converts on input:
```cpp
state(10) = msg->twist.twist.angular.x * (M_PI / 180.0);
```

### State Initialization

On startup, state is initialized to:
```cpp
current_state_ = Eigen::VectorXd::Zero(13);
current_state_(6) = 1.0;  // neutral quaternion w=1
```

Wait for both pose and odom before solving begins.

### Acceleration Feedforward

Setting acceleration feedforward to zero causes poor tracking. Always compute from thrust:
```cpp
acc = R(q) * [0, 0, fz/m] + [0, 0, -g]
```
