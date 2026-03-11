# CoManDO Planner — Documentation Index

> **CoManDO**: Constrained Model-based Nonlinear Dynamics Optimizer
> Platform: ROS 2 Humble | Crazyflie (Crazyswarm2) | PX4 uORB-over-DDS
> Solver: ALIPDDP (Augmented Lagrangian Interior-Point DDP) / Acados SQP-RTI

---

## Chapters

| # | File | Contents |
|---|------|----------|
| — | `index.md` | This file — overview, architecture, state convention |
| 01 | `01_crazyflie_messaging.md` | CF message types, trajectory dispatch, deg/s conversion |
| 02 | `02_mpc_timing.md` | Dual-timer architecture, latency compensation, skip index |
| 03 | `03_ocp_formulations.md` | Hover & landing OCP, cost matrices, constraints, warm-start |
| 04 | `04_platform_bridges.md` | Bridge base class, CF bridge, PX4 bridge, adding new platforms |
| 05 | `05_solver_tuning.md` | ALIPDDP and Acados parameter guide |
| 06 | `06_sim_and_warmstart.md` | Sim bridge usage, noise modes, RH warm-start fix (Jordan algebra) |

---

## System Design Philosophy

CoManDO separates the MPC solver from all platform-specific code via a message-passing boundary.
The solver node knows nothing about Crazyflie or PX4 — it speaks only in physics.
Platform bridges translate between the solver's internal representation and each platform's native interface.

Adding a new platform = **write one bridge file, touch nothing else.**
Adding a new OCP = **write one OCP header, add three lines to `ocp_registry.hpp`.**

---

## ROS 2 Topic Graph

```
                        /mpc/state
                    (nav_msgs/Odometry)
                            │
              ┌─────────────┘
              │  published by bridge, consumed by solver
              ▼
┌─────────────────────────────────────┐
│       comando_planner_node          │
│                                     │
│  solver_timer  (solver_rate Hz)     │
│    └─► runs MPC                     │
│    └─► publishes /mpc/command       │
└────────────────┬────────────────────┘
                 │ /mpc/command
                 │ (MpcCommand.msg)
        ┌────────┴─────────┐
        ▼                  ▼
┌──────────────┐   ┌──────────────┐
│  crazyflie   │   │  px4_bridge  │   ← any number of bridges
│  _bridge     │   │              │     can coexist on the bus
└──────┬───────┘   └──────┬───────┘
       │                  │
       │ replay_timer      │ replay_timer
       │ (ocp_dt Hz)       │ (ocp_dt Hz)
       ▼                  ▼
  cmd_full_state    VehicleRates
  (Crazyswarm2)     Setpoint / Traj
                    Setpoint (PX4)

       ▲                  ▲
       │                  │
  /cf_1/pose         /fmu/out/
  /cf_1/odom         vehicle_odometry
       │                  │
       └────────┬─────────┘
                │ converted to ENU/FLU
                ▼
           /mpc/state
```

**Topic summary:**

| Topic | Type | Direction | Notes |
|-------|------|-----------|-------|
| `/mpc/state` | `nav_msgs/Odometry` | bridge → solver | ENU/FLU, 13-dim state packed into pose+twist |
| `/mpc/command` | `MpcCommand` | solver → bridge | full X,U trajectory + u0 + replay metadata |
| `/<drone>/cmd_full_state` | `FullState` | CF bridge → drone | Crazyswarm2 full-state setpoint |
| `/fmu/in/vehicle_rates_setpoint` | `VehicleRatesSetpoint` | PX4 bridge → FMU | rates mode |
| `/fmu/in/trajectory_setpoint` | `TrajectorySetpoint` | PX4 bridge → FMU | trajectory mode |
| `/fmu/out/vehicle_odometry` | `VehicleOdometry` | FMU → PX4 bridge | NED/FRD, converted to ENU/FLU |
| `/<drone>/planned_trajectory` | `nav_msgs/Path` | solver → RViz | visualisation only |

---

## State Vector Convention

All internal state is **13-dimensional**, in **ENU** (East-North-Up) world frame and **FLU** (Forward-Left-Up) body frame.

```
x ∈ ℝ¹³
```

| Index | Symbol | Description | Unit |
|-------|--------|-------------|------|
| 0 | `px` | position x (East) | m |
| 1 | `py` | position y (North) | m |
| 2 | `pz` | position z (Up) | m |
| 3 | `vx` | velocity x | m/s |
| 4 | `vy` | velocity y | m/s |
| 5 | `vz` | velocity z | m/s |
| 6 | `qw` | quaternion scalar | — |
| 7 | `qx` | quaternion x | — |
| 8 | `qy` | quaternion y | — |
| 9 | `qz` | quaternion z | — |
| 10 | `ωx` | angular rate x (roll) | rad/s |
| 11 | `ωy` | angular rate y (pitch) | rad/s |
| 12 | `ωz` | angular rate z (yaw) | rad/s |

**Upright hover quaternion:** `[qw=1, qx=0, qy=0, qz=0]`

**Control vector** `u ∈ ℝ⁴`:

| Index | Symbol | Description | Unit |
|-------|--------|-------------|------|
| 0 | `fz` | collective thrust in body-z | N |
| 1 | `Mx` | body moment x (inertia-scaled) | Nm·J_scale |
| 2 | `My` | body moment y (inertia-scaled) | Nm·J_scale |
| 3 | `Mz` | body moment z (inertia-scaled) | Nm·J_scale |

> **Inertia scaling:** Moments are scaled by `J_scale = 1 / 1.66e-5 ≈ 60240`
> so that all entries in the control vector are O(1), improving solver conditioning.
> Physical moment = `M_scaled / J_scale`.

---

## Directory Structure

```
comando_planner/
│
├── index.md                          ← this file
├── CMakeLists.txt
│
├── msg/
│   └── MpcCommand.msg                ← solver → bridge message
│
├── include/
│   ├── ocp/
│   │   ├── ocp_base.hpp              ← shared includes for all OCPs
│   │   ├── ocp_hover.hpp             ← hover stabilisation OCP
│   │   └── ocp_landing.hpp           ← landing OCP
│   │
│   ├── solver/
│   │   ├── ocp_registry.hpp          ← factory: string → OCP
│   │   ├── quadrotor_mpc.hpp         ← ALIPDDP solver interface
│   │   └── acados_solver.hpp         ← Acados solver interface
│   │
│   ├── platform/
│   │   └── platform_bridge_base.hpp  ← abstract bridge base class
│   │
│   └── utils/
│       ├── logger.hpp                ← CSV logging (same format as original)
│       └── frame_conv.hpp            ← NED↔ENU / FRD↔FLU helpers
│
├── src/
│   ├── solver/
│   │   ├── comando_planner_node.cpp  ← CoManDO solver node (platform-agnostic)
│   │   ├── quadrotor_mpc.cpp         ← ALIPDDP implementation
│   │   └── acados_solver.cpp         ← Acados implementation
│   │
│   └── bridges/
│       ├── crazyflie_bridge.cpp      ← Crazyswarm2 bridge
│       └── px4_bridge.cpp            ← PX4 uORB bridge
│
├── launch/
│   └── (existing launch files)
│
└── acados_generated/
    └── c_generated_code/
```

---

## MpcCommand Message

The `MpcCommand.msg` is the contract between the solver and every platform bridge.
Bridges only need to read this message — they never need to understand the OCP internals.

```
Header header

bool    success
float64 solve_time_ms

float64 ocp_dt           # seconds per OCP step — sets bridge replay timer period
int32   horizon          # N: state traj has N+1 entries, control traj has N
int32   nx               # state dimension (13)
int32   nu               # control dimension (4)
int32   replay_start_idx # latency-compensated start index (see 02_mpc_timing.md)

float64[] state_trajectory    # flattened row-major: (N+1)*13 elements
float64[] control_trajectory  # flattened row-major: N*4 elements

float64 fz   # u0[0] — collective thrust (N)
float64 mx   # u0[1] — moment x
float64 my   # u0[2] — moment y
float64 mz   # u0[3] — moment z
```