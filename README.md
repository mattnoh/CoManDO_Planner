# CoManDO Planner 

**CoManDO (COnic-COnstrained Manifold Dynamic Optimizer)** is a high-performance, platform-agnostic MPC planner designed for complex maneuvering and dynamic target tracking. It utilizes the **ALIPDDP** (Augmented Lagrangian Interior Point Differential Dynamic Programming) solver to execute real-time optimal control.

---

## Architecture Overview

The system is built on a modular, **Registry-Driven Architecture**. This decouples the core solver logic from specific Optimal Control Problem (OCP) formulations.

For a deep dive into the math, see the **[Relative Dynamics & Intercept Tutorial](docs/CoManDO_planner/05_mathematical_tutorial.md)**.

- **`planner_node`**: A generic ROS2 node that handles state estimation, solver execution, and command dispatch. It is entirely agnostic of the OCP type being solved.
- **`OCPRegistry`**: A central factory where OCP descriptors are registered. Each descriptor provides the necessary callbacks for state transformation, parameter preparation, and logging.
- **Target Models**: Standardized models for circular and arbitrary target trajectories located in `include/target/`.

---

## OCP Catalog

| OCP Name | Frame | Description |
|---|---|---|
| `hover` | Absolute | Maintains a fixed (x, y, z) setpoint. |
| `landing` | Absolute | Plans a time-optimal descent to a static ground coordinate. |
| `stateswitch` | Relative | Actively intercepts a target using live odometry and acceleration feedback. |
| `tracking_circle` | Relative | Tracks a circular target using an **internal analytical model** (no ROS prediction needed). |
| `tracking_circle_target` | Relative | Tracks a moving target by subscribing to a **predicted trajectory** over ROS. |

---

## Installation & Build

### Prerequisites
- **ROS2 Humble**: Running on Linux (Ubuntu 22.04).
- **ALIPDDP**: Ensure the `ALIPDDP-main` repository is cloned alongside this workspace.

### Build Instructions
```bash
# Source ROS2 and your install
source /opt/ros/humble/setup.zsh
source ~/ros_ws/install/setup.zsh

# Build the planner
cd ~/ros_ws
colcon build --symlink-install --packages-select comando_planner
```

---

## Quick Start Tutorial

Follow these steps to test the **Dynamic Target Tracking** workflow using the `tracking_circle_target` OCP.

### 1. Launch the Planner Node
Start the core infrastructure. The node will enter an **IDLE** state, waiting for configuration.
```bash
ros2 launch comando_planner planner_launch.py
```

### 2. Configure and Trigger an OCP
Use the trigger script to launch the circular tracking maneuver. The planner will detect that this OCP requires target data and will wait in a **safe hover**.
```bash
ros2 launch comando_planner ocp_launch.py \
    ocp_type:=tracking_circle_target \
    mode:=open_loop
```

### 3. Start the Target Simulation
Launch the simulated target. As soon as the first prediction message hits the wire, the planner will solve and execute the intercept.
```bash
ros2 run comando_planner target_publisher
```

### 4. Analyze Results
The planner logs high-frequency data to `logs/`. You can generate a full PDF analysis with:
```bash
cd src/CoManDO_planner/logs
python plot_circle.py --dir <your_latest_log_folder>
```

---

## Hardware Integration

To use this repo with actual hardware (e.g. Vicon tracking or an onboard KF), your estimator must publish a **Predicted Trajectory**. 

See the detailed guide: [Dynamic Target Hardware Integration](docs/dynamic_target_hardware_integration.md)

---

## Logging System
Logs are automatically generated for every solver invocation:
- **`all_solves.csv`**: Contains the full state and control trajectories for every solve.
- **`metadata.csv`**: Contains solver diagnostics (iterations, solve time, etc.).
- **Auto-Cleanup**: The system keeps your workspace clean by organizing logs into timestamped directories.
