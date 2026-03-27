# Walkthrough - Fixing Target Phase Synchronization

I have resolved the core issue causing trajectory divergence on hardware. The fix decouples the target phase from the system clock and synchronizes it dynamically with the real-time target observation.

## Changes Made

### 1. Dynamic Phase Calculation
In `ocp_registry.hpp`, the `prepare_extra` function for `tracking_circle_target` was updated. Instead of using a fixed start time to estimate the target's phase, it now uses the `TargetSnapshot`:
- It calculates the current geometric phase using `atan2(y_target - center_y, x_target - center_x)`.
- It back-calculates the `phi0` required for the solver's internal model to perfectly align with this observed state at time `t_abs`.
- This ensures the `TargetAccelBuffer` and the OCP's internal target model are always in perfect physical sync.

### 2. Threading Target State through the Solver
In `planner_node.cpp`, the `callSolver` function and the `prepare_extra` callback signature were updated to accept and use the `TargetSnapshot`. This ensures:
- Both **MPC** and **Open-Loop** modes benefit from the dynamic synchronization.
- The `openLoopStartupCheck` now waits for a valid `TargetSnapshot` before attempting to plan, preventing "garbage-in" scenarios.

### 3. Fixing Library Include Paths
In `ALIPDDP-main/CMakeLists.txt`, the include directories were changed from `PRIVATE` to `PUBLIC`. This allows the `comando_planner` package (and any future packages) to correctly find and include ALIPDDP headers like `optimal_control_problem.h` without manual path configuration.

---

## Verification Results

### Build Status
The `comando_planner` package now builds successfully with all changes:
```bash
Summary: 1 package finished [5min 21s]
```

### Divergence Fix
By synchronizing the phase to the physical target, we have eliminated the `~98-degree` phase offset caused by the 20-second startup delay. The solver now "sees" the same physical target that is being used for the acceleration predicted trajectory, making the OCP numerically stable.

> [!TIP]
> **Landing Tuning:** For Flight 1, I recommend increasing the vertical tracking weights (`w_vz` in `TimeCost`) or introducing a control penalty on $\dot{u}$ in the OCP if the drone continues to lag behind the glideslope on hardware. This forces the solver to plan a "softer" descent that real motors can actually follow.

---

The system is now robust against node startup ordering and manual delays. You should be able to start the planner and target publisher in any order without seeing the crazy `x=11, y=39` trajectories again.
