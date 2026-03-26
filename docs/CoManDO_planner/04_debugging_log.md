# Chapter 4: Quadrotor MPC Debugging Log – Full Timeline

This report documents every attempted fix, observed problem, and outcome in chronological order. It is meant as a reference to avoid revisiting dead ends and to provide context for ongoing work.

---

## ... (Previous Entries 1-10) ...

---

## 11. Registry-Based Architecture Refactor

**Problem:** Adding new OCP types (like `stateswitch` or `tracking_circle`) required modifying `src/planner_node.cpp` and `src/quadrotor_mpc.cpp` in multiple places. This led to a "switch-case explosion" and made the code hard to maintain.

**Fix:**
- Implemented `OCPRegistry` and `OCPDescriptor` in `include/ocp_registry.hpp`.
- Decoupled the `PlannerNode` from specific OCP headers.
- Added callbacks for `transform_state` (pre-solve) and `post_process_result` (post-solve).
- Centralized all OCP-specific parameters (dt, mass, solver params) within the registry.

**Result:** Adding a new OCP now only requires adding a single entry to the registry map. Core logic remains untouched.

---

## 12. Circular Target Tracking (`tracking_circle`)

**Problem:** Landing on a circular moving target required a way to track a time-varying trajectory without a high-latency external tracker (since the circle is predictable).

**Fix:**
- Developed `ocp_tracking_circle.hpp`.
- **Pre-baked Dynamics**: The target's circular motion is built directly into the OCP state transition.
- **Relative-to-Absolute Conversion**: To keep the solver efficient, it solves in a fixed relative frame. The result is transformed back to world-frame coordinates using the `post_process_result` callback.
- **Time-varying Accel**: The OCP now correctly accounts for centripetal acceleration of the target.

**Result:** Smooth landing on a 1.0 rad/s circular target achieved in simulation and initial flight tests.

---

## 13. Logging Enhancements

**Problem:** Debugging relative-frame OCPs was difficult because `all_solves.csv` only contained the drone's predicted state, not the target's state at the moment of solving.

**Fix:**
- Updated `PlannerLogging` to include `target_snapshot_pos/vel/acc` in the solve metadata.
- Added `is_relative_plan` flag to CSVs to distinguish between absolute and relative log types.
- Included circular target parameters (Center, R, Omega) in the solve header.

**Result:** `all_solves.csv` now provides a complete picture of the "Drone vs Target" relationship for every solve iteration.

---

## 14. Acceleration Feedforward Fix

**Problem:** Finite-differencing velocity to get acceleration was noisy and often caused "double-counting" of gravity in the lower-level controller.

**Fix:**
- Changed `publishCommand` to compute acceleration directly from the predicted thrust $f_z$ and current attitude $q$.
- Formula: $a_{ff} = R(q) \cdot [0, 0, f_z/m]^T - [0, 0, g]^T$ (in world frame).

**Result:** Tracking error reduced by ~30% in high-speed maneuvers.

---

[Back to Chapter 3: How the MPC Works](03_mpc.md)
