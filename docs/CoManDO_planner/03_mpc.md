# Chapter 3: How the MPC Works

This chapter describes the internal mechanics of `QuadrotorMPC`, the `OCPRegistry` architecture, and the mathematical foundations of the problem formulations.

---

## 1. Problem Formulation

The MPC solves an **Optimal Control Problem (OCP)** over a finite horizon of $N$ steps, producing an optimal state trajectory $X$ and control sequence $U$.

### 1.1 State Space (13D)
The state vector $x \in \mathbb{R}^{13}$ is defined as:
- **0–2**: Position $p_W$ (World frame, ENU)
- **3–5**: Velocity $v_W$ (World frame, ENU)
- **6–9**: Unit Quaternion $q_{BW}$ ($w, x, y, z$)
- **10–12**: Angular Rate $\omega_B$ (Body frame, rad/s)

### 1.2 Control Space (4D)
The control vector $u \in \mathbb{R}^4$ is defined as:
- **0**: Total Thrust $f_z$ (along body-z axis)
- **1–3**: Body Moments $M_x, M_y, M_z$

> [!NOTE]
> Previous iterations used 6 controls. This was reduced to 4 to match the physical actuators of a quadrotor, which cannot produce independent lateral forces.

---

## 2. The OCP Registry Architecture

To support diverse flight behaviors (Hover, Landing, Moving Target Tracking) without bloating the core planner code, we use a **Registry Pattern**.

### 2.1 OCPDescriptor
Each OCP is defined by an `OCPDescriptor` in `include/ocp_registry.hpp`, containing:
- **`dt`**: The time-step for this specific formulation.
- **`transform_state`**: A callback to modify the current state before solving (e.g., converting world-frame position to relative-frame for `stateswitch`).
- **`post_process_result`**: A callback to modify the solver's output (e.g., converting a relative trajectory back to absolute world coordinates for `tracking_circle`).
- **`create`**: A factory function that instantiates the concrete `OptimalControlProblem`.

---

## 3. Advanced OCP Formulations

### 3.1 `stateswitch` (Moving Target)
- **Frame**: Relative (Drone - Target).
- **Dynamics**: Includes target velocity in the relative state transition.
- **Variable Time**: Includes a time-step $\theta$ as a decision variable to optimize landing timing.

### 3.2 `tracking_circle` (Circular Target)
- **Problem**: Pre-calculates a circular target trajectory.
- **Baking**: The relative dynamics are "baked-in" using `TargetSnapshot` parameters (Center, Radius, Omega).
- **Output**: The solver returns a relative-frame solution, which the Registry's `post_process_result` converts back to an absolute world-frame trajectory using the known target motion model.

---

## 4. Receding Horizon & Warm-Starting

The planner uses **Warm-Starting** to achieve real-time performance.

### 4.1 The Shift Count (`n_shift`)
Between two solves, the drone moves forward by `n_shift` steps of the previous plan.
To provide a good initial guess for the next solve, the previous solution $(X, U, K)$ is shifted forward:
- $U_{warm}[k] = U_{prev}[k + n\_shift]$
- $X_{warm}[k] = X_{prev}[k + n\_shift]$
- $K_{warm}[k] = K_{prev}[k + n\_shift]$ (Feedback gains)

### 4.2 Handling Ambiguity
Warm-starting feedback gains ($K$) is critical when using ALIPDDP. It ensures the first backward pass of the new solve starts from a nearly optimal control law, often reducing the iteration count by 50-80% compared to a cold start.

---

## 5. Solver: ALIPDDP

The backend solver is **ALIPDDP (Augmented Lagrangian Iterative Parabolic Differential Dynamic Programming)**.
- **Dynamics**: Second-order rigid-body dynamics.
- **Constraints**: Handles thrust limits, tilt constraints, and glideslope constraints via the Augmented Lagrangian method.
- **Hot-Swap**: The solver object is recreated on every solve to ensure a clean internal workspace, while the `OptimalControlProblem` is reused to preserve the objective function landscape.

---

[Next Chapter: Debugging Log](04_debugging_log.md)