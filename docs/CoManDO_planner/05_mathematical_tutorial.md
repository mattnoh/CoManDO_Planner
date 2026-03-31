# Chapter 5: Relative Dynamics & Intercept Math

This tutorial explains the **Relative Mathematical Model** used by the CoManDO Planner for dynamic target tracking and intercept maneuvers. Understanding the transition from absolute to relative frames is key to achieving precision intercepts.

---

## 1. Relative State Representation ($x_{rel}$)

For dynamic maneuvers like `stateswitch` or `tracking_circle`, the planner works in a **Target-Relative Frame**. This simplifies the problem by treating the target as the origin of the world.

### 1.1 The Relative Vector
The relative state $x_{rel} \in \mathbb{R}^{13}$ (or $\mathbb{R}^{14}$ in variable-time OCPs) is defined as:

$$
x_{rel} = \begin{bmatrix} p_{rel} \\ v_{rel} \\ q \\ \omega \end{bmatrix} = \begin{bmatrix} p_{drone} - p_{target} \\ v_{drone} - v_{target} \\ q_{BW} \\ \omega_{body} \end{bmatrix}
$$

- **$p_{rel}$**: Relative position (the vector from target to drone).
- **$v_{rel}$**: Relative velocity (the rate at which the distance is changing).
- **$q_{BW}$**: The drone's attitude remains in the **Absolute World Frame** to keep gravity alignment simple.

---

## 2. Relative System Dynamics

The dynamics of the drone in the relative frame must account for the **Target's Acceleration**.

### 2.1 The Relative Equation of Motion
$$
\dot{p}_{rel} = v_{rel}
$$
$$
\dot{v}_{rel} = \dot{v}_{drone} - \dot{v}_{target}
$$

Expanding the drone's acceleration (from Chapter 3/5):
$$
\dot{v}_{rel} = \left( \frac{1}{m} R(q) f_{body} + \mathbf{g} \right) - \mathbf{a}_{target}(t)
$$

- **$\mathbf{a}_{target}(t)$**: This is the most critical term. It represents the acceleration of the moving platform. 
- If the target is stationary (landing), $a_{target} = 0$, and the equations reduce to standard absolute dynamics.
- If the target is moving in a circle, $a_{target}$ is the centripetal acceleration.

---

## 3. Variable-Time OCPs ($x \in \mathbb{R}^{14}$)

Some OCPs (like `tracking_circle`) don't just optimize thrust; they optimize **Time** itself. These use an augmented 14-dimensional state.

### 3.1 The 14th State: $\theta$
In these problems, the time-step $\Delta t$ is treated as a decision variable $\theta$.

- **Decision Variable**: $u_{14} = \theta$ (The duration of the current step).
- **Time Evolution**: $x_{14, k+1} = x_{14, k} + \theta_k$.

This allows the MPC to "stretch" or "compress" the flight time to ensure the drone reaches the intercept point exactly when the target is there, while satisfying motor limits.

---

## 4. Target Models & Integration

CoManDO provides target acceleration data to the solver via specialized OCP classes like `Quad6DOFVarTimeRelativeTV`.

- **Analytical Model**: For `tracking_circle`, the target acceleration $a_{target}(t)$ is calculated analytically using sine/cosine functions of time.
- **Predicted Model**: For hardware integration, the planner subscribes to a `PredictedTrajectory` message, providing a discrete sequence of $a_{target}$ for the solver's horizon.

The solver uses **RK4 (Runge-Kutta 4th Order)** integration to evaluate these time-varying accelerations accurately at every sub-step of the plan.

---

## 5. Relative Constraints

### 5.1 Intercept Constraint (Terminal)
The goal of a tracking OCP is to achieve "Soft Intercept":
$$
p_{rel} = [0, 0, 0]^T
$$
$$
v_{rel, xy} = [0, 0]^T
$$
We often leave $v_{rel, z}$ unconstrained to allow the drone to descend vertically onto the target at a safe velocity (e.g., $v_{ref} = -0.15$ m/s).

### 5.2 Relative Glideslope
During a relative approach, the glideslope ensures the drone stays in a "Safe Cone" *relative to the target's position*, even if the target is moving at high speed:
$$
\tan(\phi) \cdot p_{rel, z} \ge \sqrt{p_{rel, x}^2 + p_{rel, y}^2}
$$
This prevents "side-swiping" the target platform.
