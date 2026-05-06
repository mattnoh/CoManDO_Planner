# Chapter 5: Frames, Relative Dynamics, and Tracking Math

The planner supports several OCP state frames. The descriptor's
`drone_odom_mode`, `transform_state`, and command adapter define how measured
state becomes an OCP state and how replay state becomes a platform command.

## Absolute 13D State

Absolute OCPs use the world-frame drone state:

```text
x = [p_W, v_W, q_WB, omega_B]
```

where:

| Segment | Meaning |
| --- | --- |
| `p_W` | Drone position in world/ENU |
| `v_W` | Drone velocity in world/ENU |
| `q_WB` | Quaternion stored as `[qw, qx, qy, qz]` |
| `omega_B` | Body angular velocity in rad/s |

The translational acceleration model is:

```text
v_dot = R(q) * [0, 0, fz / m] + [0, 0, -g]
```

## Target-Relative Absolute Form

`stateswitch` and `tracking_circle_target` read absolute drone state and target
state, then subtract target position and velocity before solving:

```text
p_rel = p_drone - p_target
v_rel = v_drone - v_target
```

Attitude and angular velocity remain drone attitude/rate. Target acceleration is
provided to the OCP through the target snapshot or an acceleration buffer.

The relative acceleration relation is:

```text
v_rel_dot = v_drone_dot - a_target
```

After solving, replay commands must be reconstructed into world coordinates:

```text
p_cmd_world = p_cmd_rel + p_target(t)
v_cmd_world = v_cmd_rel + v_target(t)
```

`stateswitch` uses the target snapshot and a zero-jerk predictor built from
position, velocity, and acceleration. `tracking_circle_target` uses the current
target odometry plus `/target/predicted_accel` integration.

## Body-Frame Relative State

Body-frame OCPs read `/drone/body_relative_odom`, usually produced by
`target_publisher`.

The helper publisher computes:

```text
p_T^B = R_BW * (p_target_W - p_drone_W)
v_rel^B = R_BW * (v_drone_W - v_target_W)
q_NB = q_target^-1 * q_drone
```

The OCP state stores the target position in the drone body frame. During
Crazyflie command conversion, the planner derives:

```text
vx_body = state[3]
vy_body = state[4]
z_world = target_z - (R_WB * p_T^B).z
```

Then it publishes Crazyflie `cmd_hover`.

## Target-Frame Relative State

Target-frame OCPs read `/drone/target_frame_odom`.

The helper publisher computes:

```text
p_B^N = R_NW * (p_drone_W - p_target_W)
v_B^N = R_NW * (v_drone_W - v_target_W) - Omega_N x p_B^N
q_NB = q_target^-1 * q_drone
```

`N` is the target frame. The Coriolis-like `Omega_N x p_B^N` term accounts for a
rotating target frame in the reported relative velocity.

For Crazyflie command conversion:

```text
v_body = R_NB^T * v_B^N
z_world = target_z + p_B^N.z
```

## Variable Time

Variable-time OCPs append cumulative time to the state and segment duration to
the control:

```text
x_aug = [x_physical, DT]
u_aug = [u_physical, Theta]
DT_{k+1} = DT_k + Theta_k
```

The `TrajectoryReplayer` uses the cumulative `DT` slot as the node timestamp
when `descriptor.variable_dt` is true. This lets the OCP optimize timing while
the replay sampler still interpolates by wall-clock time.

## Glideslope and Safety Constraints

Several tracking OCPs encode a cone-like approach constraint. In target-relative
form, a typical glideslope condition is:

```text
sqrt(px_rel^2 + py_rel^2) <= tan(phi) * pz_rel
```

The exact constants and constraint forms live in the OCP headers. Some relative
OCPs set `skip_trajectory_validation` because generic world-frame checks such as
negative altitude are not meaningful for raw body-relative or target-relative
state vectors.

## Target Prediction Ownership

Target prediction is deliberately OCP-specific:

| OCP | Target prediction source |
| --- | --- |
| `stateswitch` | Snapshot position, velocity, and acceleration through an OCP-owned predictor |
| `tracking_circle_target` | `/target/predicted_accel` buffer plus solve-start target odometry |
| `tracking_bodyrate_*` | Current target snapshot and OCP state augmentation |
| `hover`, `landing` | No target prediction |

[Next Chapter: Online Replanning and Trajectory Replay](06_online_replanning.md)
