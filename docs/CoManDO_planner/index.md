# CoManDO Planner Documentation

These chapters document the current `comando_planner` package as implemented in
this repository. The package is a ROS 2 Humble planner node around ALIPDDP OCPs,
with platform adapters for Crazyflie and MAVROS and helper target-publishing
tools for dynamic target experiments.

## Reading Order

1. [Crazyflie, MAVROS, and Target ROS Interfaces](01_crazyflie_ros_nodes.md)
2. [Planner Node Setup and Runtime Lifecycle](02_planner_node_setup.md)
3. [MPC, OCP Registry, and Command Modes](03_mpc.md)
4. [Logging, RViz, Bags, and Debugging](04_debugging_log.md)
5. [Frames, Relative Dynamics, and Tracking Math](05_mathematical_tutorial.md)
6. [Online Replanning and Trajectory Replay](06_online_replanning.md)

For exact topic and parameter tables, also read
[`../COMANDO_PLANNER_INTERFACE.md`](../COMANDO_PLANNER_INTERFACE.md). For
external target hardware integration, read
[`../dynamic_target_hardware_integration.md`](../dynamic_target_hardware_integration.md).

## Current Registered OCPs

```text
hover
landing
stateswitch
tracking_circle_target
tracking_bodyrate_tf_noimu
tracking_bodyrate_tf_imu
tracking_bodyrate_bf_imu
tracking_bodyrate_bf_noimu
```

There is no currently registered `tracking_circle` OCP key. Use
`tracking_circle_target` with an external predicted-acceleration stream, or edit
the registry and OCP headers if you reintroduce an analytic circle OCP.
