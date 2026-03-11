# 04 — Platform Bridge Architecture

---

## 4.1 Design Principle

The solver node (`comando_planner_node`) publishes a single `MpcCommand` message
and has no knowledge of how that command reaches the physical drone.
A **platform bridge** is a thin ROS 2 node that:

1. Subscribes to `/mpc/command`
2. Maintains a local copy of the trajectory and replays it at `ocp_dt` Hz
3. Subscribes to platform-native state topics, converts to ENU/FLU, and
   publishes `/mpc/state` for the solver
4. Converts the commanded state/control into whatever the platform needs
   and publishes it on the platform's native topic

This pattern means the solver is completely reusable across any platform.
The bridge is the only code that knows about the physical interface.

---

## 4.2 PlatformBridgeBase Class

All bridges inherit from `PlatformBridgeBase` (`include/platform/platform_bridge_base.hpp`).

The base class provides:

| Feature | Where |
|---------|-------|
| `/mpc/command` subscriber | `onMpcCommand()` |
| Trajectory storage + replay timer | `replayTick()` |
| Transition blend (13D LERP) | inside `replayTick()` |
| `/mpc/state` publisher | `publishMpcState()` |
| Logging via `CommandoLogger` | `replayTick()` |
| Common ROS params (`drone_name`, `ocp_dt`, `enable_logging`) | constructor |

**Pure virtual methods every bridge must implement:**

```cpp
// Convert x_cmd + u_cmd and send to the platform.
// Called by the replay timer at (1/ocp_dt) Hz.
virtual void sendPlatformCommand(const Eigen::VectorXd& x_cmd,
                                 const Eigen::VectorXd& u_cmd) = 0;

// Subscribe to platform state topics.
// Must call publishMpcState(state_13d) in each state callback.
// Called at the END of the derived constructor (not from base — C++ vtable rule).
virtual void setupPlatformIO() = 0;
```

> **Important C++ rule:** `setupPlatformIO()` must be called at the **end of the
> derived class constructor**, never from the base class constructor.
> The vtable is not fully populated during base construction, so calling a pure
> virtual there is undefined behaviour (linker error).

---

## 4.3 Crazyflie Bridge

**File:** `src/bridges/crazyflie_bridge.cpp`

### State subscription

```
/cf_1/pose  (PoseStamped)  ──► position + quaternion  ──┐
                                                          ├──► state_[0..12]
/cf_1/odom  (Odometry)     ──► velocity + angular rate ──┘
                                     ↓ DEG2RAD on angular rates
                               /mpc/state (Odometry, ENU/FLU)
```

State is forwarded only when both `pose_received_` and `odom_received_` are true.

### Command output

```
/mpc/command  ──► replay_timer ──► sendPlatformCommand()
                                        ──► FullState msg
                                              ──► /cf_1/cmd_full_state
```

`sendPlatformCommand()` fills `FullState` directly from `x_cmd[0..12]`.
Acceleration is set to zero (Mellinger computes it internally).

### Key quirk

Crazyswarm2 publishes angular rates in **deg/s** — converted to rad/s in `odomCallback()`.

---

## 4.4 PX4 Bridge

**File:** `src/bridges/px4_bridge.cpp`
**Compile guard:** `#ifdef HAS_PX4_MSGS`

### Frame conversion (NED/FRD → ENU/FLU)

All PX4 messages use NED world frame and FRD body frame.
The bridge converts incoming state and outgoing commands using `frame_conv.hpp`:

```cpp
// Position / velocity: swap x↔y, negate z
ned_to_enu(v) = { v.y(), v.x(), -v.z() }

// Quaternion
quat_ned_to_enu(w,x,y,z) = Quaternion(w, y, x, -z).normalized()

// Angular rates: FRD → FLU
omega_frd_to_flu(w) = { w.x(), -w.y(), -w.z() }
```

### Control modes

Selectable via ROS param `control_mode`:

**`"rates"` (default):** converts `u0 = (fz, Mx, My, Mz)` to `VehicleRatesSetpoint`

```
Mx_real = Mx_scaled / J_scale
desired_omega_x ≈ Mx_real / Jxx * ocp_dt   (first-order feedforward)

FLU rates → FRD: omega_flu_to_frd()
thrust_norm = -fz / FMAX   (PX4 body-z is down, normalised to [-1, 0])
```

**`"trajectory"`:** converts `x_cmd` to `TrajectorySetpoint` (pos + vel + yaw, NED)

```
pos_ned = enu_to_ned(x_cmd[0:3])
vel_ned = enu_to_ned(x_cmd[3:6])
yaw_ned = atan2(2(qw·qz + qx·qy), 1 - 2(qy² + qz²))  from NED quaternion
```

### Heartbeat

PX4 requires `OffboardControlMode` messages at >2 Hz to stay in offboard mode.
A 10 Hz heartbeat timer publishes this continuously.
The `position` and `velocity` fields are set `true` in trajectory mode,
`body_rate` is set `true` in rates mode.

### Arming

On the first `sendPlatformCommand()` call:

```
px4SetOffboardMode()  →  VehicleCommand (MAV_CMD_DO_SET_MODE, param1=1, param2=6)
px4Arm()              →  VehicleCommand (MAV_CMD_COMPONENT_ARM_DISARM, param1=1)
```

---

## 4.5 How to Add a New Platform Bridge

To add support for a new platform (e.g. DJI, Betaflight, a simulator):

**Step 1:** Create `src/bridges/<platform>_bridge.cpp`

```cpp
#include "platform/platform_bridge_base.hpp"
// Include platform-specific message headers

class MyPlatformBridge : public PlatformBridgeBase {
public:
    MyPlatformBridge() : PlatformBridgeBase("my_platform_bridge") {
        // Declare any platform-specific ROS params here

        // MUST be last line of constructor:
        setupPlatformIO();
    }

protected:
    void setupPlatformIO() override {
        // Create publishers for platform commands
        // Create subscribers for platform state
        // In state callbacks: call publishMpcState(state_13d)
    }

    void sendPlatformCommand(const Eigen::VectorXd& x_cmd,
                             const Eigen::VectorXd& u_cmd) override {
        // Convert x_cmd (ENU/FLU) and u_cmd (fz, Mx, My, Mz) to platform format
        // Publish on platform topic
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MyPlatformBridge>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
```

**Step 2:** Add to `CMakeLists.txt`

```cmake
add_executable(my_platform_bridge
  src/bridges/my_platform_bridge.cpp
)
configure_target(my_platform_bridge)

install(TARGETS my_platform_bridge
  DESTINATION lib/${PROJECT_NAME}
)
```

**Step 3:** Add to launch file

```python
my_bridge = Node(
    package='comando_planner',
    executable='my_platform_bridge',
    parameters=[{'drone_name': 'my_drone', 'ocp_dt': 0.05}]
)
```

That is the complete list of changes. The solver node is untouched.

---

## 4.6 How to Add a New OCP

**Step 1:** Create `include/ocp/ocp_<name>.hpp`

Use `ocp_hover.hpp` as a template. The file must contain:
- Physical constants (`HORIZON`, `DT`, `MASS`, ...)
- Solver params (`SOLVER_REG1_MIN`, `SOLVER_MAX_ITER`, ...)
- `StageCost<Scalar>` class
- `TerminalCost<Scalar>` class
- Any constraint classes
- `create(current_state, terminal_state, prev_U, prev_X)` factory function

**Step 2:** Add `#include` to `include/solver/ocp_registry.hpp`

```cpp
#include "ocp/ocp_<name>.hpp"
```

**Step 3:** Add three `if` branches to `ocp_registry.hpp`

```cpp
// In getDT():
if (ocp_type == "<name>") return <Name>OCP::DT;

// In getSolverParams():
} else if (ocp_type == "<name>") {
    p.reg1_min  = <Name>OCP::SOLVER_REG1_MIN;
    // ... fill all Param fields
}

// In create():
if (ocp_type == "<name>") return <Name>OCP::create(current_state, terminal_state, prev_U, prev_X);
```

**That is all.** The solver node, bridges, and CMakeLists are unchanged.
The new OCP is selectable via the `ocp_type` ROS param at launch.