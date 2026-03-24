/// @file px4.hpp
/// @brief PX4 platform abstraction - direct subscribe/publish.
///
/// LESSONS FROM CoManDO BRIDGE FAILURE:
/// ─────────────────────────────────────
/// 1. NO message_filters synchronization - adds latency and drops messages
/// 2. NO intermediate bridge node - direct subscribe/publish minimizes latency
/// 3. PX4 uses NED frame, planner uses ENU - convert on input/output
/// 4. PX4 needs heartbeat timer for offboard mode
///
/// Frame conversions:
/// - Position: ENU (x=E, y=N, z=Up) ↔ NED (x=N, y=E, z=Down)
/// - Rotation: q_ENU ↔ q_NED via quaternion transform

#pragma once

#ifdef HAS_PX4_MSGS

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <cmath>
#include <limits>

namespace platform {
namespace px4 {

// ─────────────────────────────────────────────────────────────────────────────
// Frame conversions: ENU (planner) ↔ NED (PX4)
// ─────────────────────────────────────────────────────────────────────────────
namespace frame_conv {

inline Eigen::Vector3d ned_to_enu(const Eigen::Vector3d& v) {
    return {v.y(), v.x(), -v.z()};
}

inline Eigen::Vector3d enu_to_ned(const Eigen::Vector3d& v) {
    return {v.y(), v.x(), -v.z()};
}

inline Eigen::Quaterniond quat_ned_to_enu(double w, double x, double y, double z) {
    return Eigen::Quaterniond(w, y, x, -z).normalized();
}

inline Eigen::Quaterniond quat_enu_to_ned(const Eigen::Quaterniond& q) {
    return Eigen::Quaterniond(q.w(), q.y(), q.x(), -q.z()).normalized();
}

inline Eigen::Vector3d omega_frd_to_flu(const Eigen::Vector3d& w) {
    return {w.x(), -w.y(), -w.z()};
}

} // namespace frame_conv

// ─────────────────────────────────────────────────────────────────────────────
// Platform state
// ─────────────────────────────────────────────────────────────────────────────
struct State {
    Eigen::VectorXd current = Eigen::VectorXd::Zero(13);
    bool received = false;

    State() {
        current(6) = 1.0;  // neutral quaternion w=1
    }

    bool hasFullState() const { return received; }

    void reset() {
        current = Eigen::VectorXd::Zero(13);
        current(6) = 1.0;
        received = false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Subscription/Publisher pair
// ─────────────────────────────────────────────────────────────────────────────
struct Handles {
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr setpoint_pub;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_pub;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr cmd_pub;
    rclcpp::TimerBase::SharedPtr heartbeat_timer;

    void reset() {
        odom_sub.reset();
        setpoint_pub.reset();
        offboard_pub.reset();
        cmd_pub.reset();
        heartbeat_timer.reset();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// setup: Create subscriptions and publishers for PX4 platform.
// ─────────────────────────────────────────────────────────────────────────────
inline void setup(
    rclcpp::Node* node,
    rclcpp::CallbackGroup::SharedPtr callback_group,
    State& state,
    std::mutex& state_mutex,
    Handles& handles)
{
    // Publishers
    handles.setpoint_pub = node->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "/fmu/in/trajectory_setpoint", 10);
    handles.offboard_pub = node->create_publisher<px4_msgs::msg::OffboardControlMode>(
        "/fmu/in/offboard_control_mode", 10);
    handles.cmd_pub = node->create_publisher<px4_msgs::msg::VehicleCommand>(
        "/fmu/in/vehicle_command", 10);

    rclcpp::SubscriptionOptions opts;
    opts.callback_group = callback_group;

    // Odometry callback - PX4 gives full state in one message
    handles.odom_sub = node->create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", 10,
        [&](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
            auto pos = frame_conv::ned_to_enu({msg->position[0], msg->position[1], msg->position[2]});
            auto vel = frame_conv::ned_to_enu({msg->velocity[0], msg->velocity[1], msg->velocity[2]});
            auto q = frame_conv::quat_ned_to_enu(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
            auto w = frame_conv::omega_frd_to_flu({msg->angular_velocity[0],
                                                    msg->angular_velocity[1],
                                                    msg->angular_velocity[2]});

            std::lock_guard<std::mutex> lk(state_mutex);
            state.current << pos.x(), pos.y(), pos.z(),
                            vel.x(), vel.y(), vel.z(),
                            q.w(), q.x(), q.y(), q.z(),
                            w.x(), w.y(), w.z();
            state.received = true;
        },
        opts);

    // Heartbeat timer - publish offboard control mode at 10Hz
    handles.heartbeat_timer = node->create_wall_timer(
        std::chrono::milliseconds(100),
        [&]() {
            px4_msgs::msg::OffboardControlMode msg;
            msg.timestamp = node->now().nanoseconds() / 1000;
            msg.position = msg.velocity = true;
            handles.offboard_pub->publish(msg);
        });

    RCLCPP_INFO(node->get_logger(),
        "[PX4] Subscribed: /fmu/out/vehicle_odometry | Publishing: /fmu/in/trajectory_setpoint");
}

// ─────────────────────────────────────────────────────────────────────────────
// publishCommand: Send TrajectorySetpoint to PX4.
// ─────────────────────────────────────────────────────────────────────────────
inline void publishCommand(
    rclcpp::Node* node,
    Handles& handles,
    const Eigen::VectorXd& state)
{
    if (!handles.setpoint_pub || state.size() < 13) return;

    auto pos = frame_conv::enu_to_ned({state(0), state(1), state(2)});
    auto vel = frame_conv::enu_to_ned({state(3), state(4), state(5)});
    auto q = frame_conv::quat_enu_to_ned(Eigen::Quaterniond(state(6), state(7), state(8), state(9)));

    double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                            1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));

    px4_msgs::msg::TrajectorySetpoint sp;
    sp.timestamp = node->now().nanoseconds() / 1000;
    sp.position[0] = pos.x();
    sp.position[1] = pos.y();
    sp.position[2] = pos.z();
    sp.velocity[0] = vel.x();
    sp.velocity[1] = vel.y();
    sp.velocity[2] = vel.z();
    sp.acceleration[0] = sp.acceleration[1] = sp.acceleration[2] =
        std::numeric_limits<float>::quiet_NaN();
    sp.jerk[0] = sp.jerk[1] = sp.jerk[2] =
        std::numeric_limits<float>::quiet_NaN();
    sp.yaw = static_cast<float>(yaw);
    sp.yawspeed = std::numeric_limits<float>::quiet_NaN();

    handles.setpoint_pub->publish(sp);
}

// ─────────────────────────────────────────────────────────────────────────────
// arm: Send arm command to PX4.
// ─────────────────────────────────────────────────────────────────────────────
inline void arm(rclcpp::Node* node, Handles& handles) {
    px4_msgs::msg::VehicleCommand msg;
    msg.timestamp = node->now().nanoseconds() / 1000;
    msg.command = 400;  // VEHICLE_CMD_COMPONENT_ARM_DISARM
    msg.param1 = 1.0;   // ARM
    msg.target_system = msg.source_system = 1;
    handles.cmd_pub->publish(msg);
}

} // namespace px4
} // namespace platform

#else // HAS_PX4_MSGS not defined - provide empty stubs

#include <Eigen/Dense>
#include <mutex>
#include <stdexcept>
#include <rclcpp/rclcpp.hpp>

namespace platform {
namespace px4 {

// Empty stubs - allow compilation without px4_msgs
// Runtime error if PX4 platform is requested but not compiled in

struct State {
    Eigen::VectorXd current = Eigen::VectorXd::Zero(13);
    bool hasFullState() const { return false; }
};

struct Handles {};

inline void setup(
    rclcpp::Node*,
    rclcpp::CallbackGroup::SharedPtr,
    State&,
    std::mutex&,
    Handles&)
{
    throw std::runtime_error("PX4 platform not compiled - install px4_msgs and rebuild");
}

inline void publishCommand(
    rclcpp::Node*,
    Handles&,
    const Eigen::VectorXd&)
{
    throw std::runtime_error("PX4 platform not compiled - install px4_msgs and rebuild");
}

inline void arm(rclcpp::Node*, Handles&) {
    throw std::runtime_error("PX4 platform not compiled - install px4_msgs and rebuild");
}

} // namespace px4
} // namespace platform

#endif // HAS_PX4_MSGS
