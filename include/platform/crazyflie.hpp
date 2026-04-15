/// @file crazyflie.hpp
/// @brief Crazyflie platform abstraction - direct subscribe/publish, no intermediate bridge.
///
/// LESSONS FROM CoManDO BRIDGE FAILURE:
/// ─────────────────────────────────────
/// 1. NO message_filters synchronization - adds latency and drops messages
/// 2. NO intermediate bridge node - direct subscribe/publish minimizes latency
/// 3. State updates via callback, not polling or separate messages
/// 4. Angular velocity from crazyswarm2 is in deg/s - convert to rad/s
/// 5. Acceleration feedforward MUST be computed and published for good tracking
///
/// This header creates subscriptions and publishers directly in the planner node,
/// avoiding the extra ROS2 hop that caused tracking issues in the CoManDO version.
///
/// EXPECTED ROS GRAPH (Crazyflie):
/// - Planner node: `/comando_planner`
/// - Input topics (from crazyswarm2): `/<drone_name>/pose`, `/<drone_name>/odom`
/// - Output topic (to crazyswarm2): `/<drone_name>/cmd_full_state`

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
#include <crazyflie_interfaces/msg/hover.hpp>
#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <mutex>
#include <functional>
#include <cmath>

namespace platform {
namespace crazyflie {

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
constexpr double DEG2RAD = M_PI / 180.0;
constexpr double DEFAULT_MASS = 0.027;  // kg

// ─────────────────────────────────────────────────────────────────────────────
// Compute world-frame acceleration from state + thrust for feedforward.
// a_world = R * [0; 0; fz/m] + [0; 0; -g]
// ─────────────────────────────────────────────────────────────────────────────
inline Eigen::Vector3d computeAcc(const Eigen::VectorXd& s, double fz,
                                   double mass = DEFAULT_MASS)
{
    Eigen::Quaterniond q(s(6), s(7), s(8), s(9));
    Eigen::Vector3d thrust_body(0.0, 0.0, fz / mass);
    Eigen::Vector3d acc = q.toRotationMatrix() * thrust_body;
    acc(2) -= 9.81;
    return acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform state - holds subscription state flags and state vector
// ─────────────────────────────────────────────────────────────────────────────
struct State {
    Eigen::VectorXd current = Eigen::VectorXd::Zero(13);
    bool pose_received = false;
    bool odom_received = false;

    State() {
        current(6) = 1.0;  // neutral quaternion w=1
    }

    bool hasFullState() const { return pose_received && odom_received; }

    void reset() {
        current = Eigen::VectorXd::Zero(13);
        current(6) = 1.0;
        pose_received = false;
        odom_received = false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Subscription/Publisher pair - owns the ROS2 handles
// ─────────────────────────────────────────────────────────────────────────────
struct Handles {
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cmd_pub;
    rclcpp::Publisher<crazyflie_interfaces::msg::Hover>::SharedPtr cmd_hover_pub;

    void reset() {
        pose_sub.reset();
        odom_sub.reset();
        cmd_pub.reset();
        cmd_hover_pub.reset();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// setup: Create subscriptions and publisher for Crazyflie platform.
//
// IMPORTANT: Subscriptions use the provided callback_group to ensure
// state updates don't block the solver or replay timers.
//
// on_state_update: Called after each state update (pose or odom).
//                  Planner can use this to trigger immediate action if needed,
//                  or just poll state.current via the state mutex.
//
// CRITICAL: No message_filters, no ApproximateTime sync - each callback
// updates its slice of state immediately on arrival.
// ─────────────────────────────────────────────────────────────────────────────
inline void setup(
    rclcpp::Node* node,
    rclcpp::CallbackGroup::SharedPtr callback_group,
    const std::string& drone_name,
    const std::string& odom_topic_override,
    State& state,
    std::mutex& state_mutex,
    Handles& handles)
{
    // Publisher: command output directly to hardware topic
    handles.cmd_pub = node->create_publisher<crazyflie_interfaces::msg::FullState>(
        "/" + drone_name + "/cmd_full_state", 10);
    handles.cmd_hover_pub = node->create_publisher<crazyflie_interfaces::msg::Hover>(
        "/" + drone_name + "/cmd_hover", 10);

    rclcpp::SubscriptionOptions opts;
    opts.callback_group = callback_group;

    if (!odom_topic_override.empty()) {
        // External full-state odometry source (e.g., already-relative odometry).
        handles.odom_sub = node->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_override, 10,
            [&](const nav_msgs::msg::Odometry::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(state_mutex);
                state.current(0) = msg->pose.pose.position.x;
                state.current(1) = msg->pose.pose.position.y;
                state.current(2) = msg->pose.pose.position.z;
                state.current(3) = msg->twist.twist.linear.x;
                state.current(4) = msg->twist.twist.linear.y;
                state.current(5) = msg->twist.twist.linear.z;
                state.current(6) = msg->pose.pose.orientation.w;
                state.current(7) = msg->pose.pose.orientation.x;
                state.current(8) = msg->pose.pose.orientation.y;
                state.current(9) = msg->pose.pose.orientation.z;
                // Odom override topics (e.g. /drone/body_relative_odom) are expected
                // to be in SI units already (rad/s). Do not convert again.
                state.current(10) = msg->twist.twist.angular.x;
                state.current(11) = msg->twist.twist.angular.y;
                state.current(12) = msg->twist.twist.angular.z;
                state.pose_received = true;
                state.odom_received = true;
            },
            opts);

        RCLCPP_INFO(node->get_logger(),
            "[Crazyflie] Subscribed state: %s | Publishing: /%s/cmd_full_state, /%s/cmd_hover",
            odom_topic_override.c_str(), drone_name.c_str(), drone_name.c_str());
    } else {
        // Pose callback - updates position and quaternion
        handles.pose_sub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name + "/pose", 10,
            [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(state_mutex);
                state.current(0) = msg->pose.position.x;
                state.current(1) = msg->pose.position.y;
                state.current(2) = msg->pose.position.z;
                state.current(6) = msg->pose.orientation.w;
                state.current(7) = msg->pose.orientation.x;
                state.current(8) = msg->pose.orientation.y;
                state.current(9) = msg->pose.orientation.z;
                state.pose_received = true;
            },
            opts);

        // Odometry callback - updates linear/angular velocity
        // NOTE: crazyswarm2 publishes angular velocity in deg/s, convert to rad/s
        handles.odom_sub = node->create_subscription<nav_msgs::msg::Odometry>(
            "/" + drone_name + "/odom", 10,
            [&](const nav_msgs::msg::Odometry::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(state_mutex);
                state.current(3) = msg->twist.twist.linear.x;
                state.current(4) = msg->twist.twist.linear.y;
                state.current(5) = msg->twist.twist.linear.z;
                state.current(10) = msg->twist.twist.angular.x * DEG2RAD;
                state.current(11) = msg->twist.twist.angular.y * DEG2RAD;
                state.current(12) = msg->twist.twist.angular.z * DEG2RAD;
                state.odom_received = true;
            },
            opts);

        RCLCPP_INFO(node->get_logger(),
            "[Crazyflie] Subscribed: /%s/pose, /%s/odom | Publishing: /%s/cmd_full_state, /%s/cmd_hover",
            drone_name.c_str(), drone_name.c_str(), drone_name.c_str(), drone_name.c_str());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// publishCommand: Send FullState command to Crazyflie.
//
// IMPORTANT: Computes and includes acceleration feedforward from thrust.
// The lower controller NEEDS this for good tracking - setting it to zero
// was a bug in the original CoManDO bridge that caused tracking failure.
// ─────────────────────────────────────────────────────────────────────────────
inline void publishCommand(
    rclcpp::Node* node,
    Handles& handles,
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    double mass = DEFAULT_MASS)
{
    if (!handles.cmd_pub || state.size() < 13) return;

    const double fz = (control.size() >= 1) ? control(0) : 0.0;
    Eigen::Vector3d acc = computeAcc(state, fz, mass);

    crazyflie_interfaces::msg::FullState msg;
    msg.header.stamp = node->now();
    msg.header.frame_id = "world";
    msg.pose.position.x = state(0);
    msg.pose.position.y = state(1);
    msg.pose.position.z = state(2);
    msg.twist.linear.x = state(3);
    msg.twist.linear.y = state(4);
    msg.twist.linear.z = state(5);
    msg.pose.orientation.w = state(6);
    msg.pose.orientation.x = state(7);
    msg.pose.orientation.y = state(8);
    msg.pose.orientation.z = state(9);
    msg.twist.angular.x = state(10);
    msg.twist.angular.y = state(11);
    msg.twist.angular.z = state(12);
    msg.acc.x = static_cast<float>(acc.x());
    msg.acc.y = static_cast<float>(acc.y());
    msg.acc.z = static_cast<float>(acc.z());

    handles.cmd_pub->publish(msg);
}

inline void publishHoverCommandDirect(
    Handles& handles,
    const std::array<float, 4>& cmd)
{
    if (!handles.cmd_hover_pub) {
        return;
    }

    crazyflie_interfaces::msg::Hover msg;
    msg.vx = cmd[0];
    msg.vy = cmd[1];
    msg.z_distance = cmd[2];
    msg.yaw_rate = cmd[3];

    handles.cmd_hover_pub->publish(msg);
}

inline void publishHoverCommand(
    Handles& handles,
    const Eigen::VectorXd& state)
{
    if (!handles.cmd_hover_pub || state.size() < 13) {
        return;
    }
    publishHoverCommandDirect(handles, {
        static_cast<float>(state(3)),
        static_cast<float>(state(4)),
        static_cast<float>(state(2)),
        static_cast<float>(state(12)),
    });
}

} // namespace crazyflie
} // namespace platform
