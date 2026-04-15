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
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
#include <crazyflie_interfaces/msg/hover.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <mutex>
#include <functional>
#include <cmath>
#include <cstdint>

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
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_legacy_pub;
    rclcpp::Publisher<crazyflie_interfaces::msg::Hover>::SharedPtr cmd_hover_pub;
    bool legacy_thrust_unlocked = false; // firmware thrustLocked starts true; must send thrust=0 first

    void reset() {
        pose_sub.reset();
        odom_sub.reset();
        cmd_pub.reset();
        cmd_vel_legacy_pub.reset();
        cmd_hover_pub.reset();
        legacy_thrust_unlocked = false;
    }
};

// Debug info returned from publishLegacyCommand, used for logging.
struct LegacyCommandDebug {
    double fz_cmd_newton = 0.0;
    uint16_t thrust_u16 = 0;
    bool unlock_packet_only = false;
    bool real_command_published = false;
};

inline Eigen::Vector3d quaternionToRollPitchYaw(const Eigen::Vector4d& q_wxyz)
{
    const double qw = q_wxyz(0);
    const double qx = q_wxyz(1);
    const double qy = q_wxyz(2);
    const double qz = q_wxyz(3);

    const double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    const double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    const double roll = std::atan2(sinr_cosp, cosr_cosp);

    const double sinp = 2.0 * (qw * qy - qz * qx);
    const double pitch = std::asin(std::clamp(sinp, -1.0, 1.0));

    const double siny_cosp = 2.0 * (qw * qz + qx * qy);
    const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double yaw = std::atan2(siny_cosp, cosy_cosp);

    return Eigen::Vector3d(roll, pitch, yaw);
}

// Convert body rates (p,q,r) to inertial yaw rate for ZYX euler convention.
inline double bodyRatesToYawRate(const Eigen::Vector3d& omega_body,
                                 double roll,
                                 double pitch)
{
    const double p = omega_body(0);
    const double q = omega_body(1);
    const double r = omega_body(2);
    (void)p;

    const double cos_pitch = std::cos(pitch);
    const double denom = (std::abs(cos_pitch) < 1e-3)
        ? ((cos_pitch >= 0.0) ? 1e-3 : -1e-3)
        : cos_pitch;
    return (q * std::sin(roll) + r * std::cos(roll)) / denom;
}

// Convert body-frame thrust (Newton) to a uint16 PWM value for cmd_vel_legacy.
//
// Uses the exact inverse of the CrazySim SITL Gazebo motor chain.
//
// The Gazebo forward chain per motor (from CrtpUtils.h + model.sdf):
//   thrust_desired = (pwm / 65535) * 0.18          [N]
//   omega          = sqrt(thrust_desired / 2.3375e-8) [rad/s]
//   force          = 1.28192e-8 * omega^2             [N]
//
// Simplifying the chain:
//   force_per_motor = (1.28192e-8 / 2.3375e-8) * (pwm / 65535) * 0.18
//                   = kGzK * pwm
// where kGzK = motorConstant * pwmScale / (pwmCoeff * 65535)
//
// Inverting:
//   pwm = fz_total / (4 * kGzK)
//
// Verified hover point: fz_total = 0.0282*9.81 ≈ 0.2766 N → u16 ≈ 45914

// CrazySim SITL Gazebo motor constants (from CrtpUtils.h and model.sdf)
static constexpr double kGzMotorConstant = 1.28192e-8;  // [N / (rad/s)²]
static constexpr double kGzPwm2OmegaCoeff = 2.3375e-8;  // [N] (divisor in PWM2OMEGA)
static constexpr double kGzPwm2OmegaScale = 0.18;       // [N] at pwm=65535 per motor
static constexpr double kGzPwm2OmegaDead  = 7000.0;     // deadzone below which force=0
// Composite linear gain: force_per_motor = kGzK * pwm
static constexpr double kGzK = kGzMotorConstant * kGzPwm2OmegaScale
                                / (kGzPwm2OmegaCoeff * 65535.0);

inline uint16_t mapThrustNToLegacyU16(double fz_newton)
{
    if (fz_newton <= 0.0) return 0;
    // Invert: pwm = fz_total / (4 * kGzK)
    const double pwm = fz_newton / (4.0 * kGzK);
    const double clamped = std::clamp(pwm, 0.0, 60000.0);
    return static_cast<uint16_t>(std::lround(clamped));
}

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
    handles.cmd_vel_legacy_pub = node->create_publisher<geometry_msgs::msg::Twist>(
        "/" + drone_name + "/cmd_vel_legacy", 10);
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
                // Keep legacy conversion for compatibility with existing pipelines.
                state.current(10) = msg->twist.twist.angular.x * DEG2RAD;
                state.current(11) = msg->twist.twist.angular.y * DEG2RAD;
                state.current(12) = msg->twist.twist.angular.z * DEG2RAD;
                state.pose_received = true;
                state.odom_received = true;
            },
            opts);

        RCLCPP_INFO(node->get_logger(),
            "[Crazyflie] Subscribed state: %s | Publishing: /%s/cmd_full_state, /%s/cmd_vel_legacy, /%s/cmd_hover",
            odom_topic_override.c_str(), drone_name.c_str(), drone_name.c_str(), drone_name.c_str());
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
            "[Crazyflie] Subscribed: /%s/pose, /%s/odom | Publishing: /%s/cmd_full_state, /%s/cmd_vel_legacy, /%s/cmd_hover",
            drone_name.c_str(), drone_name.c_str(), drone_name.c_str(), drone_name.c_str(), drone_name.c_str());
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

// Publish legacy roll/pitch/yawrate/thrust command on /cmd_vel_legacy.
// Mapping matches crazyflie_server.cpp cmd_vel_legacy_changed:
// linear.x=-pitch_deg, linear.y=roll_deg, linear.z=thrust_u16, angular.z=yawrate_deg_s.
inline LegacyCommandDebug publishLegacyCommand(
    Handles& handles,
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control)
{
    LegacyCommandDebug dbg;
    dbg.fz_cmd_newton = (control.size() >= 1) ? control(0) : 0.0;
    dbg.thrust_u16 = mapThrustNToLegacyU16(dbg.fz_cmd_newton);

    if (!handles.cmd_vel_legacy_pub || state.size() < 13) {
        return dbg;
    }

    // Crazyflie firmware's RPYT commander starts with thrustLocked=true.
    // It only unlocks when a thrust=0 packet is received. Send that once
    // before the first real command so subsequent thrust values take effect.
    if (!handles.legacy_thrust_unlocked) {
        geometry_msgs::msg::Twist unlock_msg;
        unlock_msg.linear.z = 0.0;
        handles.cmd_vel_legacy_pub->publish(unlock_msg);
        handles.legacy_thrust_unlocked = true;
        dbg.unlock_packet_only = true;
        return dbg; // skip this tick; real command will follow next tick
    }

    const Eigen::Vector4d q_cmd = state.segment<4>(6);
    const Eigen::Vector3d omega_cmd = state.segment<3>(10);
    const Eigen::Vector3d rpy = quaternionToRollPitchYaw(q_cmd);

    const double roll_deg = rpy(0) * 180.0 / M_PI;
    const double pitch_deg = rpy(1) * 180.0 / M_PI;
    const double yawrate_deg_s = bodyRatesToYawRate(omega_cmd, rpy(0), rpy(1)) * 180.0 / M_PI;

    geometry_msgs::msg::Twist msg;
    msg.linear.x = -pitch_deg;
    msg.linear.y = roll_deg;
    msg.linear.z = static_cast<double>(dbg.thrust_u16);
    msg.angular.z = yawrate_deg_s;
    msg.angular.x = 0.0;
    msg.angular.y = 0.0;

    handles.cmd_vel_legacy_pub->publish(msg);
    dbg.real_command_published = true;
    return dbg;
}

inline void publishHoverCommand(
    Handles& handles,
    const Eigen::VectorXd& state)
{
    if (!handles.cmd_hover_pub || state.size() < 13) {
        return;
    }

    crazyflie_interfaces::msg::Hover msg;
    msg.vx = static_cast<float>(state(3));
    msg.vy = static_cast<float>(state(4));
    msg.z_distance = static_cast<float>(state(2));
    msg.yaw_rate = static_cast<float>(state(12));

    handles.cmd_hover_pub->publish(msg);
}

} // namespace crazyflie
} // namespace platform
