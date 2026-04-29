/// @file mavros.hpp
/// @brief MAVROS platform adapter for PX4 via MAVROS2 bridge.
///
/// State input:  /mavros/local_position/odom (nav_msgs/Odometry) → 13D ENU
/// CmdBodyRate:  /mavros/setpoint_raw/attitude (mavros_msgs/AttitudeTarget)
///               type_mask=IGNORE_ATTITUDE(128); body_rate; thrust=T*hover_thrust/9.81
/// CmdFullState: /mavros/setpoint_raw/local (mavros_msgs/PositionTarget)
///               position + velocity + yaw, ENU frame
///
/// Pre-arm sequence:
///   Idle → WarmingUp(2s neutral setpoints) → ReadyToArm → Armed
///   MAVROS requires continuous setpoints BEFORE OFFBOARD mode switch.
///
/// Compile guard: entire implementation wrapped in #ifdef HAS_MAVROS_MSGS.
/// Stubs outside the guard throw std::runtime_error if called.
#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <functional>
#include <cmath>
#include <chrono>
#include <algorithm>

#ifdef HAS_MAVROS_MSGS
#include <mavros_msgs/msg/attitude_target.hpp>
#include <mavros_msgs/msg/position_target.hpp>
#include <mavros_msgs/srv/set_mode.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#endif

namespace platform {
namespace mavros {

// ─────────────────────────────────────────────────────────────────────────────
// Pre-arm state machine
// ─────────────────────────────────────────────────────────────────────────────
enum class ArmState {
    Idle,         // not yet started
    WarmingUp,    // publishing neutral setpoints; waiting 2 s
    ReadyToArm,   // warmup done; sending OFFBOARD + ARM
    Armed         // flying
};

// ─────────────────────────────────────────────────────────────────────────────
// Handles — owns all ROS2 publishers, subscribers, timers for MAVROS
// ─────────────────────────────────────────────────────────────────────────────
struct Handles {
    // State
    Eigen::VectorXd current = Eigen::VectorXd::Zero(13);
    bool odom_received = false;

    // Pre-arm
    ArmState arm_state = ArmState::Idle;
    std::chrono::steady_clock::time_point warmup_start;
    float hover_thrust = 0.3f;  // normalised [0,1] at which vehicle hovers

    // Latest setpoint to republish (heartbeat keepalive)
    Eigen::VectorXd last_state_cmd = Eigen::VectorXd::Zero(13);
    Eigen::VectorXd last_rate_cmd  = Eigen::VectorXd::Zero(5);
    bool last_cmd_is_bodyrate = false;

#ifdef HAS_MAVROS_MSGS
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Publisher<mavros_msgs::msg::AttitudeTarget>::SharedPtr att_pub;
    rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr pos_pub;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client;
    rclcpp::TimerBase::SharedPtr heartbeat_timer;
    rclcpp::TimerBase::SharedPtr warmup_timer;
#endif
};

#ifdef HAS_MAVROS_MSGS

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper — build a neutral AttitudeTarget (level, zero thrust)
// ─────────────────────────────────────────────────────────────────────────────
inline mavros_msgs::msg::AttitudeTarget neutralAttitudeTarget(rclcpp::Node* node) {
    mavros_msgs::msg::AttitudeTarget msg;
    msg.header.stamp = node->now();
    // IGNORE_ATTITUDE: only body_rate and thrust are used
    msg.type_mask = mavros_msgs::msg::AttitudeTarget::IGNORE_ATTITUDE;
    msg.body_rate.x = 0.0f;
    msg.body_rate.y = 0.0f;
    msg.body_rate.z = 0.0f;
    msg.thrust = 0.0f;
    return msg;
}

// ─────────────────────────────────────────────────────────────────────────────
// sendOffboardAndArm — called once warmup completes
// ─────────────────────────────────────────────────────────────────────────────
inline void sendOffboardAndArm(rclcpp::Node* node, Handles& h) {
    // Switch to OFFBOARD
    auto mode_req = std::make_shared<mavros_msgs::srv::SetMode::Request>();
    mode_req->custom_mode = "OFFBOARD";
    h.set_mode_client->async_send_request(mode_req,
        [node](rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture /*f*/) {
            RCLCPP_INFO(node->get_logger(), "[MAVROS] OFFBOARD mode requested");
        });

    // ARM
    auto arm_req = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
    arm_req->value = true;
    h.arming_client->async_send_request(arm_req,
        [node](rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedFuture /*f*/) {
            RCLCPP_INFO(node->get_logger(), "[MAVROS] ARM requested");
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// setup — create all MAVROS subscribers, publishers, service clients, timers
// ─────────────────────────────────────────────────────────────────────────────
inline void setup(rclcpp::Node* node,
                  rclcpp::CallbackGroup::SharedPtr sensor_cb_group,
                  float hover_thrust_param,
                  Eigen::VectorXd& state_out,
                  std::mutex& state_mutex,
                  Handles& h) {
    h.hover_thrust = hover_thrust_param;

    auto sub_opt = rclcpp::SubscriptionOptions();
    sub_opt.callback_group = sensor_cb_group;

    // State subscription
    h.odom_sub = node->create_subscription<nav_msgs::msg::Odometry>(
        "/mavros/local_position/odom", rclcpp::SensorDataQoS(),
        [&state_out, &state_mutex, &h](const nav_msgs::msg::Odometry::SharedPtr msg) {
            std::lock_guard<std::mutex> lk(state_mutex);
            const auto& p = msg->pose.pose.position;
            const auto& q = msg->pose.pose.orientation;
            const auto& v = msg->twist.twist.linear;
            const auto& w = msg->twist.twist.angular;
            // 13D ENU state: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
            state_out(0) = p.x;  state_out(1) = p.y;  state_out(2) = p.z;
            state_out(3) = v.x;  state_out(4) = v.y;  state_out(5) = v.z;
            state_out(6) = q.w;  state_out(7) = q.x;
            state_out(8) = q.y;  state_out(9) = q.z;
            state_out(10) = w.x; state_out(11) = w.y; state_out(12) = w.z;
            h.current = state_out;
            h.odom_received = true;
        }, sub_opt);

    // Publishers
    h.att_pub = node->create_publisher<mavros_msgs::msg::AttitudeTarget>(
        "/mavros/setpoint_raw/attitude", 10);
    h.pos_pub = node->create_publisher<mavros_msgs::msg::PositionTarget>(
        "/mavros/setpoint_raw/local", 10);

    // Service clients
    h.set_mode_client = node->create_client<mavros_msgs::srv::SetMode>(
        "/mavros/set_mode");
    h.arming_client   = node->create_client<mavros_msgs::srv::CommandBool>(
        "/mavros/cmd/arming");

    // Pre-arm: start warmup immediately
    h.arm_state    = ArmState::WarmingUp;
    h.warmup_start = std::chrono::steady_clock::now();
    RCLCPP_INFO(node->get_logger(),
        "[MAVROS] Warmup started (2 s of neutral setpoints before OFFBOARD+ARM)");

    // Heartbeat timer at 10 Hz — republishes last setpoint (MAVROS keepalive)
    h.heartbeat_timer = node->create_wall_timer(
        std::chrono::milliseconds(100),
        [node, &h]() {
            using namespace std::chrono;
            // Advance arm state machine
            if (h.arm_state == ArmState::WarmingUp) {
                const double elapsed =
                    duration<double>(steady_clock::now() - h.warmup_start).count();
                if (elapsed >= 2.0) {
                    h.arm_state = ArmState::ReadyToArm;
                    RCLCPP_INFO(node->get_logger(), "[MAVROS] Warmup done, requesting OFFBOARD+ARM");
                    sendOffboardAndArm(node, h);
                } else {
                    // Publish neutral setpoint to satisfy MAVROS pre-arm requirement
                    h.att_pub->publish(neutralAttitudeTarget(node));
                    return;
                }
            }

            // Republish last command as keepalive
            if (h.last_cmd_is_bodyrate && h.arm_state == ArmState::Armed) {
                mavros_msgs::msg::AttitudeTarget msg;
                msg.header.stamp = node->now();
                msg.type_mask = mavros_msgs::msg::AttitudeTarget::IGNORE_ATTITUDE;
                msg.body_rate.x = h.last_rate_cmd.size() > 1 ? float(h.last_rate_cmd(1)) : 0.0f;
                msg.body_rate.y = h.last_rate_cmd.size() > 2 ? float(h.last_rate_cmd(2)) : 0.0f;
                msg.body_rate.z = h.last_rate_cmd.size() > 3 ? float(h.last_rate_cmd(3)) : 0.0f;
                msg.thrust      = h.last_rate_cmd.size() > 0
                    ? float(std::clamp(double(h.last_rate_cmd(0)) * h.hover_thrust / 9.81, 0.0, 1.0))
                    : 0.0f;
                h.att_pub->publish(msg);
            } else if (!h.last_cmd_is_bodyrate && h.arm_state == ArmState::Armed) {
                // Full state keepalive republished from pos_pub — no-op here,
                // PositionTarget is latched by MAVROS setpoint buffer
            }
        });

    RCLCPP_INFO(node->get_logger(),
        "[MAVROS] Platform setup done. hover_thrust=%.3f", hover_thrust_param);
}

// ─────────────────────────────────────────────────────────────────────────────
// publishBodyRateCommand — CmdBodyRate path
// u = [T_ms2, ωx, ωy, ωz, Theta(unused)]
// thrust_norm = T_ms2 * hover_thrust / 9.81
// ─────────────────────────────────────────────────────────────────────────────
inline void publishBodyRateCommand(Handles& h, const Eigen::VectorXd& u) {
    h.last_rate_cmd    = u;
    h.last_cmd_is_bodyrate = true;
    if (h.arm_state != ArmState::Armed) {
        return;  // don't send real commands until armed
    }
    if (!h.att_pub) return;

    mavros_msgs::msg::AttitudeTarget msg;
    // IGNORE_ATTITUDE flag — only body_rate + thrust are used
    msg.type_mask = mavros_msgs::msg::AttitudeTarget::IGNORE_ATTITUDE;
    msg.body_rate.x = u.size() > 1 ? float(u(1)) : 0.0f;
    msg.body_rate.y = u.size() > 2 ? float(u(2)) : 0.0f;
    msg.body_rate.z = u.size() > 3 ? float(u(3)) : 0.0f;
    msg.thrust = u.size() > 0
        ? float(std::clamp(u(0) * h.hover_thrust / 9.81, 0.0, 1.0))
        : 0.0f;
    h.att_pub->publish(msg);
}

// ─────────────────────────────────────────────────────────────────────────────
// publishFullStateCommand — CmdFullState path
// s = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
// Sends position + velocity + yaw via PositionTarget (ENU)
// ─────────────────────────────────────────────────────────────────────────────
inline void publishFullStateCommand(Handles& h, const Eigen::VectorXd& s) {
    h.last_state_cmd   = s;
    h.last_cmd_is_bodyrate = false;
    if (h.arm_state != ArmState::Armed) return;
    if (!h.pos_pub) return;

    mavros_msgs::msg::PositionTarget msg;
    // position + velocity + yaw; ignore accel, yaw_rate, position-offset
    msg.coordinate_frame = mavros_msgs::msg::PositionTarget::FRAME_LOCAL_NED;
    // type_mask: ignore accel (0b0000111111000111 = 0x0FC7... let's use constants)
    // We want to use: position (bit 0-2 NOT set), velocity (bit 3-5 NOT set),
    // yaw (bit 10 NOT set); ignore force, accel, yaw_rate
    msg.type_mask = mavros_msgs::msg::PositionTarget::IGNORE_AFX |
                    mavros_msgs::msg::PositionTarget::IGNORE_AFY |
                    mavros_msgs::msg::PositionTarget::IGNORE_AFZ |
                    mavros_msgs::msg::PositionTarget::IGNORE_YAW_RATE;

    if (s.size() >= 3) {
        msg.position.x = float(s(0));
        msg.position.y = float(s(1));
        msg.position.z = float(s(2));
    }
    if (s.size() >= 6) {
        msg.velocity.x = float(s(3));
        msg.velocity.y = float(s(4));
        msg.velocity.z = float(s(5));
    }
    // Extract yaw from quaternion [qw, qx, qy, qz] at s[6:10]
    if (s.size() >= 10) {
        const double qw = s(6), qx = s(7), qy = s(8), qz = s(9);
        msg.yaw = float(std::atan2(2.0*(qw*qz + qx*qy),
                                   1.0 - 2.0*(qy*qy + qz*qz)));
    }
    h.pos_pub->publish(msg);
}

// ─────────────────────────────────────────────────────────────────────────────
// markArmed — call from planner after is_flying_ gate
// ─────────────────────────────────────────────────────────────────────────────
inline void markArmed(Handles& h) {
    if (h.arm_state == ArmState::ReadyToArm) {
        h.arm_state = ArmState::Armed;
    }
}

#else  // !HAS_MAVROS_MSGS — compile stubs

inline void setup(rclcpp::Node*, rclcpp::CallbackGroup::SharedPtr, float,
                  Eigen::VectorXd&, std::mutex&, Handles&) {
    throw std::runtime_error(
        "MAVROS platform requested but mavros_msgs not found at build time. "
        "Install ros-humble-mavros and rebuild.");
}

inline void publishBodyRateCommand(Handles&, const Eigen::VectorXd&) {
    throw std::runtime_error("MAVROS not available (built without HAS_MAVROS_MSGS)");
}

inline void publishFullStateCommand(Handles&, const Eigen::VectorXd&) {
    throw std::runtime_error("MAVROS not available (built without HAS_MAVROS_MSGS)");
}

inline void markArmed(Handles&) {}

#endif  // HAS_MAVROS_MSGS

}  // namespace mavros
}  // namespace platform
