/// @file target_tracker.hpp
/// @brief Target-state subscriptions (odom + accel) for stateswitch OCP.
///
/// EXPECTED ROS GRAPH (Target Tracker):
/// - Planner node: `/comando_planner`
/// - Input topics (from target estimator/sim): `<target_odom_topic>`,
///   `<target_accel_topic>`
/// - Typical defaults in launch: `/target/odom`, `/target/accel`

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>

#include <Eigen/Dense>
#include <mutex>
#include <string>

namespace platform {
namespace target_tracker {

struct TargetState {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();

    rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
    bool odom_received = false;
    bool accel_received = false;
    bool valid = false;

    bool isFresh(const rclcpp::Time& now, double max_age_sec = 0.2) const {
        if (!valid) return false;
        if (timestamp.nanoseconds() <= 0) return false;
        const double age = (now - timestamp).seconds();
        return age >= 0.0 && age < max_age_sec;
    }
};

struct Handles {
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<geometry_msgs::msg::AccelStamped>::SharedPtr accel_sub;

    void reset() {
        odom_sub.reset();
        accel_sub.reset();
    }
};

inline void setup(
    rclcpp::Node* node,
    rclcpp::CallbackGroup::SharedPtr callback_group,
    const std::string& odom_topic,
    const std::string& accel_topic,
    TargetState& target_state,
    std::mutex& target_mutex,
    Handles& handles)
{
    rclcpp::SubscriptionOptions opts;
    opts.callback_group = callback_group;

    handles.odom_sub = node->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, 10,
        [&](const nav_msgs::msg::Odometry::SharedPtr msg) {
            std::lock_guard<std::mutex> lk(target_mutex);
            target_state.position << msg->pose.pose.position.x,
                                     msg->pose.pose.position.y,
                                     msg->pose.pose.position.z;
            target_state.velocity << msg->twist.twist.linear.x,
                                     msg->twist.twist.linear.y,
                                     msg->twist.twist.linear.z;
            target_state.odom_received = true;
            if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
                target_state.timestamp = rclcpp::Time(msg->header.stamp);
            } else {
                target_state.timestamp = node->now();
            }
            target_state.valid = target_state.odom_received && target_state.accel_received;
        },
        opts);

    handles.accel_sub = node->create_subscription<geometry_msgs::msg::AccelStamped>(
        accel_topic, 10,
        [&](const geometry_msgs::msg::AccelStamped::SharedPtr msg) {
            std::lock_guard<std::mutex> lk(target_mutex);
            target_state.acceleration << msg->accel.linear.x,
                                         msg->accel.linear.y,
                                         msg->accel.linear.z;
            target_state.accel_received = true;
            if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
                target_state.timestamp = rclcpp::Time(msg->header.stamp);
            } else {
                target_state.timestamp = node->now();
            }
            target_state.valid = target_state.odom_received && target_state.accel_received;
        },
        opts);

    RCLCPP_INFO(node->get_logger(),
        "[TargetTracker] Subscribed: %s, %s",
        odom_topic.c_str(), accel_topic.c_str());
}

} // namespace target_tracker
} // namespace platform
