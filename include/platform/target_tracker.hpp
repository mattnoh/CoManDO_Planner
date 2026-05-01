/// @file target_tracker.hpp
/// @brief Target-state subscriptions (odom + optional live accel + predicted accel).
///
/// EXPECTED ROS GRAPH (Target Tracker):
/// - Planner node: `/comando_planner`
/// - Input topics (from target estimator/sim): `<target_odom_topic>`,
///   `<target_accel_topic>` (optional diagnostics), `<predicted_accel_topic>`
/// - Typical defaults in launch: `/target/odom`, `/target/accel`,
///   `/target/predicted_accel`
/// - `/target/predicted_accel` is only required by OCPs that explicitly need an
///   external target-acceleration profile, such as `tracking_circle_target`.
///   `stateswitch` builds its predictor from the target snapshot instead.
///
/// FIXES vs original:
///   BUG 1 — Single shared timestamp for two async topics.
///            Original: both callbacks overwrote the same `timestamp`.
///            If odom arrived but accel dropped out, isFresh() used the odom
///            timestamp and returned true even though acceleration was stale.
///            Fix: separate `odom_timestamp` and `accel_timestamp` per topic;
///                 isFresh() requires BOTH to be individually fresh.
///
///   BUG 2 — `valid` was permanently true once both topics fired once.
///            If a topic dropped mid-flight the solver continued using stale
///            values with no way to detect the dropout at the `valid` flag.
///            Fix: `valid` is now computed inside isFresh() dynamically rather
///                 than cached as a permanent bool. The `odom_received` and
///                 `accel_received` bools are kept for startup sequencing only.
///
///   NOTE — Lambda reference captures: `target_state`, `target_mutex`, and
///           `node` are captured by reference in long-lived subscription
///           callbacks. The caller (planner_node) must ensure these outlive
///           the subscriptions. Call `handles.reset()` before destroying the
///           referents (e.g. in the node destructor).

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include <Eigen/Dense>
#include <mutex>
#include <string>

#include "target/target_accel_buffer.hpp"

namespace platform {
namespace target_tracker {

struct TargetState {
    Eigen::Vector3d position     = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity     = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();  // Ω_N in world frame
    Eigen::Vector3d angular_acceleration = Eigen::Vector3d::Zero(); // β_N in world frame
    Eigen::Vector4d orientation  = Eigen::Vector4d(1,0,0,0);     // target quat [qw,qx,qy,qz]

    // Separate timestamps per topic so each can be checked independently.
    rclcpp::Time odom_timestamp {0, 0, RCL_ROS_TIME};
    rclcpp::Time accel_timestamp{0, 0, RCL_ROS_TIME};

    // Startup flags: true after the first message of each type ever arrives.
    // Used only to gate early-startup "waiting for first message" warnings.
    // Do NOT use these to assess current freshness — call isFresh() instead.
    bool odom_received  = false;
    bool accel_received = false;

    /// Odom-only freshness check, used by OCPs that only need target pose/velocity.
    bool isOdomFresh(const rclcpp::Time& now, double max_age_sec = 0.2) const {
        if (!odom_received) return false;
        if (odom_timestamp.nanoseconds() <= 0) return false;
        const double odom_age = (now - odom_timestamp).seconds();
        const double kClockTol = 0.001;
        return !(odom_age < -kClockTol || odom_age >= max_age_sec);
    }

    /// Live acceleration freshness check (diagnostic topic).
    bool isAccelFresh(const rclcpp::Time& now, double max_age_sec = 0.2) const {
        if (!accel_received) return false;
        if (accel_timestamp.nanoseconds() <= 0) return false;
        const double accel_age = (now - accel_timestamp).seconds();
        const double kClockTol = 0.001;
        return !(accel_age < -kClockTol || accel_age >= max_age_sec);
    }

    /// Returns true only if BOTH topics have delivered at least one message
    /// AND both of their most recent messages are younger than max_age_sec.
    ///
    /// max_age_sec default = 0.2 s (5 Hz minimum acceptable rate).
    /// Raise to 0.5 s for slow targets; lower to 0.1 s for tight requirements.
    bool isFresh(const rclcpp::Time& now, double max_age_sec = 0.2) const {
        return isOdomFresh(now, max_age_sec) && isAccelFresh(now, max_age_sec);
    }

    /// Convenience: is the state currently valid (fresh with default tolerance)?
    bool valid(const rclcpp::Time& now) const { return isFresh(now); }

    /// Age of the older of the two topics — useful for diagnostics / logging.
    double worstAgeSeconds(const rclcpp::Time& now) const {
        if (!odom_received || !accel_received) return 1e9;
        const double oa = (now - odom_timestamp).seconds();
        const double aa = (now - accel_timestamp).seconds();
        return std::max(oa, aa);
    }

    // Predicted acceleration cache used by acceleration-driven OCPs.
    struct PredictedAccel {
        rclcpp::Time origin_time{0, 0, RCL_ROS_TIME};
        target_models::TargetAccelBuffer accel_buffer;
        bool received = false;
        
        bool isFresh(const rclcpp::Time& now, double max_age_sec = 2.0) const {
            if (!received) return false;
            if (origin_time.nanoseconds() <= 0) return false;
            return (now - origin_time).seconds() < max_age_sec;
        }
    };
    PredictedAccel predicted_accel;
};

struct Handles {
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       odom_sub;
    rclcpp::Subscription<geometry_msgs::msg::AccelStamped>::SharedPtr accel_sub;
    rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr predicted_accel_sub;

    void reset() {
        odom_sub.reset();
        accel_sub.reset();
        predicted_accel_sub.reset();
    }
};

inline void setup(
    rclcpp::Node*                        node,
    rclcpp::CallbackGroup::SharedPtr     callback_group,
    const std::string&                   odom_topic,
    const std::string&                   accel_topic,
    const std::string&                   predicted_accel_topic,
    TargetState&                         target_state,
    std::mutex&                          target_mutex,
    Handles&                             handles)
{
    rclcpp::SubscriptionOptions opts;
    opts.callback_group = callback_group;

    // ── Odometry callback ────────────────────────────────────────────────────
    handles.odom_sub = node->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, 10,
        [&target_state, &target_mutex, node](
            const nav_msgs::msg::Odometry::SharedPtr msg)
        {
            std::lock_guard<std::mutex> lk(target_mutex);

            target_state.position << msg->pose.pose.position.x,
                                     msg->pose.pose.position.y,
                                     msg->pose.pose.position.z;
            target_state.velocity << msg->twist.twist.linear.x,
                                     msg->twist.twist.linear.y,
                                     msg->twist.twist.linear.z;
            Eigen::Vector3d prev_omega = target_state.angular_velocity;
            target_state.angular_velocity << msg->twist.twist.angular.x,
                                             msg->twist.twist.angular.y,
                                             msg->twist.twist.angular.z;
            target_state.orientation << msg->pose.pose.orientation.w,
                                        msg->pose.pose.orientation.x,
                                        msg->pose.pose.orientation.y,
                                        msg->pose.pose.orientation.z;

            rclcpp::Time new_timestamp =
                (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0)
                    ? rclcpp::Time(msg->header.stamp)
                    : node->now();

            if (target_state.odom_received) {
                double dt = (new_timestamp - target_state.odom_timestamp).seconds();
                if (dt > 1e-6 && dt < 0.5) {
                    target_state.angular_acceleration =
                        (target_state.angular_velocity - prev_omega) / dt;
                }
            }

            target_state.odom_received = true;

            // Update ONLY the odom timestamp; accel timestamp is unaffected.
            target_state.odom_timestamp = new_timestamp;
        },
        opts);

    // ── Acceleration callback ────────────────────────────────────────────────
    if (!accel_topic.empty()) {
        handles.accel_sub = node->create_subscription<geometry_msgs::msg::AccelStamped>(
            accel_topic, 10,
            [&target_state, &target_mutex, node](
                const geometry_msgs::msg::AccelStamped::SharedPtr msg)
            {
                std::lock_guard<std::mutex> lk(target_mutex);

                target_state.acceleration << msg->accel.linear.x,
                                             msg->accel.linear.y,
                                             msg->accel.linear.z;
                
                // Read optional angular acceleration from the AccelStamped message.
                // This overrides the finite-difference estimate from odom if both exist.
                target_state.angular_acceleration << msg->accel.angular.x,
                                                     msg->accel.angular.y,
                                                     msg->accel.angular.z;
                
                target_state.accel_received = true;

                // Update ONLY the accel timestamp; odom timestamp is unaffected.
                target_state.accel_timestamp =
                    (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0)
                        ? rclcpp::Time(msg->header.stamp)
                        : node->now();
            },
            opts);
    }

    // ── Predicted acceleration callback ──────────────────────────────────────
    handles.predicted_accel_sub = node->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
        predicted_accel_topic, 10,
        [&target_state, &target_mutex, node](
            const trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg)
        {
            if (msg->points.empty()) return;

            std::lock_guard<std::mutex> lk(target_mutex);
            
            auto& pred = target_state.predicted_accel;
            pred.origin_time =
                (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0)
                    ? rclcpp::Time(msg->header.stamp)
                    : node->now();
            
            const double t_start = pred.origin_time.seconds();
            double dt = 0.05;
            if (msg->points.size() > 1) {
                dt = (rclcpp::Duration(msg->points[1].time_from_start) -
                      rclcpp::Duration(msg->points[0].time_from_start)).seconds();
            }
            
            std::vector<Eigen::Vector3d> accels;
            accels.reserve(msg->points.size());
            for (const auto& pt : msg->points) {
                if (!pt.accelerations.empty()) {
                    accels.emplace_back(pt.accelerations[0].linear.x,
                                        pt.accelerations[0].linear.y,
                                        pt.accelerations[0].linear.z);
                } else {
                    accels.emplace_back(0, 0, 0);
                }
            }
            
            pred.accel_buffer.t_start = t_start;
            pred.accel_buffer.dt = dt <= 0.0 ? 0.05 : dt;
            pred.accel_buffer.accels = std::move(accels);
            pred.received = true;
        },
        opts);

    RCLCPP_INFO(node->get_logger(),
        "[TargetTracker] Subscribed odom: %s  accel(diagnostics): %s  predicted_accel: %s",
        odom_topic.c_str(),
        accel_topic.empty() ? "<disabled>" : accel_topic.c_str(),
        predicted_accel_topic.c_str());
}

} // namespace target_tracker
} // namespace platform
