/// @file target_tracker.hpp
/// @brief Target-state subscriptions (odom + accel) for stateswitch OCP.
///
/// EXPECTED ROS GRAPH (Target Tracker):
/// - Planner node: `/comando_planner`
/// - Input topics (from target estimator/sim): `<target_odom_topic>`,
///   `<target_accel_topic>`
/// - Typical defaults in launch: `/target/odom`, `/target/accel`
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

#include <Eigen/Dense>
#include <mutex>
#include <string>

namespace platform {
namespace target_tracker {

struct TargetState {
    Eigen::Vector3d position     = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity     = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();

    // Separate timestamps per topic so each can be checked independently.
    rclcpp::Time odom_timestamp {0, 0, RCL_ROS_TIME};
    rclcpp::Time accel_timestamp{0, 0, RCL_ROS_TIME};

    // Startup flags: true after the first message of each type ever arrives.
    // Used only to gate early-startup "waiting for first message" warnings.
    // Do NOT use these to assess current freshness — call isFresh() instead.
    bool odom_received  = false;
    bool accel_received = false;

    /// Returns true only if BOTH topics have delivered at least one message
    /// AND both of their most recent messages are younger than max_age_sec.
    ///
    /// max_age_sec default = 0.2 s (5 Hz minimum acceptable rate).
    /// Raise to 0.5 s for slow targets; lower to 0.1 s for tight requirements.
    bool isFresh(const rclcpp::Time& now, double max_age_sec = 0.2) const {
        if (!odom_received || !accel_received) return false;

        // Guard against uninitialised timestamps (nanoseconds == 0 at default init).
        if (odom_timestamp.nanoseconds()  <= 0) return false;
        if (accel_timestamp.nanoseconds() <= 0) return false;

        const double odom_age  = (now - odom_timestamp).seconds();
        const double accel_age = (now - accel_timestamp).seconds();

        // Allow a small negative tolerance (±1 ms) to absorb clock jitter and
        // sim-time initialisation artefacts.
        const double kClockTol = 0.001;
        if (odom_age  < -kClockTol || odom_age  >= max_age_sec) return false;
        if (accel_age < -kClockTol || accel_age >= max_age_sec) return false;

        return true;
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
};

struct Handles {
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       odom_sub;
    rclcpp::Subscription<geometry_msgs::msg::AccelStamped>::SharedPtr accel_sub;

    void reset() {
        odom_sub.reset();
        accel_sub.reset();
    }
};

inline void setup(
    rclcpp::Node*                        node,
    rclcpp::CallbackGroup::SharedPtr     callback_group,
    const std::string&                   odom_topic,
    const std::string&                   accel_topic,
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
            target_state.odom_received = true;

            // Update ONLY the odom timestamp; accel timestamp is unaffected.
            target_state.odom_timestamp =
                (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0)
                    ? rclcpp::Time(msg->header.stamp)
                    : node->now();
        },
        opts);

    // ── Acceleration callback ────────────────────────────────────────────────
    handles.accel_sub = node->create_subscription<geometry_msgs::msg::AccelStamped>(
        accel_topic, 10,
        [&target_state, &target_mutex, node](
            const geometry_msgs::msg::AccelStamped::SharedPtr msg)
        {
            std::lock_guard<std::mutex> lk(target_mutex);

            target_state.acceleration << msg->accel.linear.x,
                                         msg->accel.linear.y,
                                         msg->accel.linear.z;
            target_state.accel_received = true;

            // Update ONLY the accel timestamp; odom timestamp is unaffected.
            target_state.accel_timestamp =
                (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0)
                    ? rclcpp::Time(msg->header.stamp)
                    : node->now();
        },
        opts);

    RCLCPP_INFO(node->get_logger(),
        "[TargetTracker] Subscribed odom: %s  accel: %s",
        odom_topic.c_str(), accel_topic.c_str());
}

} // namespace target_tracker
} // namespace platform