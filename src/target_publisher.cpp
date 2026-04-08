/// @file target_publisher.cpp
/// @brief Standalone ROS2 node that publishes the target 
///
/// Publishes on:
///   /target/odom   (nav_msgs/msg/Odometry)   — position + velocity
///   /target/accel  (geometry_msgs/msg/AccelStamped) — centripetal acceleration
///
/// Parameters (all declare_parameter with defaults):
///   center_x, center_y, center_z  — orbit center [m]   default: 0, 0, 1.5
///   radius                         — orbit radius [m]   default: 2.0
///   omega                          — angular speed [rad/s]  default: 0.4
///   phi0                           — initial phase [rad]    default: 0.0
///   publish_hz                     — publish rate [Hz]      default: 100.0
///   frame_id                       — header frame           default: "world"
///   drone_odom_topic               — drone odom input       default: "/cf_1/odom"
///   relative_odom_topic            — relative odom output   default: "/drone/relative_odometry"
///   enable_relative_odom           — publish relative odom  default: true
///
/// Usage:
///   ros2 run comando_planner circular_target_publisher
///   ros2 run comando_planner circular_target_publisher \
///       --ros-args -p radius:=3.0 -p omega:=0.5
///
/// Quick check:
///   ros2 topic echo /target/odom --once
///   ros2 topic echo /target/accel --once
///   ros2 topic echo /drone/relative_odometry --once
///   ros2 topic hz /target/odom          # should print ~100 Hz
///   ros2 topic hz /target/accel         # should print ~100 Hz

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include "target/circular_target.hpp"

#include <cmath>
#include <chrono>
#include <mutex>

class CircularTargetPublisher : public rclcpp::Node
{
public:
    CircularTargetPublisher() : Node("circular_target_publisher")
    {
        // ── Parameters ──────────────────────────────────────────────────────
        center_x_   = declare_parameter<double>("center_x",   0.0);
        center_y_   = declare_parameter<double>("center_y",   0.0);
        center_z_   = declare_parameter<double>("center_z",   0.2);
        radius_     = declare_parameter<double>("radius",     1.0);
        omega_      = declare_parameter<double>("omega",      0.4);
        phi0_       = declare_parameter<double>("phi0",       0.0);
        publish_hz_ = declare_parameter<double>("publish_hz", 100.0);
        frame_id_   = declare_parameter<std::string>("frame_id", "world");
        drone_odom_topic_ = declare_parameter<std::string>("drone_odom_topic", "/cf_1/odom");
        relative_odom_topic_ = declare_parameter<std::string>("relative_odom_topic", "/drone/relative_odometry");
        enable_relative_odom_ = declare_parameter<bool>("enable_relative_odom", true);

        // Record wall-clock start so visual phase matches requested phi0 at node startup
        phi0_ = phi0_ - omega_ * now().seconds();

        // ── Publishers ───────────────────────────────────────────────────────
        odom_pub_  = create_publisher<nav_msgs::msg::Odometry>(
            "/target/odom",  10);
        accel_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>(
            "/target/accel", 10);
        traj_pub_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
            "/target/predicted_trajectory", 10);

        if (enable_relative_odom_) {
            rel_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(relative_odom_topic_, 10);
            drone_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
                drone_odom_topic_, 10,
                [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                    std::lock_guard<std::mutex> lk(drone_mutex_);
                    last_drone_odom_ = *msg;
                    has_drone_odom_ = true;
                });
        }

        // ── Timer ────────────────────────────────────────────────────────────
        const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / publish_hz_));
        timer_ = create_wall_timer(period_ns,
            std::bind(&CircularTargetPublisher::timerCallback, this));

        RCLCPP_INFO(get_logger(),
            "[CircularTarget] center=(%.2f,%.2f,%.2f)  R=%.2f  ω=%.2f rad/s  "
            "%.0f Hz",
            center_x_, center_y_, center_z_, radius_, omega_, publish_hz_);
        if (enable_relative_odom_) {
            RCLCPP_INFO(get_logger(),
                "[CircularTarget] Relative odom enabled: in=%s out=%s",
                drone_odom_topic_.c_str(), relative_odom_topic_.c_str());
        }
    }

private:
    void timerCallback()
    {
        const rclcpp::Time stamp = now();
        const double t = stamp.seconds();

        // ── Kinematics ───────────────────────────────────────────────────────
        target_models::CircularTarget tgt;
        tgt.center = {center_x_, center_y_, center_z_};
        tgt.R = radius_;
        tgt.omega = omega_;
        tgt.phi0 = phi0_;

        const auto px = tgt.pos(t).x();
        const auto py = tgt.pos(t).y();
        const auto pz = tgt.pos(t).z();

        const auto vx = tgt.vel(t).x();
        const auto vy = tgt.vel(t).y();
        const auto vz = tgt.vel(t).z();

        const auto ax = tgt.accel(t).x();
        const auto ay = tgt.accel(t).y();
        const auto az = tgt.accel(t).z();

        // ── Odometry message ─────────────────────────────────────────────────
        nav_msgs::msg::Odometry odom;
        odom.header.stamp    = stamp;
        odom.header.frame_id = frame_id_;
        odom.child_frame_id  = "target";

        odom.pose.pose.position.x = px;
        odom.pose.pose.position.y = py;
        odom.pose.pose.position.z = pz;
        // Orientation: identity quaternion (platform is not rotating about z here)
        odom.pose.pose.orientation.w = 1.0;
        odom.pose.pose.orientation.x = 0.0;
        odom.pose.pose.orientation.y = 0.0;
        odom.pose.pose.orientation.z = 0.0;

        odom.twist.twist.linear.x = vx;
        odom.twist.twist.linear.y = vy;
        odom.twist.twist.linear.z = vz;

        odom_pub_->publish(odom);

        // ── AccelStamped message ─────────────────────────────────────────────
        geometry_msgs::msg::AccelStamped accel;
        accel.header.stamp    = stamp;
        accel.header.frame_id = frame_id_;

        accel.accel.linear.x = ax;
        accel.accel.linear.y = ay;
        accel.accel.linear.z = az;
        // Angular acceleration is zero for a flat circular orbit
        accel.accel.angular.x = 0.0;
        accel.accel.angular.y = 0.0;
        accel.accel.angular.z = 0.0;

        accel_pub_->publish(accel);

        if (enable_relative_odom_ && rel_odom_pub_) {
            nav_msgs::msg::Odometry drone_odom_copy;
            {
                std::lock_guard<std::mutex> lk(drone_mutex_);
                if (!has_drone_odom_) {
                    goto maybe_publish_traj;
                }
                drone_odom_copy = last_drone_odom_;
            }

            const rclcpp::Time dstamp =
                (drone_odom_copy.header.stamp.sec != 0 || drone_odom_copy.header.stamp.nanosec != 0)
                ? rclcpp::Time(drone_odom_copy.header.stamp)
                : stamp;
            const double td = dstamp.seconds();

            const auto p_tgt = tgt.pos(td);
            const auto v_tgt = tgt.vel(td);

            nav_msgs::msg::Odometry rel = drone_odom_copy;
            rel.header.stamp = dstamp;
            rel.header.frame_id = frame_id_;
            rel.child_frame_id = "drone_relative";

            rel.pose.pose.position.x = drone_odom_copy.pose.pose.position.x - p_tgt.x();
            rel.pose.pose.position.y = drone_odom_copy.pose.pose.position.y - p_tgt.y();
            rel.pose.pose.position.z = drone_odom_copy.pose.pose.position.z - p_tgt.z();

            rel.twist.twist.linear.x = drone_odom_copy.twist.twist.linear.x - v_tgt.x();
            rel.twist.twist.linear.y = drone_odom_copy.twist.twist.linear.y - v_tgt.y();
            rel.twist.twist.linear.z = drone_odom_copy.twist.twist.linear.z - v_tgt.z();

            rel_odom_pub_->publish(rel);
        }

maybe_publish_traj:
        
        static int publish_traj_counter = 0;
        if (publish_traj_counter++ % static_cast<int>(publish_hz_ / 10.0) == 0) {
            trajectory_msgs::msg::MultiDOFJointTrajectory traj;
            traj.header.stamp = stamp;
            traj.header.frame_id = frame_id_;
            traj.joint_names.push_back("target");

            const int num_points = 400; // 20s at 0.05s resolution
            const double dt = 0.05;
            traj.points.reserve(num_points);

            for (int i = 0; i < num_points; ++i) {
                trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;
                pt.time_from_start = rclcpp::Duration::from_seconds(i * dt);

                double point_t = t + i * dt;
                auto p = tgt.pos(point_t);
                auto v = tgt.vel(point_t);
                auto a = tgt.accel(point_t);

                geometry_msgs::msg::Transform trans;
                trans.translation.x = p.x();
                trans.translation.y = p.y();
                trans.translation.z = p.z();
                trans.rotation.w = 1.0;
                trans.rotation.x = 0.0;
                trans.rotation.y = 0.0;
                trans.rotation.z = 0.0;
                pt.transforms.push_back(trans);

                geometry_msgs::msg::Twist vel;
                vel.linear.x = v.x();
                vel.linear.y = v.y();
                vel.linear.z = v.z();
                pt.velocities.push_back(vel);

                geometry_msgs::msg::Twist acc;
                acc.linear.x = a.x();
                acc.linear.y = a.y();
                acc.linear.z = a.z();
                pt.accelerations.push_back(acc);

                traj.points.push_back(pt);
            }
            traj_pub_->publish(traj);
        }

        // ── Diagnostics (every 5 s) ──────────────────────────────────────────
        static double last_diag = -5.0;
        if (t - last_diag >= 5.0) {
            last_diag = t;
            RCLCPP_DEBUG(get_logger(),
                "[CircularTarget] t=%.2f  pos=(%.3f,%.3f,%.3f)  "
                "vel=(%.3f,%.3f)  accel=(%.3f,%.3f)",
                t, px, py, pz, vx, vy, ax, ay);
        }
    }

    // ── Parameters ─────────────────────────────────────────────────────────
    double center_x_, center_y_, center_z_;
    double radius_, omega_, phi0_, publish_hz_;
    std::string frame_id_;
    std::string drone_odom_topic_;
    std::string relative_odom_topic_;
    bool enable_relative_odom_ = true;

    // ── ROS handles ────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        rel_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr accel_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr traj_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     drone_odom_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::mutex drone_mutex_;
    nav_msgs::msg::Odometry last_drone_odom_;
    bool has_drone_odom_ = false;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CircularTargetPublisher>());
    rclcpp::shutdown();
    return 0;
}