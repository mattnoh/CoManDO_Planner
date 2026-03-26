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
///
/// Usage:
///   ros2 run comando_planner circular_target_publisher
///   ros2 run comando_planner circular_target_publisher \
///       --ros-args -p radius:=3.0 -p omega:=0.5
///
/// Quick check:
///   ros2 topic echo /target/odom --once
///   ros2 topic echo /target/accel --once
///   ros2 topic hz /target/odom          # should print ~100 Hz
///   ros2 topic hz /target/accel         # should print ~100 Hz

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include "target/circular_target.hpp"

#include <cmath>
#include <chrono>

class CircularTargetPublisher : public rclcpp::Node
{
public:
    CircularTargetPublisher() : Node("circular_target_publisher")
    {
        // ── Parameters ──────────────────────────────────────────────────────
        center_x_   = declare_parameter<double>("center_x",   0.0);
        center_y_   = declare_parameter<double>("center_y",   0.0);
        center_z_   = declare_parameter<double>("center_z",   1.5);
        radius_     = declare_parameter<double>("radius",     1.0);
        omega_      = declare_parameter<double>("omega",      0.01);
        phi0_       = declare_parameter<double>("phi0",       0.0);
        publish_hz_ = declare_parameter<double>("publish_hz", 100.0);
        frame_id_   = declare_parameter<std::string>("frame_id", "world");

        // Record wall-clock start so visual phase matches requested phi0 at node startup
        phi0_ = phi0_ - omega_ * now().seconds();

        // ── Publishers ───────────────────────────────────────────────────────
        odom_pub_  = create_publisher<nav_msgs::msg::Odometry>(
            "/target/odom",  10);
        accel_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>(
            "/target/accel", 10);
        traj_pub_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
            "/target/predicted_trajectory", 10);

        // ── Timer ────────────────────────────────────────────────────────────
        const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / publish_hz_));
        timer_ = create_wall_timer(period_ns,
            std::bind(&CircularTargetPublisher::timerCallback, this));

        RCLCPP_INFO(get_logger(),
            "[CircularTarget] center=(%.2f,%.2f,%.2f)  R=%.2f  ω=%.2f rad/s  "
            "%.0f Hz",
            center_x_, center_y_, center_z_, radius_, omega_, publish_hz_);
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

    // ── ROS handles ────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr accel_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr traj_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CircularTargetPublisher>());
    rclcpp::shutdown();
    return 0;
}