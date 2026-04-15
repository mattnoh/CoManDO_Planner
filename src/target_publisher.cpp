/// @file target_publisher.cpp
/// @brief Standalone ROS2 node that publishes the target state for the CoManDO planner.
///
/// Two modes (target_mode parameter):
///
///   "circle"   — Fully synthetic circular orbit. No external input.
///                Phase is driven by wall-clock time and phi0.
///
///   "qualisys" — Real vehicle pose from a Qualisys-sourced PoseStamped topic.
///                Position comes from the observed pose. Velocity, acceleration,
///                and the predicted_accel trajectory are computed analytically
///                from a circular motion model (center + omega). The phase is
///                re-anchored to the real pose on every update, so the model
///                stays locked to the actual vehicle while remaining smooth.
///
/// Publishes on:
///   /target/odom             (nav_msgs/msg/Odometry)
///   /target/accel            (geometry_msgs/msg/AccelStamped)
///   /target/predicted_accel  (trajectory_msgs/msg/MultiDOFJointTrajectory)
///
/// Parameters:
///   target_mode         — "circle" or "qualisys"           default: "circle"
///   center_x/y/z        — orbit center [m]                 default: 0, 0, 0.2
///   omega               — angular speed [rad/s]            default: 0.4
///   phi0                — initial phase, circle mode [rad] default: 0.0
///   radius              — orbit radius, circle mode [m]    default: 1.0
///   publish_hz          — publish rate [Hz]                default: 100.0
///   frame_id            — header frame                     default: "world"
///   qualisys_pose_topic — input pose topic, qualisys mode  default: "/stmini/pose"
///   drone_odom_topic    — drone odom for relative odom     default: "/gogogo/odom"
///   relative_odom_topic — relative odom output topic       default: "/drone/relative_odometry"
///   enable_relative_odom — publish drone relative odom     default: true
///
/// Usage:
///   ros2 run comando_planner target_publisher --ros-args -p target_mode:=circle
///   ros2 run comando_planner target_publisher --ros-args \
///       -p target_mode:=qualisys -p center_x:=1.0 -p center_y:=0.5 -p omega:=0.3

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#ifdef HAS_MOCAP4R2_MSGS
#include <mocap4r2_msgs/msg/rigid_bodies.hpp>
#endif
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include <Eigen/Geometry>

#include "target/circular_target.hpp"
#include "target/figure8_target.hpp"

#include <cmath>
#include <chrono>
#include <mutex>

namespace {
constexpr double DEG2RAD = M_PI / 180.0;
}

class BenchmarkTargetPublisher : public rclcpp::Node
{
public:
    BenchmarkTargetPublisher() : Node("circular_target_publisher")
    {
        // ── Parameters ──────────────────────────────────────────────────────
        target_mode_ = declare_parameter<std::string>("target_mode", "circle");
        center_x_    = declare_parameter<double>("center_x",   0.0);
        center_y_    = declare_parameter<double>("center_y",   0.0);
        center_z_    = declare_parameter<double>("center_z",   0.4);
        radius_      = declare_parameter<double>("radius",     1.0);
        omega_       = declare_parameter<double>("omega",      0.3);
        phi0_        = declare_parameter<double>("phi0",       0.0);
        publish_hz_  = declare_parameter<double>("publish_hz", 100.0);
        frame_id_    = declare_parameter<std::string>("frame_id", "world");
        qualisys_pose_topic_  = declare_parameter<std::string>("rigid_body_name", "stmini");
        drone_odom_topic_     = declare_parameter<std::string>("drone_odom_topic", "/cf_1/odom");
        absolute_relative_odom_topic_  = declare_parameter<std::string>("absolute_relative_odom_topic", "/drone/relative_odometry");
        enable_absolute_relative_odom_ = declare_parameter<bool>("enable_absolute_relative_odom", true);
        enable_body_relative_odom_ = declare_parameter<bool>("enable_body_relative_odom", false);
        body_relative_odom_topic_ = declare_parameter<std::string>("body_relative_odom_topic", "/drone/body_relative_odom");
        drone_pose_topic_ = declare_parameter<std::string>("drone_pose_topic", "/cf_1/pose");
        debug_body_relative_trace_ = declare_parameter<bool>("debug_body_relative_trace", false);
        body_relative_angular_unit_ = declare_parameter<std::string>(
            "body_relative_input_angular_unit", "deg_s");

#ifndef HAS_MOCAP4R2_MSGS
        if (target_mode_ == "qualisys") {
            RCLCPP_WARN(get_logger(),
                "[TargetPublisher] target_mode=qualisys requested, but this build has no "
                "mocap4r2_msgs support. Falling back to circle mode.");
            target_mode_ = "circle";
        }
#endif

        // circle mode: shift phi0 so phase matches requested value at node startup
        phi0_ = phi0_ - omega_ * now().seconds();

        // ── Publishers ───────────────────────────────────────────────────────
        odom_pub_  = create_publisher<nav_msgs::msg::Odometry>("/target/odom", 10);
        accel_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>("/target/accel", 10);
        traj_pub_  = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
            "/target/predicted_accel", 10);

        if (enable_absolute_relative_odom_) {
            abs_rel_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(absolute_relative_odom_topic_, 10);
        }
        if (enable_body_relative_odom_) {
            body_rel_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(body_relative_odom_topic_, 10);
        }
        if (enable_absolute_relative_odom_ || enable_body_relative_odom_) {
            drone_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
                drone_odom_topic_, 10,
                [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                    std::lock_guard<std::mutex> lk(drone_mutex_);
                    last_drone_odom_ = *msg;
                    has_drone_odom_ = true;
                });
        }
        if (enable_body_relative_odom_) {
            drone_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
                drone_pose_topic_, 10,
                [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                    std::lock_guard<std::mutex> lk(drone_mutex_);
                    last_drone_pose_ = *msg;
                    has_drone_pose_ = true;
                });
        }

        // ── Qualisys rigid bodies subscriber (qualisys mode only) ───────────
#ifdef HAS_MOCAP4R2_MSGS
        if (target_mode_ == "qualisys") {
            rigid_bodies_sub_ = create_subscription<mocap4r2_msgs::msg::RigidBodies>(
                "/rigid_bodies", 10,
                [this](const mocap4r2_msgs::msg::RigidBodies::SharedPtr msg) {
                    RCLCPP_INFO_ONCE(get_logger(),
                        "[TargetPublisher] /rigid_bodies callback firing, %zu bodies",
                        msg->rigidbodies.size());
                    for (const auto & rb : msg->rigidbodies) {
                        RCLCPP_INFO_ONCE(get_logger(),
                            "[TargetPublisher] body seen: '%s'", rb.rigid_body_name.c_str());
                        if (rb.rigid_body_name != qualisys_pose_topic_) continue;
                        const double dx = rb.pose.position.x - center_x_;
                        const double dy = rb.pose.position.y - center_y_;
                        std::lock_guard<std::mutex> lk(pose_mutex_);
                        last_phi_    = std::atan2(dy, dx);
                        last_R_      = std::sqrt(dx * dx + dy * dy);
                        last_z_      = rb.pose.position.z;
                        last_pose_t_ = rclcpp::Time(msg->header.stamp).seconds();
                        has_pose_    = true;
                        break;
                    }
                });
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] mode=qualisys  body_name=%s  "
                "center=(%.2f,%.2f,%.2f)  ω=%.2f rad/s",
                qualisys_pose_topic_.c_str(),
                center_x_, center_y_, center_z_, omega_);
        } else {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] mode=circle  center=(%.2f,%.2f,%.2f)  "
                "R=%.2f  ω=%.2f rad/s  %.0f Hz",
                center_x_, center_y_, center_z_, radius_, omega_, publish_hz_);
        }
#else
        RCLCPP_INFO(get_logger(),
            "[TargetPublisher] mode=circle  center=(%.2f,%.2f,%.2f)  "
            "R=%.2f  ω=%.2f rad/s  %.0f Hz",
            center_x_, center_y_, center_z_, radius_, omega_, publish_hz_);
#endif

        if (enable_absolute_relative_odom_) {
            RCLCPP_INFO(get_logger(),
            "[TargetPublisher] Absolute-relative odom enabled: in=%s out=%s",
            drone_odom_topic_.c_str(), absolute_relative_odom_topic_.c_str());
        }
        if (enable_body_relative_odom_) {
            RCLCPP_INFO(get_logger(),
            "[TargetPublisher] Body-relative odom enabled: pose=%s odom=%s out=%s",
            drone_pose_topic_.c_str(), drone_odom_topic_.c_str(), body_relative_odom_topic_.c_str());
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] Body-relative angular input unit='%s' (set body_relative_input_angular_unit to 'rad_s' if upstream already publishes rad/s)",
                body_relative_angular_unit_.c_str());
        }

        // ── Timer ────────────────────────────────────────────────────────────
        const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / publish_hz_));
        timer_ = create_wall_timer(period_ns,
            std::bind(&BenchmarkTargetPublisher::timerCallback, this));
    }

private:

    // ── Kinematics from phase ────────────────────────────────────────────────
    // All returned in world frame. R is the instantaneous orbit radius.
    struct Kinematics {
        double px, py, pz;
        double vx, vy, vz;
        double ax, ay, az;
    };

    Kinematics circleKinematics(double phi, double R, double pz) const
    {
        const double cp = std::cos(phi);
        const double sp = std::sin(phi);
        return {
            center_x_ + R * cp,          // px
            center_y_ + R * sp,          // py
            pz,                          // pz
            -omega_ * R * sp,            // vx
             omega_ * R * cp,            // vy
            0.0,                         // vz
            -omega_ * omega_ * R * cp,   // ax  (centripetal)
            -omega_ * omega_ * R * sp,   // ay
            0.0                          // az
        };
    }

    // ── Timer callback ───────────────────────────────────────────────────────
    void timerCallback()
    {
        const rclcpp::Time stamp = now();
        const double t = stamp.seconds();

        Kinematics k{};
        double phi_now = 0.0;
        double R_now   = radius_;

        if (target_mode_ == "qualisys") {
            // ── qualisys mode: re-anchor phase from latest observed pose ─────
            double phi_obs, R_obs, z_obs, t_obs;
            {
                std::lock_guard<std::mutex> lk(pose_mutex_);
                if (!has_pose_) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                        "[TargetPublisher] Waiting for '%s' in /rigid_bodies ...",
                        qualisys_pose_topic_.c_str());
                    return;
                }
                phi_obs = last_phi_;
                R_obs   = last_R_;
                z_obs   = last_z_;
                t_obs   = last_pose_t_;
            }

            // Extrapolate phase from last observation to now
            phi_now = phi_obs + omega_ * (t - t_obs);
            R_now   = R_obs;
            k = circleKinematics(phi_now, R_now, z_obs);

        } else {
            // ── circle mode: fully synthetic ─────────────────────────────────
            phi_now = phi0_ + omega_ * t;
            k = circleKinematics(phi_now, radius_, center_z_);
        }

        // ── Odometry message ─────────────────────────────────────────────────
        nav_msgs::msg::Odometry odom;
        odom.header.stamp    = stamp;
        odom.header.frame_id = frame_id_;
        odom.child_frame_id  = "target";

        odom.pose.pose.position.x    = k.px;
        odom.pose.pose.position.y    = k.py;
        odom.pose.pose.position.z    = k.pz;
        odom.pose.pose.orientation.w = 1.0;

        odom.twist.twist.linear.x = k.vx;
        odom.twist.twist.linear.y = k.vy;
        odom.twist.twist.linear.z = k.vz;

        odom_pub_->publish(odom);

        // ── AccelStamped message ─────────────────────────────────────────────
        geometry_msgs::msg::AccelStamped accel;
        accel.header.stamp    = stamp;
        accel.header.frame_id = frame_id_;
        accel.accel.linear.x  = k.ax;
        accel.accel.linear.y  = k.ay;
        accel.accel.linear.z  = k.az;

        accel_pub_->publish(accel);

        // ── Relative odometry outputs ───────────────────────────────────────
        nav_msgs::msg::Odometry drone_odom_copy;
        geometry_msgs::msg::PoseStamped drone_pose_copy;
        bool have_drone_odom = false;
        bool have_drone_pose = false;
        {
            std::lock_guard<std::mutex> lk(drone_mutex_);
            have_drone_odom = has_drone_odom_;
            have_drone_pose = has_drone_pose_;
            if (have_drone_odom) {
                drone_odom_copy = last_drone_odom_;
            }
            if (have_drone_pose) {
                drone_pose_copy = last_drone_pose_;
            }
        }

        if (enable_absolute_relative_odom_ && abs_rel_odom_pub_ && have_drone_odom) {
            nav_msgs::msg::Odometry rel = drone_odom_copy;
            rel.header.stamp    = drone_odom_copy.header.stamp;
            rel.header.frame_id = frame_id_;
            rel.child_frame_id  = "drone_absolute_relative";

            rel.pose.pose.position.x = drone_odom_copy.pose.pose.position.x - k.px;
            rel.pose.pose.position.y = drone_odom_copy.pose.pose.position.y - k.py;
            rel.pose.pose.position.z = drone_odom_copy.pose.pose.position.z - k.pz;

            rel.twist.twist.linear.x = drone_odom_copy.twist.twist.linear.x - k.vx;
            rel.twist.twist.linear.y = drone_odom_copy.twist.twist.linear.y - k.vy;
            rel.twist.twist.linear.z = drone_odom_copy.twist.twist.linear.z - k.vz;

            abs_rel_odom_pub_->publish(rel);
        }

        if (enable_body_relative_odom_ && body_rel_odom_pub_ && have_drone_odom && have_drone_pose) {
            const Eigen::Vector3d target_world(k.px, k.py, k.pz);
            const Eigen::Vector3d drone_world(
                drone_pose_copy.pose.position.x,
                drone_pose_copy.pose.position.y,
                drone_pose_copy.pose.position.z);
            const Eigen::Vector3d drone_vel_world(
                drone_odom_copy.twist.twist.linear.x,
                drone_odom_copy.twist.twist.linear.y,
                drone_odom_copy.twist.twist.linear.z);

            Eigen::Quaterniond q(
                drone_pose_copy.pose.orientation.w,
                drone_pose_copy.pose.orientation.x,
                drone_pose_copy.pose.orientation.y,
                drone_pose_copy.pose.orientation.z);
            q.normalize();
            const Eigen::Matrix3d R = q.toRotationMatrix();

            const Eigen::Vector3d p_t_body = R.transpose() * (target_world - drone_world);
            const Eigen::Vector3d v_d_body = R.transpose() * drone_vel_world;

            nav_msgs::msg::Odometry body_rel;
            body_rel.header.stamp = stamp;
            body_rel.header.frame_id = "drone_body";
            body_rel.child_frame_id = "body_relative";
            body_rel.pose.pose.position.x = p_t_body.x();
            body_rel.pose.pose.position.y = p_t_body.y();
            body_rel.pose.pose.position.z = p_t_body.z();
            body_rel.pose.pose.orientation = drone_pose_copy.pose.orientation;
            body_rel.twist.twist.linear.x = v_d_body.x();
            body_rel.twist.twist.linear.y = v_d_body.y();
            body_rel.twist.twist.linear.z = v_d_body.z();
            const double ang_scale = (body_relative_angular_unit_ == "rad_s") ? 1.0 : DEG2RAD;
            body_rel.twist.twist.angular.x = drone_odom_copy.twist.twist.angular.x * ang_scale;
            body_rel.twist.twist.angular.y = drone_odom_copy.twist.twist.angular.y * ang_scale;
            body_rel.twist.twist.angular.z = drone_odom_copy.twist.twist.angular.z * ang_scale;

            body_rel_odom_pub_->publish(body_rel);

            if (debug_body_relative_trace_) {
                RCLCPP_INFO_THROTTLE(
                    get_logger(), *get_clock(), 1000,
                    "[BodyRelTrace] raw_w=(%.4f,%.4f,%.4f) converted_w=(%.4f,%.4f,%.4f) unit=%s q=(%.4f,%.4f,%.4f,%.4f)|norm=%.4f pT=(%.3f,%.3f,%.3f) vdb=(%.3f,%.3f,%.3f)",
                    drone_odom_copy.twist.twist.angular.x,
                    drone_odom_copy.twist.twist.angular.y,
                    drone_odom_copy.twist.twist.angular.z,
                    body_rel.twist.twist.angular.x,
                    body_rel.twist.twist.angular.y,
                    body_rel.twist.twist.angular.z,
                    body_relative_angular_unit_.c_str(),
                    q.w(), q.x(), q.y(), q.z(), q.norm(),
                    p_t_body.x(), p_t_body.y(), p_t_body.z(),
                    v_d_body.x(), v_d_body.y(), v_d_body.z());
            }
        }

        // ── Predicted acceleration trajectory (10 Hz) ────────────────────────
        static int traj_counter = 0;
        if (traj_counter++ % static_cast<int>(publish_hz_ / 10.0) == 0) {
            trajectory_msgs::msg::MultiDOFJointTrajectory traj;
            traj.header.stamp    = stamp;
            traj.header.frame_id = frame_id_;
            traj.joint_names.push_back("target");

            constexpr int   NUM_POINTS = 400;   // 20 s at 0.05 s resolution
            constexpr double DT        = 0.05;
            traj.points.reserve(NUM_POINTS);

            for (int i = 0; i < NUM_POINTS; ++i) {
                const double phi_i = phi_now + omega_ * i * DT;
                const Kinematics ki = circleKinematics(phi_i, R_now,
                    (target_mode_ == "qualisys") ? k.pz : center_z_);

                trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;
                pt.time_from_start = rclcpp::Duration::from_seconds(i * DT);

                geometry_msgs::msg::Transform trans;
                trans.translation.x = ki.px;
                trans.translation.y = ki.py;
                trans.translation.z = ki.pz;
                trans.rotation.w    = 1.0;
                pt.transforms.push_back(trans);

                geometry_msgs::msg::Twist vel;
                vel.linear.x = ki.vx;
                vel.linear.y = ki.vy;
                vel.linear.z = ki.vz;
                pt.velocities.push_back(vel);

                geometry_msgs::msg::Twist acc;
                acc.linear.x = ki.ax;
                acc.linear.y = ki.ay;
                acc.linear.z = ki.az;
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
                "[TargetPublisher] t=%.2f  phi=%.3f  R=%.3f  "
                "pos=(%.3f,%.3f,%.3f)  vel=(%.3f,%.3f)",
                t, phi_now, R_now, k.px, k.py, k.pz, k.vx, k.vy);
        }
    }

    // ── Parameters ─────────────────────────────────────────────────────────
    std::string target_mode_;
    double center_x_, center_y_, center_z_;
    double radius_, amp_x_, amp_y_, omega_, phi0_, publish_hz_;
    std::string frame_id_;
    std::string qualisys_pose_topic_;
    std::string drone_odom_topic_;
    std::string absolute_relative_odom_topic_;
    bool enable_absolute_relative_odom_ = true;
    bool enable_body_relative_odom_ = false;
    std::string body_relative_odom_topic_;
    std::string drone_pose_topic_;
    bool debug_body_relative_trace_ = false;
    std::string body_relative_angular_unit_ = "deg_s";

    // ── ROS handles ────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr                      odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr                      abs_rel_odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr                      body_rel_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr             accel_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr traj_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr                   drone_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr           drone_pose_sub_;
#ifdef HAS_MOCAP4R2_MSGS
    rclcpp::Subscription<mocap4r2_msgs::msg::RigidBodies>::SharedPtr            rigid_bodies_sub_;
#endif
    rclcpp::TimerBase::SharedPtr timer_;

    // ── Qualisys state (guarded by pose_mutex_) ─────────────────────────────
    std::mutex pose_mutex_;
    double last_phi_    = 0.0;
    double last_R_      = 1.0;
    double last_z_      = 0.0;
    double last_pose_t_ = 0.0;
    bool   has_pose_    = false;

    // ── Drone odom (guarded by drone_mutex_) ────────────────────────────────
    std::mutex drone_mutex_;
    nav_msgs::msg::Odometry last_drone_odom_;
    geometry_msgs::msg::PoseStamped last_drone_pose_;
    bool has_drone_odom_ = false;
    bool has_drone_pose_ = false;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BenchmarkTargetPublisher>());
    rclcpp::shutdown();
    return 0;
}
