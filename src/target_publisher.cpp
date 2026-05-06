/// @file target_publisher.cpp
/// @brief Standalone ROS2 node that publishes the target state for the CoManDO planner.
///
/// Two modes (target_mode parameter):
///
///   "circle"   — Fully synthetic circular orbit. No external input.
///                Phase is driven by wall-clock time and phi0.
///
///   "qualisys" — Real vehicle pose from a Qualisys-sourced rigid-bodies topic.
///                Position comes from the observed pose. Velocity and acceleration
///                are estimated by finite-differencing a short pose history window.
///                No circular model assumed — works for any target trajectory.
///
/// Publishes on:
///   /target/odom   (nav_msgs/msg/Odometry)
///   /target/accel  (geometry_msgs/msg/AccelStamped)
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
///   ros2 run comando_planner target_publisher --ros-args
///       -p target_mode:=qualisys -p rigid_body_name:=stmini

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#ifdef HAS_MOCAP4R2_MSGS
#include <mocap4r2_msgs/msg/rigid_bodies.hpp>
#endif
#include <Eigen/Geometry>
#include <deque>

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
        // Circle trajectory shape is configured in include/target/circular_target.hpp
        target_mode_ = declare_parameter<std::string>("target_mode", "circle");
        target_yaw_rate_ = declare_parameter<double>("target_yaw_rate", 0.3);
        publish_hz_  = declare_parameter<double>("publish_hz", 100.0);
        frame_id_    = declare_parameter<std::string>("frame_id", "world");
        qualisys_pose_topic_  = declare_parameter<std::string>("rigid_body_name", "stmini");
        drone_odom_topic_     = declare_parameter<std::string>("drone_odom_topic", "/cf_1/odom");
        // drone_odom_mode: which relative odometry to publish alongside /target/odom
        //   "none"         — no drone relative output (absolute OCPs)
        //   "shifted_world" — /drone/relative_odometry: p_drone-p_tgt, v_drone-v_tgt in world
        //   "target_frame" — /drone/target_frame_odom: p_B^N, v_B^N, q_NB
        //   "body_frame"   — /drone/body_relative_odom: p_T^B, v_rel^B, q_NB
        drone_odom_mode_ = declare_parameter<std::string>("drone_odom_mode", "none");
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

        t0_ = now().seconds();  // circle mode uses relative time from here

        // ── Publishers ───────────────────────────────────────────────────────
        odom_pub_  = create_publisher<nav_msgs::msg::Odometry>("/target/odom", 10);
        accel_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>("/target/accel", 10);

        if (drone_odom_mode_ == "shifted_world") {
            abs_rel_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/drone/relative_odometry", 10);
        } else if (drone_odom_mode_ == "body_frame") {
            body_rel_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/drone/body_relative_odom", 10);
        } else if (drone_odom_mode_ == "target_frame") {
            target_frame_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/drone/target_frame_odom", 10);
        }
        if (drone_odom_mode_ != "none") {
            drone_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
                drone_odom_topic_, 10,
                [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                    std::lock_guard<std::mutex> lk(drone_mutex_);
                    last_drone_odom_ = *msg;
                    has_drone_odom_ = true;
                });
        }
        if (drone_odom_mode_ == "body_frame" || drone_odom_mode_ == "target_frame") {
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
                        PoseObs obs;
                        obs.t   = rclcpp::Time(msg->header.stamp).seconds();
                        obs.pos = {rb.pose.position.x, rb.pose.position.y, rb.pose.position.z};
                        obs.q   = Eigen::Quaterniond(rb.pose.orientation.w, rb.pose.orientation.x, rb.pose.orientation.y, rb.pose.orientation.z);
                        obs.q.normalize();
                        std::lock_guard<std::mutex> lk(pose_mutex_);
                        pose_history_.push_back(obs);
                        if (pose_history_.size() > N_POSE_HISTORY) {
                            pose_history_.pop_front();
                        }
                        break;
                    }
                });
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] mode=qualisys  body_name=%s",
                qualisys_pose_topic_.c_str());
        } else {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] mode=circle  center=(%.2f,%.2f,%.2f)  "
                "R=%.2f  ω=%.2f rad/s  %.0f Hz",
                circle_model_.center.x(), circle_model_.center.y(), circle_model_.center.z(),
                circle_model_.R, circle_model_.omega, publish_hz_);
        }
#else
        RCLCPP_INFO(get_logger(),
            "[TargetPublisher] mode=circle  center=(%.2f,%.2f,%.2f)  "
            "R=%.2f  ω=%.2f rad/s  %.0f Hz",
            circle_model_.center.x(), circle_model_.center.y(), circle_model_.center.z(),
            circle_model_.R, circle_model_.omega, publish_hz_);
#endif

        if (drone_odom_mode_ == "shifted_world") {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] drone_odom_mode=shifted_world → /drone/relative_odometry (drone-target in world frame)");
        } else if (drone_odom_mode_ == "body_frame") {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] drone_odom_mode=body_frame → /drone/body_relative_odom (target in drone body frame)");
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] Angular unit='%s' (set body_relative_input_angular_unit:=rad_s if upstream publishes rad/s)",
                body_relative_angular_unit_.c_str());
        } else if (drone_odom_mode_ == "target_frame") {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] drone_odom_mode=target_frame → /drone/target_frame_odom (drone in target N frame)");
        } else {
            RCLCPP_INFO(get_logger(), "[TargetPublisher] drone_odom_mode=none — no relative odometry output");
        }

        // ── Timer ────────────────────────────────────────────────────────────
        const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / publish_hz_));
        timer_ = create_wall_timer(period_ns,
            std::bind(&BenchmarkTargetPublisher::timerCallback, this));
    }

private:

    struct Kinematics {
        double px, py, pz;
        double vx, vy, vz;
        double ax, ay, az;
        double wx, wy, wz;
        double alfx, alfy, alfz;
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    };

    // ── Timer callback ───────────────────────────────────────────────────────
    void timerCallback()
    {
        const rclcpp::Time stamp = now();
        const double t = stamp.seconds();

        Kinematics k{};

        if (target_mode_ == "qualisys") {
            // ── qualisys mode: finite-differenced velocity and acceleration ──
            std::lock_guard<std::mutex> lk(pose_mutex_);
            if (pose_history_.empty()) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "[TargetPublisher] Waiting for '%s' in /rigid_bodies ...",
                    qualisys_pose_topic_.c_str());
                return;
            }
            const auto& latest = pose_history_.back();
            k.px = latest.pos.x();
            k.py = latest.pos.y();
            k.pz = latest.pos.z();
            k.q  = latest.q;
            
            Eigen::Vector3d w0 = Eigen::Vector3d::Zero();
            Eigen::Vector3d w1 = Eigen::Vector3d::Zero();

            if (pose_history_.size() >= 2) {
                const auto& oldest = pose_history_.front();
                const double dt = latest.t - oldest.t;
                if (dt > 1e-6) {
                    const Eigen::Vector3d vel = (latest.pos - oldest.pos) / dt;
                    k.vx = vel.x(); k.vy = vel.y(); k.vz = vel.z();
                    
                    Eigen::Quaterniond dq = latest.q * oldest.q.conjugate();
                    Eigen::AngleAxisd aa(dq);
                    Eigen::Vector3d w = aa.axis() * aa.angle() / dt;
                    k.wx = w.x(); k.wy = w.y(); k.wz = w.z();
                }
            }
            if (pose_history_.size() >= 3) {
                const auto& p0   = pose_history_.front();
                const auto& pmid = pose_history_[pose_history_.size() / 2];
                const auto& p1   = pose_history_.back();
                const double dt0 = pmid.t - p0.t;
                const double dt1 = p1.t  - pmid.t;
                if (dt0 > 1e-6 && dt1 > 1e-6) {
                    const Eigen::Vector3d v0  = (pmid.pos - p0.pos) / dt0;
                    const Eigen::Vector3d v1  = (p1.pos  - pmid.pos) / dt1;
                    const Eigen::Vector3d acc = (v1 - v0) / (0.5 * (dt0 + dt1));
                    k.ax = acc.x(); k.ay = acc.y(); k.az = acc.z();
                    
                    Eigen::Quaterniond dq0 = pmid.q * p0.q.conjugate();
                    Eigen::AngleAxisd aa0(dq0);
                    w0 = aa0.axis() * aa0.angle() / dt0;
                    
                    Eigen::Quaterniond dq1 = p1.q * pmid.q.conjugate();
                    Eigen::AngleAxisd aa1(dq1);
                    w1 = aa1.axis() * aa1.angle() / dt1;
                    
                    Eigen::Vector3d alf = (w1 - w0) / (0.5 * (dt0 + dt1));
                    k.alfx = alf.x(); k.alfy = alf.y(); k.alfz = alf.z();
                }
            }

        } else {
            // ── circle mode: fully synthetic, trajectory defined in circular_target.hpp
            const double t_rel = t - t0_;
            const Eigen::Vector3d p = circle_model_.pos(t_rel);
            const Eigen::Vector3d v = circle_model_.vel(t_rel);
            const Eigen::Vector3d a = circle_model_.accel(t_rel);
            k = {p.x(), p.y(), p.z(), v.x(), v.y(), v.z(), a.x(), a.y(), a.z(), 0.0, 0.0, target_yaw_rate_, 0.0, 0.0, 0.0, Eigen::Quaterniond::Identity()};
            
            // Yaw model
            double psi = target_yaw_rate_ * t_rel;
            k.q = Eigen::Quaterniond(std::cos(psi/2.0), 0.0, 0.0, std::sin(psi/2.0));
        }

        // ── Odometry message ─────────────────────────────────────────────────
        nav_msgs::msg::Odometry odom;
        odom.header.stamp    = stamp;
        odom.header.frame_id = frame_id_;
        odom.child_frame_id  = "target";

        odom.pose.pose.position.x    = k.px;
        odom.pose.pose.position.y    = k.py;
        odom.pose.pose.position.z    = k.pz;
        odom.pose.pose.orientation.w = k.q.w();
        odom.pose.pose.orientation.x = k.q.x();
        odom.pose.pose.orientation.y = k.q.y();
        odom.pose.pose.orientation.z = k.q.z();

        odom.twist.twist.linear.x = k.vx;
        odom.twist.twist.linear.y = k.vy;
        odom.twist.twist.linear.z = k.vz;

        odom.twist.twist.angular.x = k.wx;
        odom.twist.twist.angular.y = k.wy;
        odom.twist.twist.angular.z = k.wz;

        // Record target quaternion for relative odom computations below.
        last_target_quat_ = k.q;

        odom_pub_->publish(odom);

        // ── AccelStamped message ─────────────────────────────────────────────
        geometry_msgs::msg::AccelStamped accel;
        accel.header.stamp    = stamp;
        accel.header.frame_id = frame_id_;
        accel.accel.linear.x  = k.ax;
        accel.accel.linear.y  = k.ay;
        accel.accel.linear.z  = k.az;
        
        accel.accel.angular.x = k.alfx;
        accel.accel.angular.y = k.alfy;
        accel.accel.angular.z = k.alfz;

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

        if (drone_odom_mode_ == "shifted_world" && abs_rel_odom_pub_ && have_drone_odom) {
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

        if (drone_odom_mode_ == "body_frame" && body_rel_odom_pub_ && have_drone_odom && have_drone_pose) {
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

            const Eigen::Vector3d target_vel_world(k.vx, k.vy, k.vz);
            const Eigen::Vector3d p_t_body = R.transpose() * (target_world - drone_world);
            const Eigen::Vector3d v_d_body = R.transpose() * (drone_vel_world - target_vel_world);

            nav_msgs::msg::Odometry body_rel;
            body_rel.header.stamp = stamp;
            body_rel.header.frame_id = "drone_body";
            body_rel.child_frame_id = "body_relative";
            body_rel.pose.pose.position.x = p_t_body.x();
            body_rel.pose.pose.position.y = p_t_body.y();
            body_rel.pose.pose.position.z = p_t_body.z();
            // q_NB = q_target^{-1} * q_drone  (relative attitude: drone wrt target frame)
            {
                const Eigen::Quaterniond q_NB = last_target_quat_.conjugate() * q;
                body_rel.pose.pose.orientation.w = q_NB.w();
                body_rel.pose.pose.orientation.x = q_NB.x();
                body_rel.pose.pose.orientation.y = q_NB.y();
                body_rel.pose.pose.orientation.z = q_NB.z();
            }
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

        if (drone_odom_mode_ == "target_frame" && target_frame_odom_pub_ && have_drone_odom && have_drone_pose) {
            // Cheating estimator: uses ground-truth world-frame drone pose/odom to compute
            // what an ideal target-frame relative estimator would output for the tf OCP.
            // p_B^N = R_NW * (p_drone - p_tgt)
            // v_B^N = R_NW * (v_drone - v_tgt) - Om_N x p_B^N
            // q_NB  = q_target^{-1} * q_drone
            const Eigen::Matrix3d R_NW = last_target_quat_.conjugate().toRotationMatrix();
            const Eigen::Vector3d omega_W(k.wx, k.wy, k.wz);
            const Eigen::Vector3d Om_N = R_NW * omega_W;

            const Eigen::Vector3d p_drone(drone_pose_copy.pose.position.x,
                                           drone_pose_copy.pose.position.y,
                                           drone_pose_copy.pose.position.z);
            const Eigen::Vector3d v_drone(drone_odom_copy.twist.twist.linear.x,
                                           drone_odom_copy.twist.twist.linear.y,
                                           drone_odom_copy.twist.twist.linear.z);
            const Eigen::Vector3d p_tgt(k.px, k.py, k.pz);
            const Eigen::Vector3d v_tgt(k.vx, k.vy, k.vz);

            const Eigen::Vector3d p_BN = R_NW * (p_drone - p_tgt);
            const Eigen::Vector3d v_BN = R_NW * (v_drone - v_tgt) - Om_N.cross(p_BN);

            Eigen::Quaterniond q_drone_q(drone_pose_copy.pose.orientation.w,
                                          drone_pose_copy.pose.orientation.x,
                                          drone_pose_copy.pose.orientation.y,
                                          drone_pose_copy.pose.orientation.z);
            q_drone_q.normalize();
            const Eigen::Quaterniond q_NB = last_target_quat_.conjugate() * q_drone_q;

            nav_msgs::msg::Odometry tf_odom;
            tf_odom.header.stamp    = stamp;
            tf_odom.header.frame_id = "target_frame";
            tf_odom.child_frame_id  = "drone_in_target_frame";
            tf_odom.pose.pose.position.x    = p_BN.x();
            tf_odom.pose.pose.position.y    = p_BN.y();
            tf_odom.pose.pose.position.z    = p_BN.z();
            tf_odom.pose.pose.orientation.w = q_NB.w();
            tf_odom.pose.pose.orientation.x = q_NB.x();
            tf_odom.pose.pose.orientation.y = q_NB.y();
            tf_odom.pose.pose.orientation.z = q_NB.z();
            tf_odom.twist.twist.linear.x    = v_BN.x();
            tf_odom.twist.twist.linear.y    = v_BN.y();
            tf_odom.twist.twist.linear.z    = v_BN.z();
            const double ang_scale = (body_relative_angular_unit_ == "rad_s") ? 1.0 : DEG2RAD;
            tf_odom.twist.twist.angular.x = drone_odom_copy.twist.twist.angular.x * ang_scale;
            tf_odom.twist.twist.angular.y = drone_odom_copy.twist.twist.angular.y * ang_scale;
            tf_odom.twist.twist.angular.z = drone_odom_copy.twist.twist.angular.z * ang_scale;
            target_frame_odom_pub_->publish(tf_odom);
        }

        // ── Diagnostics (every 5 s) ──────────────────────────────────────────
        static double last_diag = -5.0;
        if (t - last_diag >= 5.0) {
            last_diag = t;
            RCLCPP_DEBUG(get_logger(),
                "[TargetPublisher] t=%.2f  pos=(%.3f,%.3f,%.3f)  vel=(%.3f,%.3f,%.3f)",
                t, k.px, k.py, k.pz, k.vx, k.vy, k.vz);
        }
    }

    // ── Parameters ─────────────────────────────────────────────────────────
    std::string target_mode_;
    double target_yaw_rate_;
    double publish_hz_;
    std::string frame_id_;
    target_models::CircularTarget circle_model_;  // trajectory defined in circular_target.hpp
    double t0_ = 0.0;  // wall-clock time at startup, for relative time in circle mode
    std::string qualisys_pose_topic_;
    std::string drone_odom_topic_;
    std::string drone_odom_mode_;  // "none" | "shifted_world" | "body_frame" | "target_frame"
    std::string drone_pose_topic_;
    bool debug_body_relative_trace_ = false;
    std::string body_relative_angular_unit_ = "deg_s";
    Eigen::Quaterniond last_target_quat_ = Eigen::Quaterniond::Identity();

    // ── ROS handles ────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          abs_rel_odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          body_rel_odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          target_frame_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr accel_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       drone_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr drone_pose_sub_;
#ifdef HAS_MOCAP4R2_MSGS
    rclcpp::Subscription<mocap4r2_msgs::msg::RigidBodies>::SharedPtr rigid_bodies_sub_;
#endif
    rclcpp::TimerBase::SharedPtr timer_;

    // ── Qualisys pose history (guarded by pose_mutex_) ───────────────────────
    struct PoseObs { double t; Eigen::Vector3d pos; Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); };
    static constexpr std::size_t N_POSE_HISTORY = 8;
    std::mutex pose_mutex_;
    std::deque<PoseObs> pose_history_;

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
