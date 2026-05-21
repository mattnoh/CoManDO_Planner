/// @file target_publisher.cpp
/// @brief Standalone ROS2 node that publishes the target state for the CoManDO planner.
///
/// Target modes:
///
///   "gazebo_circle"  — Synthetic circular pose measurements fed through the
///                      same estimator used by mocap.
///
///   "gazebo_figure8" — Synthetic figure-8 pose measurements fed through the
///                      same estimator used by mocap.
///
///   "mocap"          — Real target pose from a mocap4r2 rigid-bodies
///                      topic, selected by rigid_body_name.
///
/// Publishes on:
///   /target/odom   (nav_msgs/msg/Odometry)
///   /target/accel  (geometry_msgs/msg/AccelStamped)
///   /target/true_odom, /target/true_accel in gazebo_* modes only
///
/// Parameters:
///   target_mode         — "gazebo_circle", "gazebo_figure8", or "mocap"
///                         default: "gazebo_circle"
///   center_x/y/z        — orbit center [m]                 default: 0, 0, 0.2
///   omega               — angular speed [rad/s]            default: 0.4
///   phi0                — initial phase, circle mode [rad] default: 0.0
///   radius              — orbit radius, circle mode [m]    default: 1.0
///   publish_hz          — publish rate [Hz]                default: 100.0
///   frame_id            — header frame                     default: "world"
///   rigid_body_name     — mocap rigid body name            default: "stmini"
///   drone_odom_topic    — drone odom for relative odom     default: "/gogogo/odom"
///   planning_frame      — "world", "shifted", "target_frame", or "body_frame"
///   drone_odom_mode     — legacy alias for planning_frame
///
/// Usage:
///   ros2 run comando_planner target_publisher --ros-args
///       -p target_mode:=gazebo_circle
///   ros2 run comando_planner target_publisher --ros-args
///       -p target_mode:=mocap -p rigid_body_name:=stmini

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#ifdef HAS_MOCAP4R2_MSGS
#include <mocap4r2_msgs/msg/rigid_bodies.hpp>
#endif
#include <Eigen/Geometry>

#include "target/circular_target.hpp"
#include "target/figure8_target.hpp"

#include <cmath>
#include <chrono>
#include <mutex>

namespace {
constexpr double DEG2RAD = M_PI / 180.0;

std::string normalizeTargetMode(const std::string& requested)
{
    if (requested == "gazebo_circle" ||
        requested == "gazebo_figure8" ||
        requested == "mocap") {
        return requested;
    }
    return "gazebo_circle";
}

std::string normalizePlanningFrame(const std::string& requested,
                                   const std::string& legacy_mode)
{
    const std::string raw = !requested.empty() ? requested : legacy_mode;
    if (raw.empty() || raw == "none" || raw == "world") {
        return "world";
    }
    if (raw == "shifted_world" || raw == "shifted") {
        return "shifted";
    }
    if (raw == "target" || raw == "target_frame") {
        return "target_frame";
    }
    if (raw == "body" || raw == "body_frame") {
        return "body_frame";
    }
    return raw;
}
}

class BenchmarkTargetPublisher : public rclcpp::Node
{
public:
    BenchmarkTargetPublisher() : Node("target_publisher")
    {
        // ── Parameters ──────────────────────────────────────────────────────
        const std::string requested_target_mode =
            declare_parameter<std::string>("target_mode", "gazebo_circle");
        target_mode_ = normalizeTargetMode(requested_target_mode);
        target_yaw_rate_ = declare_parameter<double>("target_yaw_rate", 0.3);
        publish_hz_  = declare_parameter<double>("publish_hz", 100.0);
        frame_id_    = declare_parameter<std::string>("frame_id", "world");
        rigid_body_name_  = declare_parameter<std::string>("rigid_body_name", "stmini");
        drone_odom_topic_     = declare_parameter<std::string>("drone_odom_topic", "/cf_1/odom");
        const std::string legacy_drone_odom_mode = declare_parameter<std::string>("drone_odom_mode", "");
        planning_frame_ = normalizePlanningFrame(
            declare_parameter<std::string>("planning_frame", ""),
            legacy_drone_odom_mode);
        // Internal normalized mode:
        //   "world"        — no drone relative output
        //   "shifted"      — /drone/relative_odometry: p_drone-p_tgt, v_drone-v_tgt in world
        //   "target_frame" — /drone/target_frame_odom: p_B^N, v_B^N, q_NB
        //   "body_frame"   — /drone/body_relative_odom: p_T^B, v_rel^B, q_NB
        drone_odom_mode_ = planning_frame_;
        drone_pose_topic_ = declare_parameter<std::string>("drone_pose_topic", "/cf_1/pose");
        debug_body_relative_trace_ = declare_parameter<bool>("debug_body_relative_trace", false);
        body_relative_angular_unit_ = declare_parameter<std::string>(
            "body_relative_input_angular_unit", "deg_s");

        // Filter gains for target pose measurements.
        filter_alpha_   = declare_parameter<double>("filter_alpha",   0.8);
        filter_beta_    = declare_parameter<double>("filter_beta",    0.4);
        filter_gamma_   = declare_parameter<double>("filter_gamma",   0.001);
        filter_alpha_w_ = declare_parameter<double>("filter_alpha_w", 0.7);
        filter_beta_w_  = declare_parameter<double>("filter_beta_w",  0.3);

        if (target_mode_ != requested_target_mode) {
            RCLCPP_WARN(get_logger(),
                "[TargetPublisher] Unknown target_mode='%s'; using gazebo_circle.",
                requested_target_mode.c_str());
        }
        if (drone_odom_mode_ != "world" && drone_odom_mode_ != "shifted" &&
            drone_odom_mode_ != "target_frame" && drone_odom_mode_ != "body_frame") {
            RCLCPP_WARN(get_logger(),
                "[TargetPublisher] Unknown planning_frame='%s'; using world.",
                drone_odom_mode_.c_str());
            drone_odom_mode_ = "world";
            planning_frame_ = "world";
        }

#ifndef HAS_MOCAP4R2_MSGS
        if (target_mode_ == "mocap") {
            RCLCPP_WARN(get_logger(),
                "[TargetPublisher] target_mode=mocap requested, but this build has no "
                "mocap4r2_msgs support. Falling back to gazebo_circle.");
            target_mode_ = "gazebo_circle";
        }
#endif

        t0_ = now().seconds();

        // ── Publishers ───────────────────────────────────────────────────────
        odom_pub_  = create_publisher<nav_msgs::msg::Odometry>("/target/odom", 10);
        accel_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>("/target/accel", 10);
        true_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/target/true_odom", 10);
        true_accel_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>("/target/true_accel", 10);

        if (drone_odom_mode_ == "shifted") {
            abs_rel_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/drone/relative_odometry", 10);
        } else if (drone_odom_mode_ == "body_frame") {
            body_rel_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/drone/body_relative_odom", 10);
        } else if (drone_odom_mode_ == "target_frame") {
            target_frame_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/drone/target_frame_odom", 10);
        }
        if (drone_odom_mode_ != "world") {
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

        pos_filter_.configure(filter_alpha_, filter_beta_, filter_gamma_);
        angvel_filter_.configure(filter_alpha_w_, filter_beta_w_);

        // ── Mocap rigid bodies subscriber (mocap mode only) ─────────────────
#ifdef HAS_MOCAP4R2_MSGS
        if (target_mode_ == "mocap") {
            rigid_bodies_sub_ = create_subscription<mocap4r2_msgs::msg::RigidBodies>(
                "/rigid_bodies", 10,
                [this](const mocap4r2_msgs::msg::RigidBodies::SharedPtr msg) {
                    RCLCPP_INFO_ONCE(get_logger(),
                        "[TargetPublisher] /rigid_bodies callback firing, %zu bodies",
                        msg->rigidbodies.size());
                    for (const auto & rb : msg->rigidbodies) {
                        RCLCPP_INFO_ONCE(get_logger(),
                            "[TargetPublisher] body seen: '%s'", rb.rigid_body_name.c_str());
                        if (rb.rigid_body_name != rigid_body_name_) continue;
                        const double t = rclcpp::Time(msg->header.stamp).seconds();
                        const Eigen::Vector3d p_meas(
                            rb.pose.position.x, rb.pose.position.y, rb.pose.position.z);
                        Eigen::Quaterniond q_meas(
                            rb.pose.orientation.w, rb.pose.orientation.x,
                            rb.pose.orientation.y, rb.pose.orientation.z);
                        q_meas.normalize();
                        ingestTargetPoseMeasurement(p_meas, q_meas, t);
                        break;
                    }
                });
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] mode=mocap  body_name=%s  abg(alpha=%.2f beta=%.2f gamma=%.4f)  ab(alpha=%.2f beta=%.2f)",
                rigid_body_name_.c_str(),
                filter_alpha_, filter_beta_, filter_gamma_,
                filter_alpha_w_, filter_beta_w_);
        } else {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] mode=%s synthetic pose measurements  circle_center=(%.2f,%.2f,%.2f)  "
                "circle_R=%.2f  circle_omega=%.2f rad/s  %.0f Hz",
                target_mode_.c_str(),
                circle_model_.center.x(), circle_model_.center.y(), circle_model_.center.z(),
                circle_model_.R, circle_model_.omega, publish_hz_);
        }
#else
        RCLCPP_INFO(get_logger(),
            "[TargetPublisher] mode=%s synthetic pose measurements  circle_center=(%.2f,%.2f,%.2f)  "
            "circle_R=%.2f  circle_omega=%.2f rad/s  %.0f Hz",
            target_mode_.c_str(),
            circle_model_.center.x(), circle_model_.center.y(), circle_model_.center.z(),
            circle_model_.R, circle_model_.omega, publish_hz_);
#endif

        if (drone_odom_mode_ == "shifted") {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] planning_frame=shifted -> /drone/relative_odometry (drone-target in world frame)");
        } else if (drone_odom_mode_ == "body_frame") {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] planning_frame=body_frame -> /drone/body_relative_odom (target in drone body frame)");
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] Angular unit='%s' (set body_relative_input_angular_unit:=rad_s if upstream publishes rad/s)",
                body_relative_angular_unit_.c_str());
        } else if (drone_odom_mode_ == "target_frame") {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] planning_frame=target_frame -> /drone/target_frame_odom (drone in target N frame)");
        } else {
            RCLCPP_INFO(get_logger(), "[TargetPublisher] planning_frame=world - no relative odometry output");
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

    // Alpha-beta-gamma tracking filter: measurement = position, outputs p, v, a.
    struct ABGFilter3D {
        double alpha = 0.8, beta = 0.4, gamma = 0.001;
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
        bool initialized = false;
        double last_t = 0.0;

        void configure(double a_, double b_, double g_) { alpha = a_; beta = b_; gamma = g_; }
        void reset() { initialized = false; v.setZero(); a.setZero(); }

        void update(const Eigen::Vector3d& meas, double t) {
            if (!initialized) {
                p = meas; v.setZero(); a.setZero();
                initialized = true; last_t = t; return;
            }
            const double dt = t - last_t;
            if (dt < 1e-9) return;
            last_t = t;
            const Eigen::Vector3d p_pred = p + v * dt + 0.5 * a * dt * dt;
            const Eigen::Vector3d v_pred = v + a * dt;
            const Eigen::Vector3d r = meas - p_pred;
            p = p_pred + alpha * r;
            v = v_pred + (beta / dt) * r;
            a = a + (2.0 * gamma / (dt * dt)) * r;
        }
    };

    // Alpha-beta tracking filter: measurement = velocity (raw omega), outputs x and dx (ang accel).
    struct ABFilter3D {
        double alpha = 0.7, beta = 0.3;
        Eigen::Vector3d x = Eigen::Vector3d::Zero();
        Eigen::Vector3d dx = Eigen::Vector3d::Zero();
        bool initialized = false;
        double last_t = 0.0;

        void configure(double a_, double b_) { alpha = a_; beta = b_; }
        void reset() { initialized = false; dx.setZero(); }

        void update(const Eigen::Vector3d& meas, double t) {
            if (!initialized) {
                x = meas; dx.setZero();
                initialized = true; last_t = t; return;
            }
            const double dt = t - last_t;
            if (dt < 1e-9) return;
            last_t = t;
            const Eigen::Vector3d x_pred = x + dx * dt;
            const Eigen::Vector3d r = meas - x_pred;
            x = x_pred + alpha * r;
            dx = dx + (beta / dt) * r;
        }
    };

    // ── Timer callback ───────────────────────────────────────────────────────
    void timerCallback()
    {
        const rclcpp::Time stamp = now();
        const double t = stamp.seconds();

        if (target_mode_ == "gazebo_circle" || target_mode_ == "gazebo_figure8") {
            const Kinematics truth = syntheticTargetTruth(t);
            publishSyntheticTruth(truth, stamp);
            ingestTargetPoseMeasurement(
                Eigen::Vector3d(truth.px, truth.py, truth.pz), truth.q, t);
        }

        Kinematics k{};
        if (!readEstimatedTarget(k)) {
            if (target_mode_ == "mocap") {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "[TargetPublisher] Waiting for '%s' in /rigid_bodies ...",
                    rigid_body_name_.c_str());
            }
            return;
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

        if (drone_odom_mode_ != "world") {
            if (!have_drone_odom) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "[TargetPublisher] Waiting for drone odom on '%s' — "
                    "relative odom output will not publish until data arrives.",
                    drone_odom_topic_.c_str());
            } else if (!have_drone_pose &&
                       (drone_odom_mode_ == "body_frame" || drone_odom_mode_ == "target_frame")) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "[TargetPublisher] Waiting for drone pose on '%s' — "
                    "relative odom output will not publish until data arrives.",
                    drone_pose_topic_.c_str());
            }
        }

        if (drone_odom_mode_ == "shifted" && abs_rel_odom_pub_ && have_drone_odom) {
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

    bool readEstimatedTarget(Kinematics& k) {
        std::lock_guard<std::mutex> lk(pose_mutex_);
        if (!pos_filter_.initialized) {
            return false;
        }
        k.px = pos_filter_.p.x();
        k.py = pos_filter_.p.y();
        k.pz = pos_filter_.p.z();
        k.vx = pos_filter_.v.x();
        k.vy = pos_filter_.v.y();
        k.vz = pos_filter_.v.z();
        k.ax = pos_filter_.a.x();
        k.ay = pos_filter_.a.y();
        k.az = pos_filter_.a.z();
        k.wx = angvel_filter_.x.x();
        k.wy = angvel_filter_.x.y();
        k.wz = angvel_filter_.x.z();
        k.alfx = angvel_filter_.dx.x();
        k.alfy = angvel_filter_.dx.y();
        k.alfz = angvel_filter_.dx.z();
        k.q = target_q_latest_;
        return true;
    }

    Kinematics syntheticTargetTruth(double t) const
    {
        const double t_rel = t - t0_;
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
        if (target_mode_ == "gazebo_figure8") {
            p = figure8_model_.pos(t_rel);
            v = figure8_model_.vel(t_rel);
            a = figure8_model_.accel(t_rel);
        } else {
            p = circle_model_.pos(t_rel);
            v = circle_model_.vel(t_rel);
            a = circle_model_.accel(t_rel);
        }
        const double psi = target_yaw_rate_ * t_rel;
        Kinematics k{};
        k.px = p.x(); k.py = p.y(); k.pz = p.z();
        k.vx = v.x(); k.vy = v.y(); k.vz = v.z();
        k.ax = a.x(); k.ay = a.y(); k.az = a.z();
        k.wx = 0.0; k.wy = 0.0; k.wz = target_yaw_rate_;
        k.alfx = 0.0; k.alfy = 0.0; k.alfz = 0.0;
        k.q = Eigen::Quaterniond(std::cos(psi / 2.0), 0.0, 0.0, std::sin(psi / 2.0));
        return k;
    }

    void publishSyntheticTruth(const Kinematics& k, const rclcpp::Time& stamp)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = frame_id_;
        odom.child_frame_id = "target_true";
        odom.pose.pose.position.x = k.px;
        odom.pose.pose.position.y = k.py;
        odom.pose.pose.position.z = k.pz;
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
        true_odom_pub_->publish(odom);

        geometry_msgs::msg::AccelStamped accel;
        accel.header.stamp = stamp;
        accel.header.frame_id = frame_id_;
        accel.accel.linear.x = k.ax;
        accel.accel.linear.y = k.ay;
        accel.accel.linear.z = k.az;
        accel.accel.angular.x = k.alfx;
        accel.accel.angular.y = k.alfy;
        accel.accel.angular.z = k.alfz;
        true_accel_pub_->publish(accel);
    }

    void ingestTargetPoseMeasurement(const Eigen::Vector3d& p_meas,
                                     const Eigen::Quaterniond& q_meas,
                                     double t)
    {
        std::lock_guard<std::mutex> lk(pose_mutex_);
        pos_filter_.update(p_meas, t);

        Eigen::Vector3d omega_raw = Eigen::Vector3d::Zero();
        if (target_q_initialized_) {
            const double dt = t - target_t_prev_;
            if (dt > 1e-9) {
                Eigen::Quaterniond dq = q_meas * target_q_prev_.conjugate();
                if (dq.w() < 0.0) dq.coeffs() *= -1.0;  // shortest path
                const Eigen::AngleAxisd aa(dq);
                omega_raw = aa.axis() * aa.angle() / dt;
                angvel_filter_.update(omega_raw, t);
            }
        } else {
            target_q_initialized_ = true;
        }
        target_q_prev_   = q_meas;
        target_q_latest_ = q_meas;
        target_t_prev_   = t;
    }

    // ── Parameters ─────────────────────────────────────────────────────────
    std::string target_mode_;
    double target_yaw_rate_;
    double publish_hz_;
    std::string frame_id_;
    target_models::CircularTarget circle_model_;  // circle model defaults live in circular_target.hpp
    target_models::Figure8Target figure8_model_;  // figure-8 model defaults live in figure8_target.hpp
    double t0_ = 0.0;  // wall-clock time at startup, for relative time in model source
    std::string rigid_body_name_;
    std::string drone_odom_topic_;
    std::string planning_frame_;
    std::string drone_odom_mode_;  // "world" | "shifted" | "body_frame" | "target_frame"
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
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          true_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr true_accel_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       drone_odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr drone_pose_sub_;
#ifdef HAS_MOCAP4R2_MSGS
    rclcpp::Subscription<mocap4r2_msgs::msg::RigidBodies>::SharedPtr rigid_bodies_sub_;
#endif
    rclcpp::TimerBase::SharedPtr timer_;

    // ── Target estimator state (guarded by pose_mutex_) ─────────────────────
    std::mutex pose_mutex_;
    ABGFilter3D pos_filter_;
    ABFilter3D  angvel_filter_;
    Eigen::Quaterniond target_q_prev_   = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond target_q_latest_ = Eigen::Quaterniond::Identity();
    double target_t_prev_ = 0.0;
    bool   target_q_initialized_ = false;

    // Filter gains
    double filter_alpha_   = 0.8;
    double filter_beta_    = 0.4;
    double filter_gamma_   = 0.001;
    double filter_alpha_w_ = 0.7;
    double filter_beta_w_  = 0.3;

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
