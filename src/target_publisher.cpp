/// @file target_publisher.cpp
/// @brief Standalone ROS2 benchmark target publisher.
///
/// Publishes on:
///   /target/odom   (nav_msgs/msg/Odometry)   — position + velocity
///   /target/accel  (geometry_msgs/msg/AccelStamped) — live target acceleration
///   /target/predicted_accel (trajectory_msgs/msg/MultiDOFJointTrajectory)
///       — future acceleration samples used by tracking_circle_target.
///
/// Integration note:
///   tracking_circle_target consumes future acceleration from
///   /target/predicted_accel inside the OCP. The planner uses /target/odom
///   as the solve-start anchor to reconstruct world-frame target motion for
///   command publishing and logging.
///
/// Parameters (all declare_parameter with defaults):
///   target_mode                    — target motion type: circle | figure8
///   center_x, center_y, center_z   — motion center [m]
///   radius                         — circle radius [m] (circle mode)
///   amp_x, amp_y                   — Gerono figure-8 amplitudes [m] (figure8)
///   omega                          — angular speed [rad/s]
///   phi0                           — initial phase [rad]    default: 0.0
///   publish_hz                     — publish rate [Hz]      default: 100.0
///   frame_id                       — header frame           default: "world"
///   drone_odom_topic               — drone odom input       default: "/cf_1/odom"
///   relative_odom_topic            — relative odom output   default: "/drone/relative_odometry"
///   enable_relative_odom           — publish relative odom  default: true
///
/// Usage:
///   ros2 run comando_planner target_publisher --ros-args -p target_mode:=circle
///   ros2 run comando_planner target_publisher --ros-args -p target_mode:=figure8 -p amp_x:=1.2 -p amp_y:=0.8
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
#include "target/figure8_target.hpp"

#include <cmath>
#include <chrono>
#include <mutex>

class BenchmarkTargetPublisher : public rclcpp::Node
{
public:
    BenchmarkTargetPublisher() : Node("circular_target_publisher")
    {
        // ── Parameters ──────────────────────────────────────────────────────
        target_mode_ = declare_parameter<std::string>("target_mode", "circle");
        center_x_   = declare_parameter<double>("center_x",   0.0);
        center_y_   = declare_parameter<double>("center_y",   0.0);
        center_z_   = declare_parameter<double>("center_z",   0.2);
        radius_     = declare_parameter<double>("radius",     1.0);
        amp_x_      = declare_parameter<double>("amp_x",      1.0);
        amp_y_      = declare_parameter<double>("amp_y",      1.0);
        omega_      = declare_parameter<double>("omega",      0.4);
        phi0_       = declare_parameter<double>("phi0",       0.0);
        publish_hz_ = declare_parameter<double>("publish_hz", 100.0);
        frame_id_   = declare_parameter<std::string>("frame_id", "world");
        drone_odom_topic_ = declare_parameter<std::string>("drone_odom_topic", "/cf_1/odom");
        relative_odom_topic_ = declare_parameter<std::string>("relative_odom_topic", "/drone/relative_odometry");
        enable_relative_odom_ = declare_parameter<bool>("enable_relative_odom", true);

        if (target_mode_ != "circle" && target_mode_ != "figure8") {
            RCLCPP_WARN(get_logger(),
                "Unknown target_mode='%s'; falling back to 'circle'", target_mode_.c_str());
            target_mode_ = "circle";
        }

        // Record wall-clock start so visual phase matches requested phi0 at node startup.
        phi0_ = phi0_ - omega_ * now().seconds();

        // ── Publishers ───────────────────────────────────────────────────────
        odom_pub_  = create_publisher<nav_msgs::msg::Odometry>(
            "/target/odom",  10);
        accel_pub_ = create_publisher<geometry_msgs::msg::AccelStamped>(
            "/target/accel", 10);
        traj_pub_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
            "/target/predicted_accel", 10);

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
            std::bind(&BenchmarkTargetPublisher::timerCallback, this));

        RCLCPP_INFO(get_logger(),
            "[TargetPublisher] mode=%s center=(%.2f,%.2f,%.2f) R=%.2f Ax=%.2f Ay=%.2f omega=%.2f rad/s %.0f Hz",
            target_mode_.c_str(),
            center_x_, center_y_, center_z_, radius_, amp_x_, amp_y_, omega_, publish_hz_);
        if (enable_relative_odom_) {
            RCLCPP_INFO(get_logger(),
                "[TargetPublisher] Relative odom enabled: in=%s out=%s",
                drone_odom_topic_.c_str(), relative_odom_topic_.c_str());
        }
    }

private:
    struct TargetKinematics {
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        Eigen::Vector3d v = Eigen::Vector3d::Zero();
        Eigen::Vector3d a = Eigen::Vector3d::Zero();
    };

    TargetKinematics sampleTarget(double t) const {
        TargetKinematics out;

        if (target_mode_ == "figure8") {
            target_models::Figure8Target tgt;
            tgt.center = {center_x_, center_y_, center_z_};
            tgt.amp_x = amp_x_;
            tgt.amp_y = amp_y_;
            tgt.omega = omega_;
            tgt.phi0 = phi0_;
            out.p = tgt.pos(t);
            out.v = tgt.vel(t);
            out.a = tgt.accel(t);
            return out;
        }

        target_models::CircularTarget tgt;
        tgt.center = {center_x_, center_y_, center_z_};
        tgt.R = radius_;
        tgt.omega = omega_;
        tgt.phi0 = phi0_;
        out.p = tgt.pos(t);
        out.v = tgt.vel(t);
        out.a = tgt.accel(t);
        return out;
    }

    void timerCallback()
    {
        const rclcpp::Time stamp = now();
        const double t = stamp.seconds();

        // ── Kinematics ───────────────────────────────────────────────────────
        const TargetKinematics k = sampleTarget(t);
        const auto px = k.p.x();
        const auto py = k.p.y();
        const auto pz = k.p.z();

        const auto vx = k.v.x();
        const auto vy = k.v.y();
        const auto vz = k.v.z();

        const auto ax = k.a.x();
        const auto ay = k.a.y();
        const auto az = k.a.z();

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
        // Angular acceleration is unused for this planar benchmark target.
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

            const TargetKinematics kd = sampleTarget(td);
            const auto p_tgt = kd.p;
            const auto v_tgt = kd.v;

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
                TargetKinematics kp = sampleTarget(point_t);

                geometry_msgs::msg::Transform trans;
                trans.translation.x = kp.p.x();
                trans.translation.y = kp.p.y();
                trans.translation.z = kp.p.z();
                trans.rotation.w = 1.0;
                trans.rotation.x = 0.0;
                trans.rotation.y = 0.0;
                trans.rotation.z = 0.0;
                pt.transforms.push_back(trans);

                geometry_msgs::msg::Twist vel;
                vel.linear.x = kp.v.x();
                vel.linear.y = kp.v.y();
                vel.linear.z = kp.v.z();
                pt.velocities.push_back(vel);

                geometry_msgs::msg::Twist acc;
                acc.linear.x = kp.a.x();
                acc.linear.y = kp.a.y();
                acc.linear.z = kp.a.z();
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
                "[TargetPublisher] mode=%s t=%.2f pos=(%.3f,%.3f,%.3f) "
                "vel=(%.3f,%.3f)  accel=(%.3f,%.3f)",
                target_mode_.c_str(), t, px, py, pz, vx, vy, ax, ay);
        }
    }

    // ── Parameters ─────────────────────────────────────────────────────────
    std::string target_mode_;
    double center_x_, center_y_, center_z_;
    double radius_, amp_x_, amp_y_, omega_, phi0_, publish_hz_;
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
    rclcpp::spin(std::make_shared<BenchmarkTargetPublisher>());
    rclcpp::shutdown();
    return 0;
}