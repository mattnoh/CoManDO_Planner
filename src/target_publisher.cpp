/// @file target_publisher.cpp  (ROS1 Noetic branch)
/// @brief Publishes synthetic target state for the CoManDO planner.
///
/// Source: "model" (circle or figure8) — fully synthetic, no external input.
///
/// Publishes:
///   /target/odom   (nav_msgs/Odometry)
///   /target/accel  (geometry_msgs/AccelStamped)
///
/// Optional relative odometry outputs (controlled by planning_frame param):
///   "shifted"      → /drone/relative_odometry
///   "body_frame"   → /drone/body_relative_odom
///   "target_frame" → /drone/target_frame_odom

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/AccelStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Geometry>
#include <deque>
#include <mutex>
#include <cmath>

#include "target/circular_target.hpp"
#include "target/figure8_target.hpp"

namespace {
constexpr double DEG2RAD = M_PI / 180.0;

std::string normalizePlanningFrame(const std::string& raw) {
    if (raw.empty() || raw == "none" || raw == "world") return "world";
    if (raw == "shifted_world") return "shifted";
    return raw;
}
}

class TargetPublisherNode {
public:
    explicit TargetPublisherNode(ros::NodeHandle& nh) : nh_(nh) {
        std::string traj, frame_raw, drone_odom_mode_legacy;
        nh_.param<std::string>("target_trajectory", traj, "circle");
        nh_.param<double>     ("target_yaw_rate",   target_yaw_rate_, 0.3);
        nh_.param<double>     ("publish_hz",         publish_hz_,      100.0);
        nh_.param<std::string>("frame_id",           frame_id_,        "world");
        nh_.param<std::string>("drone_odom_topic",   drone_odom_topic_, "/drone/odom");
        nh_.param<std::string>("drone_pose_topic",   drone_pose_topic_, "/drone/pose");
        nh_.param<std::string>("planning_frame",     frame_raw,        "world");
        nh_.param<std::string>("drone_odom_mode",    drone_odom_mode_legacy, "");
        nh_.param<bool>       ("debug_body_relative_trace", debug_body_relative_trace_, false);
        nh_.param<std::string>("body_relative_input_angular_unit", body_relative_angular_unit_, "deg_s");

        if (!frame_raw.empty())
            drone_odom_mode_ = normalizePlanningFrame(frame_raw);
        else if (!drone_odom_mode_legacy.empty())
            drone_odom_mode_ = normalizePlanningFrame(drone_odom_mode_legacy);
        else
            drone_odom_mode_ = "world";

        target_trajectory_ = (traj == "figure8") ? "figure8" : "circle";

        t0_ = ros::Time::now().toSec();

        odom_pub_  = nh_.advertise<nav_msgs::Odometry>("/target/odom",  10);
        accel_pub_ = nh_.advertise<geometry_msgs::AccelStamped>("/target/accel", 10);

        if (drone_odom_mode_ == "shifted") {
            abs_rel_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/drone/relative_odometry", 10);
        } else if (drone_odom_mode_ == "body_frame") {
            body_rel_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/drone/body_relative_odom", 10);
        } else if (drone_odom_mode_ == "target_frame") {
            target_frame_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/drone/target_frame_odom", 10);
        }

        if (drone_odom_mode_ != "world") {
            drone_odom_sub_ = nh_.subscribe<nav_msgs::Odometry>(
                drone_odom_topic_, 10,
                [this](const nav_msgs::Odometry::ConstPtr& msg) {
                    std::lock_guard<std::mutex> lk(drone_mutex_);
                    last_drone_odom_ = *msg; has_drone_odom_ = true;
                });
        }
        if (drone_odom_mode_ == "body_frame" || drone_odom_mode_ == "target_frame") {
            drone_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(
                drone_pose_topic_, 10,
                [this](const geometry_msgs::PoseStamped::ConstPtr& msg) {
                    std::lock_guard<std::mutex> lk(drone_mutex_);
                    last_drone_pose_ = *msg; has_drone_pose_ = true;
                });
        }

        ROS_INFO("[TargetPublisher] trajectory=%s  planning_frame=%s  %.0f Hz",
            target_trajectory_.c_str(), drone_odom_mode_.c_str(), publish_hz_);

        timer_ = nh_.createTimer(ros::Duration(1.0 / publish_hz_),
                                 &TargetPublisherNode::timerCallback, this);
    }

private:
    void timerCallback(const ros::TimerEvent&) {
        const ros::Time stamp = ros::Time::now();
        const double t_rel = stamp.toSec() - t0_;

        Eigen::Vector3d p, v, a;
        if (target_trajectory_ == "figure8") {
            p = figure8_model_.pos(t_rel);
            v = figure8_model_.vel(t_rel);
            a = figure8_model_.accel(t_rel);
        } else {
            p = circle_model_.pos(t_rel);
            v = circle_model_.vel(t_rel);
            a = circle_model_.accel(t_rel);
        }
        const double psi = target_yaw_rate_ * t_rel;
        const Eigen::Quaterniond q_tgt(std::cos(psi/2.0), 0.0, 0.0, std::sin(psi/2.0));
        last_target_quat_ = q_tgt;

        // Odometry
        nav_msgs::Odometry odom;
        odom.header.stamp    = stamp;
        odom.header.frame_id = frame_id_;
        odom.child_frame_id  = "target";
        odom.pose.pose.position.x = p.x(); odom.pose.pose.position.y = p.y(); odom.pose.pose.position.z = p.z();
        odom.pose.pose.orientation.w = q_tgt.w(); odom.pose.pose.orientation.x = q_tgt.x();
        odom.pose.pose.orientation.y = q_tgt.y(); odom.pose.pose.orientation.z = q_tgt.z();
        odom.twist.twist.linear.x = v.x(); odom.twist.twist.linear.y = v.y(); odom.twist.twist.linear.z = v.z();
        odom.twist.twist.angular.z = target_yaw_rate_;
        odom_pub_.publish(odom);

        // AccelStamped
        geometry_msgs::AccelStamped accel;
        accel.header.stamp    = stamp;
        accel.header.frame_id = frame_id_;
        accel.accel.linear.x = a.x(); accel.accel.linear.y = a.y(); accel.accel.linear.z = a.z();
        accel_pub_.publish(accel);

        // Relative odometry outputs
        nav_msgs::Odometry drone_odom_copy;
        geometry_msgs::PoseStamped drone_pose_copy;
        bool have_odom = false, have_pose = false;
        {
            std::lock_guard<std::mutex> lk(drone_mutex_);
            have_odom = has_drone_odom_; have_pose = has_drone_pose_;
            if (have_odom) drone_odom_copy = last_drone_odom_;
            if (have_pose) drone_pose_copy = last_drone_pose_;
        }

        if (drone_odom_mode_ == "shifted" && have_odom) {
            nav_msgs::Odometry rel = drone_odom_copy;
            rel.header.stamp = stamp; rel.header.frame_id = frame_id_; rel.child_frame_id = "drone_absolute_relative";
            rel.pose.pose.position.x -= p.x(); rel.pose.pose.position.y -= p.y(); rel.pose.pose.position.z -= p.z();
            rel.twist.twist.linear.x -= v.x(); rel.twist.twist.linear.y -= v.y(); rel.twist.twist.linear.z -= v.z();
            abs_rel_odom_pub_.publish(rel);
        }

        if (drone_odom_mode_ == "body_frame" && have_odom && have_pose) {
            const Eigen::Vector3d tgt_w(p.x(), p.y(), p.z());
            const Eigen::Vector3d drn_w(drone_pose_copy.pose.position.x, drone_pose_copy.pose.position.y, drone_pose_copy.pose.position.z);
            const Eigen::Vector3d drn_v(drone_odom_copy.twist.twist.linear.x, drone_odom_copy.twist.twist.linear.y, drone_odom_copy.twist.twist.linear.z);
            Eigen::Quaterniond q(drone_pose_copy.pose.orientation.w, drone_pose_copy.pose.orientation.x,
                                 drone_pose_copy.pose.orientation.y, drone_pose_copy.pose.orientation.z);
            q.normalize();
            const Eigen::Matrix3d R = q.toRotationMatrix();
            const Eigen::Vector3d p_t_body = R.transpose() * (tgt_w - drn_w);
            const Eigen::Vector3d v_d_body = R.transpose() * (drn_v - v);
            const Eigen::Quaterniond q_NB = last_target_quat_.conjugate() * q;

            nav_msgs::Odometry body_rel;
            body_rel.header.stamp = stamp; body_rel.header.frame_id = "drone_body"; body_rel.child_frame_id = "body_relative";
            body_rel.pose.pose.position.x = p_t_body.x(); body_rel.pose.pose.position.y = p_t_body.y(); body_rel.pose.pose.position.z = p_t_body.z();
            body_rel.pose.pose.orientation.w = q_NB.w(); body_rel.pose.pose.orientation.x = q_NB.x();
            body_rel.pose.pose.orientation.y = q_NB.y(); body_rel.pose.pose.orientation.z = q_NB.z();
            body_rel.twist.twist.linear.x = v_d_body.x(); body_rel.twist.twist.linear.y = v_d_body.y(); body_rel.twist.twist.linear.z = v_d_body.z();
            const double ang_scale = (body_relative_angular_unit_ == "rad_s") ? 1.0 : DEG2RAD;
            body_rel.twist.twist.angular.x = drone_odom_copy.twist.twist.angular.x * ang_scale;
            body_rel.twist.twist.angular.y = drone_odom_copy.twist.twist.angular.y * ang_scale;
            body_rel.twist.twist.angular.z = drone_odom_copy.twist.twist.angular.z * ang_scale;
            body_rel_odom_pub_.publish(body_rel);

            if (debug_body_relative_trace_) {
                ROS_INFO_THROTTLE(1.0, "[BodyRelTrace] pT=(%.3f,%.3f,%.3f) vdb=(%.3f,%.3f,%.3f)",
                    p_t_body.x(), p_t_body.y(), p_t_body.z(),
                    v_d_body.x(), v_d_body.y(), v_d_body.z());
            }
        }

        if (drone_odom_mode_ == "target_frame" && have_odom && have_pose) {
            const Eigen::Matrix3d R_NW = last_target_quat_.conjugate().toRotationMatrix();
            const Eigen::Vector3d Om_N = R_NW * Eigen::Vector3d(0, 0, target_yaw_rate_);
            const Eigen::Vector3d p_drone(drone_pose_copy.pose.position.x, drone_pose_copy.pose.position.y, drone_pose_copy.pose.position.z);
            const Eigen::Vector3d v_drone(drone_odom_copy.twist.twist.linear.x, drone_odom_copy.twist.twist.linear.y, drone_odom_copy.twist.twist.linear.z);
            const Eigen::Vector3d p_BN = R_NW * (p_drone - p);
            const Eigen::Vector3d v_BN = R_NW * (v_drone - v) - Om_N.cross(p_BN);
            Eigen::Quaterniond q_drn(drone_pose_copy.pose.orientation.w, drone_pose_copy.pose.orientation.x,
                                     drone_pose_copy.pose.orientation.y, drone_pose_copy.pose.orientation.z);
            q_drn.normalize();
            const Eigen::Quaterniond q_NB = last_target_quat_.conjugate() * q_drn;

            nav_msgs::Odometry tf_odom;
            tf_odom.header.stamp = stamp; tf_odom.header.frame_id = "target_frame"; tf_odom.child_frame_id = "drone_in_target_frame";
            tf_odom.pose.pose.position.x = p_BN.x(); tf_odom.pose.pose.position.y = p_BN.y(); tf_odom.pose.pose.position.z = p_BN.z();
            tf_odom.pose.pose.orientation.w = q_NB.w(); tf_odom.pose.pose.orientation.x = q_NB.x();
            tf_odom.pose.pose.orientation.y = q_NB.y(); tf_odom.pose.pose.orientation.z = q_NB.z();
            tf_odom.twist.twist.linear.x = v_BN.x(); tf_odom.twist.twist.linear.y = v_BN.y(); tf_odom.twist.twist.linear.z = v_BN.z();
            const double ang_scale = (body_relative_angular_unit_ == "rad_s") ? 1.0 : DEG2RAD;
            tf_odom.twist.twist.angular.x = drone_odom_copy.twist.twist.angular.x * ang_scale;
            tf_odom.twist.twist.angular.y = drone_odom_copy.twist.twist.angular.y * ang_scale;
            tf_odom.twist.twist.angular.z = drone_odom_copy.twist.twist.angular.z * ang_scale;
            target_frame_odom_pub_.publish(tf_odom);
        }
    }

    ros::NodeHandle& nh_;

    std::string target_trajectory_ = "circle";
    double target_yaw_rate_ = 0.3;
    double publish_hz_ = 100.0;
    std::string frame_id_ = "world";
    std::string drone_odom_topic_;
    std::string drone_pose_topic_;
    std::string drone_odom_mode_ = "world";
    bool debug_body_relative_trace_ = false;
    std::string body_relative_angular_unit_ = "deg_s";
    double t0_ = 0.0;
    Eigen::Quaterniond last_target_quat_ = Eigen::Quaterniond::Identity();

    target_models::CircularTarget circle_model_;
    target_models::Figure8Target  figure8_model_;

    ros::Publisher  odom_pub_, accel_pub_;
    ros::Publisher  abs_rel_odom_pub_, body_rel_odom_pub_, target_frame_odom_pub_;
    ros::Subscriber drone_odom_sub_, drone_pose_sub_;
    ros::Timer      timer_;

    std::mutex drone_mutex_;
    nav_msgs::Odometry last_drone_odom_;
    geometry_msgs::PoseStamped last_drone_pose_;
    bool has_drone_odom_ = false;
    bool has_drone_pose_ = false;
};

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "target_publisher");
    ros::NodeHandle nh("~");
    TargetPublisherNode node(nh);
    ros::spin();
    return 0;
}
