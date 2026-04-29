#pragma once

#include <Eigen/Dense>

/// Snapshot of target state taken at solve time.
/// Passed to transform_state, validate_target, post_process_result callbacks.
struct TargetSnapshot {
    bool valid = false;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();  // Ω_N in world frame
    Eigen::Vector3d angular_acceleration = Eigen::Vector3d::Zero(); // β_N in world frame
    Eigen::Vector4d orientation = Eigen::Vector4d(1,0,0,0);      // target quat [qw,qx,qy,qz]
    double odom_stamp_sec = 0.0;
    double accel_stamp_sec = 0.0;
};
