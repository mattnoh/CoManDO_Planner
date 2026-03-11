/// @file frame_conv.hpp
/// @brief NED ↔ ENU / FRD ↔ FLU frame conversion helpers.
///
/// Originally embedded in planner_node.cpp — extracted so both the PX4 bridge
/// and any future bridge can use them without duplication.
///
/// Conventions:
///   ENU = East-North-Up   (ROS / Crazyflie default)
///   NED = North-East-Down (PX4 / uORB default)
///   FLU = Forward-Left-Up  (ROS body frame)
///   FRD = Forward-Right-Down (PX4 body frame)

#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace frame_conv {

// ── Position / velocity ───────────────────────────────────────────────────────

inline Eigen::Vector3d ned_to_enu(const Eigen::Vector3d& v_ned) {
    return { v_ned.y(), v_ned.x(), -v_ned.z() };
}

inline Eigen::Vector3d enu_to_ned(const Eigen::Vector3d& v_enu) {
    return { v_enu.y(), v_enu.x(), -v_enu.z() };
}

// ── Quaternion ────────────────────────────────────────────────────────────────

/// PX4 VehicleOdometry gives q = [w, x, y, z] in NED/FRD.
inline Eigen::Quaterniond quat_ned_to_enu(double w, double x, double y, double z) {
    return Eigen::Quaterniond(w, y, x, -z).normalized();
}

inline Eigen::Quaterniond quat_enu_to_ned(const Eigen::Quaterniond& q_enu) {
    return Eigen::Quaterniond(q_enu.w(), q_enu.y(), q_enu.x(), -q_enu.z()).normalized();
}

// ── Angular velocity ──────────────────────────────────────────────────────────

inline Eigen::Vector3d omega_frd_to_flu(const Eigen::Vector3d& w_frd) {
    return { w_frd.x(), -w_frd.y(), -w_frd.z() };
}

inline Eigen::Vector3d omega_flu_to_frd(const Eigen::Vector3d& w_flu) {
    return { w_flu.x(), -w_flu.y(), -w_flu.z() };
}

// ── Yaw extraction (ENU quaternion → NED yaw for PX4 TrajectorySetpoint) ─────

inline float yaw_ned_from_enu_quat(const Eigen::Quaterniond& q_enu) {
    Eigen::Quaterniond q_ned = quat_enu_to_ned(q_enu);
    double yaw = std::atan2(
        2.0 * (q_ned.w() * q_ned.z() + q_ned.x() * q_ned.y()),
        1.0 - 2.0 * (q_ned.y() * q_ned.y() + q_ned.z() * q_ned.z()));
    return static_cast<float>(yaw);
}

} // namespace frame_conv