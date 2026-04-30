/// @file planner_core/types.hpp
/// @brief ROS-free planner data contracts for future ROS1/ROS2 wrappers.
#pragma once

#include <Eigen/Dense>

#include <chrono>
#include <string>
#include <vector>

namespace planner_core {

enum class Platform {
    Crazyflie,
    Mavros,
};

enum class CommandKind {
    None,
    FullState,
    Hover,
    BodyRate,
};

struct PlannerInputState {
    Eigen::VectorXd state;
    std::string frame_label = "absolute";
    std::chrono::steady_clock::time_point stamp{};
};

struct TargetInputState {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector4d orientation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_acceleration = Eigen::Vector3d::Zero();
    bool valid = false;
};

struct PlannerCommand {
    CommandKind kind = CommandKind::None;
    Eigen::VectorXd state;
    Eigen::VectorXd control;
    Eigen::Vector4d hover = Eigen::Vector4d::Zero();  // [vx_body, vy_body, z_world, yaw_rate]
};

struct PlannerPath {
    std::vector<Eigen::VectorXd> states;
    std::string frame_id = "world";
};

struct PlannerEvent {
    std::string level = "info";
    std::string message;
    double value = 0.0;
};

}  // namespace planner_core
