/// @file planner_core/types.hpp
/// @brief ROS-free planner data contracts for future ROS1/ROS2 wrappers.
#pragma once

#include <Eigen/Dense>

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "planner_core/ocp_descriptor.hpp"
#include "planner_core/target_snapshot.hpp"
#include "target/target_accel_buffer.hpp"

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

enum class DiagnosticSeverity {
    Info,
    Warn,
    Error,
};

struct PlannerEvent {
    std::string level = "info";
    std::string message;
    double value = 0.0;
};

struct PlannerDiagnostic {
    DiagnosticSeverity severity = DiagnosticSeverity::Info;
    std::string code;
    std::string message;
    std::map<std::string, double> values;
    Eigen::VectorXd state_a;
    Eigen::VectorXd state_b;
    Eigen::VectorXd control_a;
    Eigen::VectorXd control_b;
};

struct PlannerCoreConfig {
    std::string ocp_type;
    Platform platform = Platform::Crazyflie;
    std::string solver_type = "alipddp";
    double ocp_dt = 0.05;
    int n_replay = 4;
    double max_constraint_error = 1.0;
    bool skip_trajectory_validation = false;
    bool enable_terminal_freeze = true;
    double terminal_freeze_enter_pos = 0.20;
    double terminal_freeze_enter_vel = 0.10;
    bool terminal_freeze_require_vel = false;
    double terminal_freeze_exit_pos = 0.20;
    Eigen::Vector3d terminal_position_abs = Eigen::Vector3d::Zero();
    Eigen::VectorXd terminal_state;
    std::string body_relative_odom_topic = "/drone/body_relative_odom";
    std::string target_frame_odom_topic = "/drone/target_frame_odom";
};

struct PlannerCoreInput {
    Eigen::VectorXd current_state;
    TargetSnapshot target_snapshot;
    target_models::TargetAccelBuffer target_accel_buffer;
    bool has_target_accel_buffer = false;
    OCPDescriptor::DroneOdomMode active_odom_mode = OCPDescriptor::DroneOdomMode::Absolute;
    std::string active_odom_topic;
    std::chrono::steady_clock::time_point now{};
    double ros_time_sec = 0.0;
};

struct PlannerSolveLogData {
    int solve_num = 0;
    double solve_time_ms = 0.0;
    int solve_iters = 0;
    bool is_relative_plan = false;
    double ocp_dt = 0.05;
    std::string coord_mode = "absolute";
    Eigen::Vector3d target_snapshot_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_acc = Eigen::Vector3d::Zero();
    Eigen::Vector4d target_snapshot_quat = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector3d target_snapshot_omega = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_beta = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> target_world_pos_trajectory;
    std::vector<Eigen::Vector3d> target_world_vel_trajectory;
};

struct PlannerCoreStepResult {
    bool solve_attempted = false;
    bool solve_accepted = false;
    bool has_replay_sample = false;
    bool terminal_freeze_engaged = false;
    bool terminal_freeze_released = false;
    bool stale = false;
    std::string rejection_reason;
    std::string x0_source = "live";
    double handoff_pos_err = 0.0;
    double handoff_vel_err = 0.0;
    double replan_delay_sec = 0.0;
    int solve_num = -1;
    SolverResult solve_result;
    PlannerPath path;
    PlannerCommand command;
    PlannerSolveLogData solve_log;
    Eigen::VectorXd actual_state_for_log;
    TargetSnapshot target_for_log;
    std::vector<PlannerDiagnostic> diagnostics;
};

}  // namespace planner_core
