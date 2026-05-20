/// @file planner_core/types.hpp
/// @brief ROS-free planner/OCP data contracts for future ROS1/ROS2 wrappers.
#pragma once

#include <Eigen/Dense>

#include <any>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "target/target_accel_buffer.hpp"

/// Snapshot of target state taken at solve time.
/// Passed to transform_state, validate_target, post_process_result callbacks.
struct TargetSnapshot {
    bool valid = false;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector4d orientation = Eigen::Vector4d(1,0,0,0);
    double odom_stamp_sec = 0.0;
    double accel_stamp_sec = 0.0;
};

struct PlannerConfig {
    std::string ocp_type = "";
    std::string drone_name = "gogogo";
    bool enable_logging = true;
    std::string platform = "crazyflie";
    std::string solver_type = "alipddp";
    std::string mode = "";

    double hover_target_x = 0.0;
    double hover_target_y = 0.0;
    double hover_target_z = 0.0;

    double circle_center_x = 0.0;
    double circle_center_y = 0.0;
    double circle_center_z = 0.2;
    double circle_R = 1.0;
    double circle_omega = 0.4;
    double circle_phi0 = 0.0;

    int n_replay = 0;

    bool drone_state_is_relative = false;
    std::string drone_odom_topic = "";
    std::string body_relative_odom_topic = "/drone/body_relative_odom";

    std::string target_odom_topic = "/target/odom";
    std::string target_accel_topic = "/target/accel";
    std::string target_predicted_accel_topic = "/target/predicted_accel";
    bool debug_body_relative_trace = false;

    bool enable_terminal_freeze = true;
    double terminal_freeze_enter_pos = 0.20;
    double terminal_freeze_enter_vel = 0.10;
    bool terminal_freeze_require_vel = false;
    double terminal_freeze_exit_pos = 0.20;

    bool start_paused = true;
    int command_seq = 0;

    double ocp_dt = 0.0;
    double dt = 0.0;
    double mass_kg = 0.0;
    bool open_loop_abort_on_divergence = false;
    double open_loop_abort_max_z_error_m = 0.50;
    double open_loop_abort_max_vz_error_mps = 1.00;

    double t_start_abs = 0.0;
    std::optional<target_models::TargetAccelBuffer> target_accel_buffer;

    bool isConfigured() const {
        return !ocp_type.empty() && !mode.empty();
    }
};

namespace OCPLoggerDefaults {

inline Eigen::VectorXd makeHoverState13D(const Eigen::VectorXd& current_state) {
    Eigen::VectorXd x_hover = (current_state.size() >= 13)
        ? current_state
        : Eigen::VectorXd::Zero(13);
    x_hover.segment(3, 3).setZero();
    x_hover(6) = 1.0;
    x_hover.segment(7, 3).setZero();
    x_hover.segment(10, 3).setZero();
    return x_hover;
}

inline std::vector<std::string> getStateHeaders13D() {
    return {
        "x","y","z","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz",
        "abs_x","abs_y","abs_z","abs_vx","abs_vy","abs_vz","abs_qw","abs_qx","abs_qy","abs_qz","abs_wx","abs_wy","abs_wz",
        "rel_x","rel_y","rel_z","rel_vx","rel_vy","rel_vz","rel_qw","rel_qx","rel_qy","rel_qz","rel_wx","rel_wy","rel_wz",
        "tgt_x","tgt_y","tgt_z","tgt_vx","tgt_vy","tgt_vz"
    };
}

inline std::vector<double> getActualStateRow13D(const Eigen::VectorXd& state, const TargetSnapshot& tgt, const std::string& coord_mode) {
    if (state.size() < 13) return std::vector<double>(45, 0.0);
    std::vector<double> row;
    row.reserve(45);
    Eigen::VectorXd x_abs = state.head(13);
    Eigen::VectorXd x_rel = state.head(13);
    if (coord_mode == "relative") {
        x_abs.segment(0, 3) += tgt.position;
        x_abs.segment(3, 3) += tgt.velocity;
    } else {
        x_rel.segment(0, 3) -= tgt.position;
        x_rel.segment(3, 3) -= tgt.velocity;
    }
    for (int i=0; i<13; ++i) row.push_back(state(i));
    for (int i=0; i<13; ++i) row.push_back(x_abs(i));
    for (int i=0; i<13; ++i) row.push_back(x_rel(i));
    row.push_back(tgt.position.x()); row.push_back(tgt.position.y()); row.push_back(tgt.position.z());
    row.push_back(tgt.velocity.x()); row.push_back(tgt.velocity.y()); row.push_back(tgt.velocity.z());
    return row;
}

inline std::vector<std::string> getStateNames13D() {
    return {"px","py","pz","vx","vy","vz","qw","qx","qy","qz","wx","wy","wz"};
}

inline std::vector<std::string> getControlNames4D() {
    return {"fz","mx","my","mz"};
}

inline std::vector<std::string> getStateNames22D() {
    return {"p0","p1","p2","v0","v1","v2","qw","qx","qy","qz",
            "wx","wy","wz","OmN0","OmN1","OmN2","aT0","aT1","aT2",
            "bN0","bN1","bN2"};
}

inline std::vector<std::string> getStateHeaders23D_Body() {
    return {
        "p0","p1","p2",
        "v0","v1","v2",
        "qw","qx","qy","qz",
        "wx","wy","wz",
        "drone_x","drone_y","drone_z",
        "p_err",
        "tgt_x","tgt_y","tgt_z","tgt_vx","tgt_vy","tgt_vz"
    };
}

inline std::vector<double> getActualStateRow23D_Body(
    const Eigen::VectorXd& state,
    const TargetSnapshot& tgt,
    const std::string&)
{
    constexpr int N_OUT  = 19;
    std::vector<double> row;
    row.reserve(N_OUT);

    auto s = [&](int i) -> double {
        return (i < (int)state.size()) ? state(i) : 0.0;
    };

    row.push_back(s(0)); row.push_back(s(1)); row.push_back(s(2));
    row.push_back(s(3)); row.push_back(s(4)); row.push_back(s(5));
    row.push_back(s(6)); row.push_back(s(7)); row.push_back(s(8)); row.push_back(s(9));
    row.push_back(s(10)); row.push_back(s(11)); row.push_back(s(12));

    {
        const double qw = s(6), qx = s(7), qy = s(8), qz = s(9);
        Eigen::Matrix3d R_WB;
        R_WB << 1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw,
                2*qx*qy + 2*qz*qw,      1 - 2*qx*qx - 2*qz*qz,  2*qy*qz - 2*qx*qw,
                2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,      1 - 2*qx*qx - 2*qy*qy;
        Eigen::Vector3d p_T_b(s(0), s(1), s(2));
        Eigen::Vector3d drone_w = tgt.position - R_WB * p_T_b;
        row.push_back(drone_w(0)); row.push_back(drone_w(1)); row.push_back(drone_w(2));
        row.push_back(p_T_b.norm());
    }

    row.push_back(tgt.position.x()); row.push_back(tgt.position.y()); row.push_back(tgt.position.z());
    row.push_back(tgt.velocity.x()); row.push_back(tgt.velocity.y()); row.push_back(tgt.velocity.z());

    return row;
}

inline Eigen::VectorXd makeHoverState23D_Body(const Eigen::VectorXd& current_state) {
    Eigen::VectorXd x_hover = (current_state.size() >= 22)
        ? current_state
        : Eigen::VectorXd::Zero(22);
    x_hover.segment(3, 3).setZero();
    x_hover(6) = 1.0;
    x_hover.segment(7, 3).setZero();
    x_hover.segment(10, 3).setZero();
    return x_hover;
}

}  // namespace OCPLoggerDefaults

namespace planner_logging { struct SolveLogMeta; }

struct SolverResult {
    bool success = false;
    Eigen::VectorXd next_state;
    std::vector<Eigen::VectorXd> state_trajectory;
    std::vector<Eigen::VectorXd> control_trajectory;
    double solve_time_ms = 0.0;
    double constraint_error = 0.0;
    int solve_iters = 0;
    std::chrono::steady_clock::time_point solve_start_time;
    std::chrono::steady_clock::time_point solve_finish_time;
    std::chrono::steady_clock::time_point solve_timestamp;

    bool is_relative_plan = false;
    Eigen::Vector3d target_snapshot_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_acc = Eigen::Vector3d::Zero();
    Eigen::Vector4d target_snapshot_quat = Eigen::Vector4d(1,0,0,0);
    Eigen::Vector3d target_snapshot_omega = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_beta = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> target_world_pos_trajectory;
    std::vector<Eigen::Vector3d> target_world_vel_trajectory;
    std::string target_motion_source = "snapshot";
    std::any extra;
};

struct OCPCreateArgs {
    Eigen::VectorXd current_state;
    Eigen::VectorXd terminal_state;
    std::vector<Eigen::VectorXd> prev_U;
    std::vector<Eigen::VectorXd> prev_X;
    std::vector<Eigen::MatrixXd> prev_K;
    Eigen::Vector3d target_accel;
    std::any extra;
    double t_abs = 0.0;
};

struct OCPDescriptor {
    enum class CommandMode  { CmdFullState, CmdBodyRate };
    enum class DroneOdomMode { Absolute, AbsoluteShiftedTarget, BodyFrameRelative, TargetFrameRelative };
    enum class WarmStart    { Shift, Feedback };

    std::string name;
    double dt = 0.05;
    int default_n_replay = 4;
    double default_mass_kg = 0.027;
    WarmStart warm_start = WarmStart::Shift;
    CommandMode command_mode = CommandMode::CmdFullState;
    DroneOdomMode drone_odom_mode = DroneOdomMode::Absolute;

    bool needs_target_trajectory = false;
    bool skip_altitude_validation = false;
    bool skip_trajectory_validation = false;
    bool variable_dt = false;
    bool use_predicted_handoff_state = true;
    bool disarm_on_landing_finish = false;

    int state_dim = 13;
    int control_dim = 4;

    std::vector<std::string> log_state_headers;
    std::vector<std::string> state_names;
    std::vector<std::string> control_names;
    std::function<std::vector<double>(
        const Eigen::VectorXd&, const TargetSnapshot&, const std::string&)>
        extract_actual_state_row;

    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> make_hover_state;
    std::function<Eigen::VectorXd(
        const Eigen::VectorXd& x_drone, const TargetSnapshot&)>
        transform_state;
    std::function<bool(const TargetSnapshot&, double now_sec, double max_age)>
        validate_target;
    std::function<void(SolverResult&, const TargetSnapshot&)>
        post_process_result;
    std::function<std::any(const PlannerConfig&, double t_abs, const TargetSnapshot&)>
        prepare_extra;
    std::function<void(planner_logging::SolveLogMeta&, const std::any& extra_params,
                       const std::any& runtime_cfg)>
        prepare_log_meta;
    std::function<void(Eigen::VectorXd&,
                       const std::vector<Eigen::VectorXd>&,
                       int)> merge_prev_augmented;
    std::function<void(Eigen::VectorXd&)> sanitize_warm_control;
    std::function<Eigen::VectorXd(
        const Eigen::VectorXd& x_ocp,
        const TargetSnapshot& target)> reconstruct_world_state;

    std::function<Param()> getSolverParams;
    std::function<std::shared_ptr<OptimalControlProblem<double>>(const OCPCreateArgs&)>
        create;
};

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
    double solve_lead_guard_sec = 0.05;
    double initial_solve_lead_sec = 0.20;
    double recovery_pos_err = 0.50;
    double recovery_vel_err = 2.00;
    double max_first_solve_age_sec = 1.00;
    double max_relative_position_norm = 10.0;
    double max_relative_vertical_abs = 5.0;
    double max_relative_velocity_norm = 8.0;
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
    double active_elapsed_now_sec = 0.0;
    double activation_elapsed_sec = 0.0;
    double activation_wall_time_sec = 0.0;
    double solve_lead_sec = 0.0;
    double solve_finish_late_by_sec = 0.0;
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
