#pragma once

#include <string>

#include <optional>

#include "core/planner_config.hpp"
#include "target/target_accel_buffer.hpp"

struct PlannerRuntimeConfig {
    std::string ocp_type = "";
    std::string drone_name = "gogogo";
    bool enable_logging = true;
    std::string platform = "crazyflie";
    std::string solver = "alipddp";
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

    // If true, current_state is already relative and registry must not subtract target.
    bool drone_state_is_relative = false;
    // Optional override topic carrying full drone odometry (pose+twist) for planner state input.
    // Example for direct relative-state testing: /drone/relative_odometry
    std::string drone_odom_topic = "";

    std::string target_odom_topic = "/target/odom";
    std::string target_accel_topic = "/target/accel";
    std::string target_predicted_accel_topic = "/target/predicted_accel";

    bool enable_terminal_freeze = true;
    double terminal_freeze_enter_pos = 0.20;
    double terminal_freeze_enter_vel = 0.10;
    bool terminal_freeze_require_vel = false;
    double terminal_freeze_exit_pos = 0.20;

    bool start_paused = true;
    int command_seq = 0;

    double ocp_dt = 0.0;
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

inline PlannerConfig toPlannerConfig(const PlannerRuntimeConfig& cfg) {
    PlannerConfig out;
    out.ocp_type = cfg.ocp_type;
    out.drone_name = cfg.drone_name;
    out.enable_logging = cfg.enable_logging;
    out.platform = cfg.platform;
    out.solver_type = cfg.solver;
    out.mode = cfg.mode;
    out.hover_target_x = cfg.hover_target_x;
    out.hover_target_y = cfg.hover_target_y;
    out.hover_target_z = cfg.hover_target_z;
    out.circle_center_x = cfg.circle_center_x;
    out.circle_center_y = cfg.circle_center_y;
    out.circle_center_z = cfg.circle_center_z;
    out.circle_R = cfg.circle_R;
    out.circle_omega = cfg.circle_omega;
    out.circle_phi0 = cfg.circle_phi0;
    out.n_replay = cfg.n_replay;
    out.drone_state_is_relative = cfg.drone_state_is_relative;
    out.drone_odom_topic = cfg.drone_odom_topic;
    out.target_odom_topic = cfg.target_odom_topic;
    out.target_accel_topic = cfg.target_accel_topic;
    out.target_predicted_accel_topic = cfg.target_predicted_accel_topic;
    out.enable_terminal_freeze = cfg.enable_terminal_freeze;
    out.terminal_freeze_enter_pos = cfg.terminal_freeze_enter_pos;
    out.terminal_freeze_enter_vel = cfg.terminal_freeze_enter_vel;
    out.terminal_freeze_require_vel = cfg.terminal_freeze_require_vel;
    out.terminal_freeze_exit_pos = cfg.terminal_freeze_exit_pos;
    out.start_paused = cfg.start_paused;
    out.command_seq = cfg.command_seq;
    out.ocp_dt = cfg.ocp_dt;
    out.dt = cfg.ocp_dt;
    out.mass_kg = cfg.mass_kg;
    out.open_loop_abort_on_divergence = cfg.open_loop_abort_on_divergence;
    out.open_loop_abort_max_z_error_m = cfg.open_loop_abort_max_z_error_m;
    out.open_loop_abort_max_vz_error_mps = cfg.open_loop_abort_max_vz_error_mps;
    out.t_start_abs = cfg.t_start_abs;
    out.target_accel_buffer = cfg.target_accel_buffer;
    return out;
}
