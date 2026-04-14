#pragma once

#include <optional>
#include <string>

#include "target/target_accel_buffer.hpp"

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
    double dt = 0.0;
    double mass_kg = 0.0;

    double t_start_abs = 0.0;
    std::optional<target_models::TargetAccelBuffer> target_accel_buffer;

    bool isConfigured() const {
        return !ocp_type.empty() && !mode.empty();
    }
};
