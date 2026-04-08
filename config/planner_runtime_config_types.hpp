#pragma once

#include <string>

#include <optional>
#include "target/target_accel_buffer.hpp"

struct PlannerRuntimeConfig {
    std::string ocp_type = "";
    std::string drone_name = "cf_1";
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

    std::string target_odom_topic = "/target/odom";
    std::string target_accel_topic = "/target/accel";

    bool enable_terminal_freeze = true;
    double terminal_freeze_enter_pos = 0.20;
    double terminal_freeze_enter_vel = 0.10;
    bool terminal_freeze_require_vel = false;
    double terminal_freeze_exit_pos = 0.20;

    bool start_paused = true;
    int command_seq = 0;

    double ocp_dt = 0.0;
    double mass_kg = 0.0;

    double t_start_abs = 0.0;
    std::optional<target_models::TargetAccelBuffer> target_accel_buffer;

    bool isConfigured() const {
        return !ocp_type.empty() && !mode.empty();
    }
};
