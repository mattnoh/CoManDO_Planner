#pragma once

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include <string>

#include "ocp_registry.hpp"

struct PlannerRuntimeConfig {
    std::string ocp_type = "landing";
    std::string drone_name = "cf_1";
    bool enable_logging = true;
    std::string platform = "crazyflie";
    std::string solver = "alipddp";
    std::string mode = "mpc";

    Eigen::Vector3d hover_target = Eigen::Vector3d(0.0, 0.0, 1.0);

    int n_replay = 0;
    double mass_kg = 0.0;

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
};

inline PlannerRuntimeConfig loadPlannerRuntimeConfig(rclcpp::Node* node) {
    PlannerRuntimeConfig cfg;

    node->declare_parameter("ocp_type", cfg.ocp_type);
    cfg.ocp_type = node->get_parameter("ocp_type").as_string();

    cfg.ocp_dt = OCPRegistry::getDT(cfg.ocp_type);
    cfg.n_replay = OCPRegistry::getDefaultNReplay(cfg.ocp_type);
    cfg.mass_kg = OCPRegistry::getDefaultMassKg(cfg.ocp_type);

    node->declare_parameter("drone_name", cfg.drone_name);
    node->declare_parameter("enable_logging", cfg.enable_logging);
    node->declare_parameter("platform", cfg.platform);
    node->declare_parameter("solver", cfg.solver);
    node->declare_parameter("mode", cfg.mode);

    node->declare_parameter("hover_target_x", cfg.hover_target.x());
    node->declare_parameter("hover_target_y", cfg.hover_target.y());
    node->declare_parameter("hover_target_z", cfg.hover_target.z());

    node->declare_parameter("n_replay", cfg.n_replay);
    node->declare_parameter("mass_kg", cfg.mass_kg);

    node->declare_parameter("target_odom_topic", cfg.target_odom_topic);
    node->declare_parameter("target_accel_topic", cfg.target_accel_topic);

    node->declare_parameter("enable_terminal_freeze", cfg.enable_terminal_freeze);
    node->declare_parameter("terminal_freeze_enter_pos", cfg.terminal_freeze_enter_pos);
    node->declare_parameter("terminal_freeze_enter_vel", cfg.terminal_freeze_enter_vel);
    node->declare_parameter("terminal_freeze_require_vel", cfg.terminal_freeze_require_vel);
    node->declare_parameter("terminal_freeze_exit_pos", cfg.terminal_freeze_exit_pos);

    node->declare_parameter("start_paused", cfg.start_paused);
    node->declare_parameter("command_seq", cfg.command_seq);

    cfg.drone_name = node->get_parameter("drone_name").as_string();
    cfg.enable_logging = node->get_parameter("enable_logging").as_bool();
    cfg.platform = node->get_parameter("platform").as_string();
    cfg.solver = node->get_parameter("solver").as_string();
    cfg.mode = node->get_parameter("mode").as_string();

    cfg.hover_target.x() = node->get_parameter("hover_target_x").as_double();
    cfg.hover_target.y() = node->get_parameter("hover_target_y").as_double();
    cfg.hover_target.z() = node->get_parameter("hover_target_z").as_double();

    cfg.n_replay = node->get_parameter("n_replay").as_int();
    cfg.mass_kg = node->get_parameter("mass_kg").as_double();

    cfg.target_odom_topic = node->get_parameter("target_odom_topic").as_string();
    cfg.target_accel_topic = node->get_parameter("target_accel_topic").as_string();

    cfg.enable_terminal_freeze = node->get_parameter("enable_terminal_freeze").as_bool();
    cfg.terminal_freeze_enter_pos = node->get_parameter("terminal_freeze_enter_pos").as_double();
    cfg.terminal_freeze_enter_vel = node->get_parameter("terminal_freeze_enter_vel").as_double();
    cfg.terminal_freeze_require_vel = node->get_parameter("terminal_freeze_require_vel").as_bool();
    cfg.terminal_freeze_exit_pos = node->get_parameter("terminal_freeze_exit_pos").as_double();

    cfg.start_paused = node->get_parameter("start_paused").as_bool();
    cfg.command_seq = node->get_parameter("command_seq").as_int();

    return cfg;
}
