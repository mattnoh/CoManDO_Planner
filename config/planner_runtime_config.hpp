#pragma once

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include <string>

#include "ocp_registry.hpp"

#include "planner_runtime_config_types.hpp"

inline PlannerRuntimeConfig loadPlannerRuntimeConfig(rclcpp::Node* node) {
    PlannerRuntimeConfig cfg;

    node->declare_parameter("ocp_type", cfg.ocp_type);
    cfg.ocp_type = node->get_parameter("ocp_type").as_string();

    if (cfg.isConfigured()) {
        cfg.ocp_dt = OCPRegistry::getDT(cfg.ocp_type);
        cfg.n_replay = OCPRegistry::getDefaultNReplay(cfg.ocp_type);
        cfg.mass_kg = OCPRegistry::getDefaultMassKg(cfg.ocp_type);
    }

    node->declare_parameter("drone_name", cfg.drone_name);
    node->declare_parameter("enable_logging", cfg.enable_logging);
    node->declare_parameter("platform", cfg.platform);
    node->declare_parameter("solver", cfg.solver);
    node->declare_parameter("mode", cfg.mode);
    
    // Used for Hover OCP
    node->declare_parameter("hover_target_x", cfg.hover_target_x);
    node->declare_parameter("hover_target_y", cfg.hover_target_y);
    node->declare_parameter("hover_target_z", cfg.hover_target_z);

    // Used as a target movement for Tracking Circle OCP
    node->declare_parameter("circle_center_x", cfg.circle_center_x);
    node->declare_parameter("circle_center_y", cfg.circle_center_y);
    node->declare_parameter("circle_center_z", cfg.circle_center_z);
    node->declare_parameter("circle_R", cfg.circle_R);
    node->declare_parameter("circle_omega", cfg.circle_omega);
    node->declare_parameter("circle_phi0", cfg.circle_phi0);

    // Decides how many NEX setpoints we send each RH. Does not get called for Open Loop
    node->declare_parameter("n_replay", cfg.n_replay);

    node->declare_parameter("drone_state_is_relative", cfg.drone_state_is_relative);
    node->declare_parameter("drone_odom_topic", cfg.drone_odom_topic);

    // Topics for externernal target tracking <- need to work on this later
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

    cfg.hover_target_x = node->get_parameter("hover_target_x").as_double();
    cfg.hover_target_y = node->get_parameter("hover_target_y").as_double();
    cfg.hover_target_z = node->get_parameter("hover_target_z").as_double();

    cfg.circle_center_x = node->get_parameter("circle_center_x").as_double();
    cfg.circle_center_y = node->get_parameter("circle_center_y").as_double();
    cfg.circle_center_z = node->get_parameter("circle_center_z").as_double();
    cfg.circle_R = node->get_parameter("circle_R").as_double();
    cfg.circle_omega = node->get_parameter("circle_omega").as_double();
    cfg.circle_phi0 = node->get_parameter("circle_phi0").as_double();

    cfg.n_replay = node->get_parameter("n_replay").as_int();

    cfg.drone_state_is_relative = node->get_parameter("drone_state_is_relative").as_bool();
    cfg.drone_odom_topic = node->get_parameter("drone_odom_topic").as_string();

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
