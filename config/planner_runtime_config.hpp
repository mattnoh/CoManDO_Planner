#pragma once

#include <Eigen/Dense>
#include <ros/ros.h>

#include <string>

#include "planner_core/ocp_registry.hpp"

#include "planner_runtime_config_types.hpp"

inline PlannerRuntimeConfig loadPlannerRuntimeConfig(ros::NodeHandle& nh) {
    PlannerRuntimeConfig cfg;

    nh.param<std::string>("ocp_type", cfg.ocp_type, cfg.ocp_type);

    nh.param<std::string>("drone_name",    cfg.drone_name,    cfg.drone_name);
    nh.param<bool>       ("enable_logging", cfg.enable_logging, cfg.enable_logging);
    nh.param<std::string>("platform",      cfg.platform,      cfg.platform);
    nh.param<std::string>("solver",        cfg.solver,        cfg.solver);
    nh.param<std::string>("mode",          cfg.mode,          cfg.mode);

    // Must run AFTER the mode param is read: isConfigured() needs both
    // ocp_type and mode, and a node restarted with sticky rosparams would
    // otherwise come up configured with ocp_dt=0 (no replay timer).
    if (cfg.isConfigured()) {
        cfg.ocp_dt = OCPRegistry::getDT(cfg.ocp_type);
        cfg.n_replay = OCPRegistry::getDefaultNReplay(cfg.ocp_type);
        cfg.mass_kg = OCPRegistry::getDefaultMassKg(cfg.ocp_type);
    }

    nh.param<double>("hover_target_x", cfg.hover_target_x, cfg.hover_target_x);
    nh.param<double>("hover_target_y", cfg.hover_target_y, cfg.hover_target_y);
    nh.param<double>("hover_target_z", cfg.hover_target_z, cfg.hover_target_z);

    nh.param<double>("circle_center_x", cfg.circle_center_x, cfg.circle_center_x);
    nh.param<double>("circle_center_y", cfg.circle_center_y, cfg.circle_center_y);
    nh.param<double>("circle_center_z", cfg.circle_center_z, cfg.circle_center_z);
    nh.param<double>("circle_R",        cfg.circle_R,        cfg.circle_R);
    nh.param<double>("circle_omega",    cfg.circle_omega,    cfg.circle_omega);
    nh.param<double>("circle_phi0",     cfg.circle_phi0,     cfg.circle_phi0);

    nh.param<int>   ("n_replay",     cfg.n_replay,     cfg.n_replay);
    nh.param<bool>  ("open_loop_abort_on_divergence",    cfg.open_loop_abort_on_divergence,    cfg.open_loop_abort_on_divergence);
    nh.param<double>("open_loop_abort_max_z_error_m",   cfg.open_loop_abort_max_z_error_m,   cfg.open_loop_abort_max_z_error_m);
    nh.param<double>("open_loop_abort_max_vz_error_mps",cfg.open_loop_abort_max_vz_error_mps,cfg.open_loop_abort_max_vz_error_mps);

    nh.param<std::string>("body_relative_odom_topic",     cfg.body_relative_odom_topic,     cfg.body_relative_odom_topic);
    nh.param<std::string>("target_odom_topic",            cfg.target_odom_topic,            cfg.target_odom_topic);
    nh.param<std::string>("target_accel_topic",           cfg.target_accel_topic,           cfg.target_accel_topic);
    nh.param<std::string>("target_predicted_accel_topic", cfg.target_predicted_accel_topic, cfg.target_predicted_accel_topic);
    nh.param<bool>       ("debug_body_relative_trace",    cfg.debug_body_relative_trace,    cfg.debug_body_relative_trace);

    nh.param<bool>  ("enable_terminal_freeze",      cfg.enable_terminal_freeze,      cfg.enable_terminal_freeze);
    nh.param<double>("terminal_freeze_enter_pos",   cfg.terminal_freeze_enter_pos,   cfg.terminal_freeze_enter_pos);
    nh.param<double>("terminal_freeze_enter_vel",   cfg.terminal_freeze_enter_vel,   cfg.terminal_freeze_enter_vel);
    nh.param<bool>  ("terminal_freeze_require_vel", cfg.terminal_freeze_require_vel, cfg.terminal_freeze_require_vel);
    nh.param<double>("terminal_freeze_exit_pos",    cfg.terminal_freeze_exit_pos,    cfg.terminal_freeze_exit_pos);

    nh.param<bool>("start_paused",  cfg.start_paused,  cfg.start_paused);
    nh.param<int> ("command_seq",   cfg.command_seq,   cfg.command_seq);

    return cfg;
}
