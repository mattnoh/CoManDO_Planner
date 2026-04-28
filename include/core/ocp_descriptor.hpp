/// @file ocp_descriptor.hpp
/// @brief OCPDescriptor and supporting structs.
///
/// Include this from OCP headers (ocp/*.hpp) to implement descriptor().
/// Do NOT include ocp_registry.hpp from OCP headers — that would be circular.
#pragma once

#include <Eigen/Dense>
#include <any>
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/target_snapshot.hpp"
#include "core/planner_config.hpp"
#include "core/ocp_logger_defaults.hpp"
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

// Forward declaration — avoid pulling ROS2 into OCP headers.
namespace planner_logging { struct SolveLogMeta; }

// ─── SolverResult ─────────────────────────────────────────────────────────────

struct SolverResult {
    bool success = false;
    Eigen::VectorXd next_state;
    std::vector<Eigen::VectorXd> state_trajectory;
    std::vector<Eigen::VectorXd> control_trajectory;
    double solve_time_ms = 0.0;
    double constraint_error = 0.0;
    int solve_iters = 0;
    std::chrono::steady_clock::time_point solve_timestamp;

    bool is_relative_plan = false;
    Eigen::Vector3d target_snapshot_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_acc = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> target_world_pos_trajectory;
    std::vector<Eigen::Vector3d> target_world_vel_trajectory;
    std::string target_motion_source = "snapshot";
    std::any extra;
};

// ─── OCPCreateArgs ────────────────────────────────────────────────────────────

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

// ─── OCPDescriptor ────────────────────────────────────────────────────────────

struct OCPDescriptor {
    enum class CommandMode  { CmdFullState, CmdBodyRate };
    enum class DroneOdomMode { Absolute, AbsoluteShiftedTarget, BodyFrameRelative, TargetFrameRelative };
    enum class WarmStart    { Shift, Feedback };

    // --- Identity / timing ---
    std::string name;
    double dt = 0.05;
    int default_n_replay = 4;
    double default_mass_kg = 0.027;
    WarmStart warm_start = WarmStart::Shift;
    CommandMode command_mode = CommandMode::CmdFullState;
    DroneOdomMode drone_odom_mode = DroneOdomMode::Absolute;

    // --- OCP behaviour flags ---
    bool needs_target_trajectory = false;
    bool skip_altitude_validation = false;
    bool variable_dt = false;

    // --- State / control dimensions ---
    int state_dim = 13;
    int control_dim = 4;

    // --- Logging schema (set in descriptor(); defaults cover 13D OCPs) ---
    std::vector<std::string> log_state_headers;
    /// Column names for raw OCP state vector in all_solves.csv.
    /// Must match the dynamics state layout (e.g. "px","py","pz","vx",...)
    std::vector<std::string> state_names;
    /// Column names for raw OCP control vector in all_solves.csv.
    /// Must match the control layout (e.g. "fz","mx","my","mz")
    std::vector<std::string> control_names;
    /// Row extractor for actual_state.csv.
    /// Args: (raw sensor state, target snapshot, coord_mode string)
    std::function<std::vector<double>(
        const Eigen::VectorXd&, const TargetSnapshot&, const std::string&)>
        extract_actual_state_row;

    // --- Hover-state factory (nullptr → standard 13D zero-vel flat-attitude) ---
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> make_hover_state;

    // --- Callbacks (nullptr = not needed) ---
    /// Transform raw drone state into OCP state at solve time.
    /// e.g. subtract target for relative OCPs, or assemble 22D augmented state.
    std::function<Eigen::VectorXd(
        const Eigen::VectorXd& x_drone, const TargetSnapshot&)>
        transform_state;

    /// Return false → planner withholds solve (target data stale/invalid).
    std::function<bool(const TargetSnapshot&, double now_sec, double max_age)>
        validate_target;

    /// Modify SolverResult after solve (e.g. mark as relative, reconstruct abs traj).
    std::function<void(SolverResult&, const TargetSnapshot&)>
        post_process_result;

    /// Build OCP-specific extra params passed to create().
    std::function<std::any(const PlannerConfig&, double t_abs, const TargetSnapshot&)>
        prepare_extra;

    /// Populate additional SolveLogMeta fields (e.g. circle phase).
    std::function<void(planner_logging::SolveLogMeta&, const std::any& extra_params,
                       const std::any& runtime_cfg)>
        prepare_log_meta;

    /// Merge solver-propagated augmented states into the freshly-transformed x0.
    /// Called after transform_state when a previous trajectory exists.
    /// The physical indices [0:13] are kept from the live sensor state.
    /// The augmented indices (e.g. Ω_N, a_T^B, β_N at [13:22]) are overwritten
    /// from prev_X[n_shift] so the solver carries forward the target kinematics
    /// that its own dynamics have propagated, rather than resetting from snapshot.
    /// Args: (x0 after transform_state, previous X trajectory, n_shift count)
    std::function<void(Eigen::VectorXd&,
                       const std::vector<Eigen::VectorXd>&,
                       int)> merge_prev_augmented;

    // --- OCP factory ---
    std::function<Param()> getSolverParams;
    std::function<std::shared_ptr<OptimalControlProblem<double>>(const OCPCreateArgs&)>
        create;
};
