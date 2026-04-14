/// @file ocp_registry.hpp
/// @brief Centralized OCP lookup and factory — reduces duplication and eases adding new OCPs
///
/// tracking_circle: Single-shot open-loop OCP for landing on circular target.
/// Uses time-varying dynamics (Quad6DOFVarTimeRelativeTV) and outputs absolute coordinates.

#pragma once

#include "ocp/ocp_hover.hpp"
#include "ocp/ocp_landing.hpp"
#include "ocp/ocp_stateswitch.hpp"
#include "ocp/ocp_tracking_circle.hpp"
#include "ocp/ocp_tracking_circle_target.hpp"

#include "target/circular_target.hpp"
#include "target/target_accel_buffer.hpp"

#include <string>
#include <stdexcept>
#include <memory>
#include <map>
#include <functional>
#include <any>

#include "core/planner_config.hpp"
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

#include "planner_logging.hpp"

// Forward declare struct so we can use it in OCPRegistry
struct TargetSnapshot {
    bool valid = false;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    double odom_stamp_sec = 0.0;
    double accel_stamp_sec = 0.0;
};

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

// Note: target_models::CircularTarget must be in target/circular_target.hpp

struct OCPCreateArgs {
    Eigen::VectorXd current_state;
    Eigen::VectorXd terminal_state;
    std::vector<Eigen::VectorXd> prev_U;
    std::vector<Eigen::VectorXd> prev_X;
    std::vector<Eigen::MatrixXd> prev_K;
    Eigen::Vector3d target_accel;
    std::any extra;  // OCP-specific params (e.g. TrackingCircleOCP::CircularTarget)
    double t_abs = 0.0;
};

struct OCPDescriptor {
    std::string name;
    double dt;
    int default_n_replay;
    double default_mass_kg;
    enum class WarmStart { Shift, Feedback } warm_start;
    bool needs_target_trajectory = false;

    // Callbacks — nullptr = not needed
    std::function<Eigen::VectorXd(
        const Eigen::VectorXd& x_drone,
        const TargetSnapshot&)>              transform_state;

    std::function<bool(const TargetSnapshot&,
        double now_sec, double max_age)>     validate_target;

    std::function<void(SolverResult&,
        const TargetSnapshot&)>              post_process_result;
    
    std::function<std::any(const PlannerConfig& cfg, double t_abs, const TargetSnapshot& tgt_snap)> prepare_extra;

    std::function<void(planner_logging::SolveLogMeta& meta, const std::any& extra_params, const std::any& runtime_cfg)> prepare_log_meta;

    std::function<Param()>                   getSolverParams;
    std::function<std::shared_ptr<OptimalControlProblem<double>>(
        const OCPCreateArgs&)>               create;
};

namespace OCPRegistry {

inline const std::map<std::string, OCPDescriptor>& getTable() {
    static const std::map<std::string, OCPDescriptor> table = {
        {"hover", {
            "hover",
            HoverOCP::DT,
            HoverOCP::DEFAULT_N_REPLAY,
            HoverOCP::DEFAULT_MASS_KG,
            OCPDescriptor::WarmStart::Shift,
            false, // needs_target_trajectory
            nullptr, // transform_state
            nullptr, // validate_target
            nullptr, // post_process_result
            nullptr, // prepare_extra
            nullptr, // prepare_log_meta
            HoverOCP::getSolverParams,
            [](const OCPCreateArgs& a) {
                return HoverOCP::create(a.current_state, a.terminal_state);
            }
        }},
        {"landing", {
            "landing",
            LandingOCP::DT,
            LandingOCP::DEFAULT_N_REPLAY,
            LandingOCP::DEFAULT_MASS_KG,
            OCPDescriptor::WarmStart::Shift,
            false, // needs_target_trajectory
            nullptr, // transform_state
            nullptr, // validate_target
            nullptr, // post_process_result
            nullptr, // prepare_extra
            nullptr, // prepare_log_meta
            LandingOCP::getSolverParams,
            [](const OCPCreateArgs& a) {
                return LandingOCP::create(a.current_state, a.terminal_state, a.prev_U, a.prev_X);
            }
        }},
        {"stateswitch", {
            "stateswitch",
            StateswitchOCP::TH_INIT,
            StateswitchOCP::DEFAULT_N_REPLAY,
            StateswitchOCP::DEFAULT_MASS_KG,
            OCPDescriptor::WarmStart::Feedback,
            false, // needs_target_trajectory
            [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
                if (t.valid) {
                    Eigen::VectorXd xr = x;
                    xr.segment(0, 3) -= t.position;
                    xr.segment(3, 3) -= t.velocity;
                    return xr;
                }
                return x;
            },
            nullptr, // validate_target
            [](SolverResult& r, const TargetSnapshot& t) {
                r.is_relative_plan = t.valid;
                if (t.valid) {
                    r.target_snapshot_pos = t.position;
                    r.target_snapshot_vel = t.velocity;
                    r.target_snapshot_acc = t.acceleration;
                } else {
                    r.target_snapshot_pos.setZero();
                    r.target_snapshot_vel.setZero();
                    r.target_snapshot_acc.setZero();
                }
            },
            [](const PlannerConfig& cfg, double t_abs, const TargetSnapshot& /*tgt_snap*/) {
                StateswitchOCP::StateswitchExtra ex;
                ex.t0_abs = t_abs;

                if (cfg.target_accel_buffer.has_value() && !cfg.target_accel_buffer->accels.empty()) {
                    ex.buf = cfg.target_accel_buffer.value();
                } else {
                    target_models::CircularTarget circ;
                    circ.center << cfg.circle_center_x, cfg.circle_center_y, cfg.circle_center_z;
                    circ.R = cfg.circle_R;
                    circ.omega = cfg.circle_omega;
                    circ.phi0 = cfg.circle_phi0 - cfg.circle_omega * cfg.t_start_abs;

                    const double buf_dur = StateswitchOCP::HORIZON * StateswitchOCP::THH + 1.0;
                    ex.buf.populateFromModel(circ, t_abs, buf_dur, 0.05);
                }

                return std::any(ex);
            },
            nullptr, // prepare_log_meta
            StateswitchOCP::getSolverParams,
            [](const OCPCreateArgs& a) {
                StateswitchOCP::StateswitchExtra ex;
                if (a.extra.has_value()) {
                    try {
                        ex = std::any_cast<StateswitchOCP::StateswitchExtra>(a.extra);
                    } catch (const std::bad_any_cast&) {}
                }
                return StateswitchOCP::create(a.current_state, a.terminal_state,
                                              a.prev_U, a.prev_X, a.target_accel, a.prev_K,
                                              ex.buf, ex.t0_abs);
            }
        }},
        {"tracking_circle", {
            "tracking_circle",
            TrackingCircleOCP::TH_INIT,
            TrackingCircleOCP::DEFAULT_N_REPLAY,
            TrackingCircleOCP::MASS,
            OCPDescriptor::WarmStart::Shift,
            false, // needs_target_trajectory
            [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
                if (t.valid) {
                    Eigen::VectorXd xr = x;
                    xr.segment(0, 3) -= t.position;
                    xr.segment(3, 3) -= t.velocity;
                    return xr;
                }
                return x;
            },
            nullptr, // validate_target
            [](SolverResult& r, const TargetSnapshot& /*t*/) {
                if (r.extra.has_value()) {
                    try {
                        auto ex = std::any_cast<TrackingCircleOCP::TrackingCircleExtra>(r.extra);
                        // Convert relative trajectory to absolute using baked-in dynamics
                        r.state_trajectory = TrackingCircleOCP::convertToAbsolute(r.state_trajectory, ex.tgt, ex.t_abs);
                        // We set is_relative_plan=false so mpcReplayTick doesn't try to add live target state
                        r.is_relative_plan = false;
                    } catch (const std::bad_any_cast&) {
                        std::cerr << "TrackingCircle: bad_any_cast in post_process_result\n";
                    }
                }
            },
            [](const PlannerConfig& cfg, double t_abs, const TargetSnapshot& tgt_snap) {
                // Return ct so create() can use it AND post_process_result can use it.
                TrackingCircleOCP::TrackingCircleExtra ex;
                ex.tgt.center.setZero();
                ex.tgt.R = 0.0;
                ex.tgt.omega = 0.0;
                ex.tgt.phi0 = 0.0;
                if (tgt_snap.valid) {
                    ex.tgt.center << cfg.circle_center_x, cfg.circle_center_y, cfg.circle_center_z;
                    ex.tgt.R = cfg.circle_R;
                    ex.tgt.omega = cfg.circle_omega;
                    ex.tgt.phi0 = cfg.circle_phi0 - cfg.circle_omega * cfg.t_start_abs;
                }
                ex.t_abs = t_abs;
                return std::any(ex); 
            },
            [](planner_logging::SolveLogMeta& meta, const std::any& extra_params, const std::any& /*runtime_cfg*/) {
                try {
                    auto ex = std::any_cast<TrackingCircleOCP::TrackingCircleExtra>(extra_params);
                    meta.circle_center = ex.tgt.center;
                    meta.circle_radius = ex.tgt.R;
                    meta.circle_omega = ex.tgt.omega;
                    meta.circle_phi0 = ex.tgt.phi0;
                    meta.circle_t_abs = ex.t_abs;
                } catch (const std::bad_any_cast&) {}
            },
            TrackingCircleOCP::getSolverParams,
            [](const OCPCreateArgs& a) {
                if (a.extra.has_value()) {
                   try {
                       auto ex = std::any_cast<TrackingCircleOCP::TrackingCircleExtra>(a.extra);
                       return TrackingCircleOCP::create(a.current_state, ex.tgt, ex.t_abs);
                   } catch (const std::bad_any_cast&) {}
                }
                return TrackingCircleOCP::create(a.current_state, target_models::getDefaultCircularTarget(), a.t_abs);
            }
        }},
        {"tracking_circle_target", {
            "tracking_circle_target",
            TrackingCircleTargetOCP::TH_INIT,
            TrackingCircleTargetOCP::DEFAULT_N_REPLAY,
            TrackingCircleTargetOCP::MASS,
            OCPDescriptor::WarmStart::Shift,
            true, // needs_target_trajectory
            [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
                Eigen::VectorXd xr = x;
                xr.segment(0, 3) -= t.position;
                xr.segment(3, 3) -= t.velocity;
                return xr;
            },
            [](const TargetSnapshot& t, double now_sec, double max_age) {
                if (t.odom_stamp_sec <= 0.0) {
                    return false;
                }
                const double age = now_sec - t.odom_stamp_sec;
                const double kClockTol = 0.001;
                return !(age < -kClockTol || age >= max_age);
            },
            [](SolverResult& r, const TargetSnapshot& t) {
                // The OCP state is relative by design; planner-side plumbing
                // reconstructs absolute target motion for publications/logging.
                r.is_relative_plan = true;
                if (t.odom_stamp_sec > 0.0) {
                    r.target_snapshot_pos = t.position;
                    r.target_snapshot_vel = t.velocity;
                } else {
                    r.target_snapshot_pos.setZero();
                    r.target_snapshot_vel.setZero();
                }
                r.target_snapshot_acc = t.acceleration;
            },
            [](const PlannerConfig& cfg, double t_abs, const TargetSnapshot& /*tgt_snap*/) {
                TrackingCircleTargetOCP::TrackingCircleTargetExtra ex;
                ex.t_abs = t_abs;

                if (cfg.target_accel_buffer.has_value()) {
                    ex.buf = cfg.target_accel_buffer.value();
                }

                return std::any(ex);
            },
            nullptr, // prepare_log_meta
            TrackingCircleTargetOCP::getSolverParams,
            [](const OCPCreateArgs& a) {
                if (a.extra.has_value()) {
                   try {
                       auto ex = std::any_cast<TrackingCircleTargetOCP::TrackingCircleTargetExtra>(a.extra);
                       return TrackingCircleTargetOCP::create(a.current_state, ex.buf, ex.t_abs);
                   } catch (const std::bad_any_cast&) {}
                }
                return TrackingCircleTargetOCP::create(
                    a.current_state,
                    target_models::TargetAccelBuffer{},
                    a.t_abs);
            }
        }}
    };
    return table;
}

inline const OCPDescriptor& getDescriptor(const std::string& ocp_type) {
    const auto& table = getTable();
    auto it = table.find(ocp_type);
    if (it == table.end()) {
        throw std::runtime_error("Unknown OCP type: " + ocp_type);
    }
    return it->second;
}

inline double getDT(const std::string& ocp_type) { return getDescriptor(ocp_type).dt; }
inline int getDefaultNReplay(const std::string& ocp_type) { return getDescriptor(ocp_type).default_n_replay; }
inline double getDefaultMassKg(const std::string& ocp_type) { return getDescriptor(ocp_type).default_mass_kg; }
inline Param getSolverParams(const std::string& ocp_type) { return getDescriptor(ocp_type).getSolverParams(); }
inline std::shared_ptr<OptimalControlProblem<double>> create(const std::string& ocp_type, const OCPCreateArgs& args) {
    return getDescriptor(ocp_type).create(args);
}

} // namespace OCPRegistry
