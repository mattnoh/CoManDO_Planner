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

#include <string>
#include <stdexcept>
#include <memory>
#include <map>
#include <functional>
#include <any>

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "rclcpp/rclcpp.hpp"

// Forward declare struct so we can use it in OCPRegistry
struct TargetSnapshot {
    bool valid = false;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    rclcpp::Time odom_timestamp{0, 0, RCL_ROS_TIME};
    rclcpp::Time accel_timestamp{0, 0, RCL_ROS_TIME};
};

struct SolverResult {
    bool success = false;
    Eigen::VectorXd next_state;
    std::vector<Eigen::VectorXd> state_trajectory;
    std::vector<Eigen::VectorXd> control_trajectory;
    double solve_time_ms = 0.0;
    int solve_iters = 0;
    std::chrono::steady_clock::time_point solve_timestamp;

    bool is_relative_plan = false;
    Eigen::Vector3d target_snapshot_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_acc = Eigen::Vector3d::Zero();
};

// Note: TrackingCircleOCP::CircularTarget must be in ocp_tracking_circle.hpp

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

    // Callbacks — nullptr = not needed
    std::function<Eigen::VectorXd(
        const Eigen::VectorXd& x_drone,
        const TargetSnapshot&)>              transform_state;

    std::function<bool(const TargetSnapshot&,
        const rclcpp::Time&, double max_age)> validate_target;

    std::function<void(SolverResult&,
        const TargetSnapshot&)>              post_process_result;

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
            nullptr, // transform_state
            nullptr, // validate_target
            nullptr, // post_process_result
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
            nullptr, // transform_state
            nullptr, // validate_target
            nullptr, // post_process_result
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
            [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
                Eigen::VectorXd xr = Eigen::VectorXd::Zero(13);
                xr.segment(0, 3) = x.segment(0, 3) - t.position;
                xr.segment(3, 3) = x.segment(3, 3) - t.velocity;
                xr.segment(6, 7) = x.segment(6, 7);
                return xr;
            },
            [](const TargetSnapshot& t, const rclcpp::Time& now, double age) {
                return t.valid; // state_monitor already handles freshness checks
            },
            [](SolverResult& r, const TargetSnapshot& t) {
                r.is_relative_plan = true;
                r.target_snapshot_pos = t.position;
                r.target_snapshot_vel = t.velocity;
                r.target_snapshot_acc = t.acceleration;
            },
            StateswitchOCP::getSolverParams,
            [](const OCPCreateArgs& a) {
                return StateswitchOCP::create(a.current_state, a.terminal_state,
                                              a.prev_U, a.prev_X, a.target_accel, a.prev_K);
            }
        }},
        {"tracking_circle", {
            "tracking_circle",
            TrackingCircleOCP::TH_INIT,
            TrackingCircleOCP::DEFAULT_N_REPLAY,
            TrackingCircleOCP::MASS,
            OCPDescriptor::WarmStart::Shift,
            [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
                Eigen::VectorXd xr = Eigen::VectorXd::Zero(13);
                xr.segment(0, 3) = x.segment(0, 3) - t.position;
                xr.segment(3, 3) = x.segment(3, 3) - t.velocity;
                xr.segment(6, 7) = x.segment(6, 7);
                return xr;
            },
            [](const TargetSnapshot& t, const rclcpp::Time& now, double age) {
                return t.valid;
            },
            [](SolverResult& r, const TargetSnapshot& t) {
                r.is_relative_plan = true;
                r.target_snapshot_pos = t.position;
                r.target_snapshot_vel = t.velocity;
                // Tracking circle might not need acceleration in post processing since it uses TV
            },
            TrackingCircleOCP::getSolverParams,
            [](const OCPCreateArgs& a) {
                TrackingCircleOCP::CircularTarget default_ct;
                const TrackingCircleOCP::CircularTarget* ct_ptr = &default_ct;
                if (a.extra.has_value()) {
                    ct_ptr = std::any_cast<const TrackingCircleOCP::CircularTarget>(&a.extra);
                }
                return TrackingCircleOCP::create(a.current_state, ct_ptr ? *ct_ptr : default_ct, a.t_abs);
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
