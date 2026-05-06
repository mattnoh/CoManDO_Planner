#pragma once

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include "planner_core/types.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace planner_logging {

struct SolveLogMeta {
    int solve_num = 0;
    double solve_time_ms = 0.0;
    int solve_iters = 0;
    bool is_relative_plan = false;
    double ocp_dt = 0.05;
    /// Coordinate frame of the solved trajectory.
    /// "absolute"          — world-frame absolute coordinates (hover, landing, …)
    /// "absolute_relative" — world-frame relative to target (stateswitch, tracking_circle*)
    /// "body_relative"     — body-frame relative to target  (tracking_cmdhover)
    std::string coord_mode = "absolute";
    Eigen::Vector3d target_snapshot_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_snapshot_acc = Eigen::Vector3d::Zero();
    Eigen::Vector4d target_snapshot_quat = Eigen::Vector4d(1,0,0,0); // [qw,qx,qy,qz]
    Eigen::Vector3d target_snapshot_omega = Eigen::Vector3d::Zero(); // Ω_N
    Eigen::Vector3d target_snapshot_beta = Eigen::Vector3d::Zero();  // β_N

    // Optional node-wise reconstructed target trajectory (world frame).
    std::vector<Eigen::Vector3d> target_world_pos_trajectory;
    std::vector<Eigen::Vector3d> target_world_vel_trajectory;
};

struct SolverEventLogRow {
    int solve_num = -1;
    std::string event;
    std::string reason;
    std::string coord_mode;
    std::string x0_source;
    double solve_time_ms = std::numeric_limits<double>::quiet_NaN();
    double solve_iters = std::numeric_limits<double>::quiet_NaN();
    double constraint_error = std::numeric_limits<double>::quiet_NaN();
    double active_elapsed_now_sec = std::numeric_limits<double>::quiet_NaN();
    double activation_elapsed_sec = std::numeric_limits<double>::quiet_NaN();
    double activation_wall_time_sec = std::numeric_limits<double>::quiet_NaN();
    double solve_lead_sec = std::numeric_limits<double>::quiet_NaN();
    double solve_finish_late_by_sec = std::numeric_limits<double>::quiet_NaN();
    double handoff_pos_err = std::numeric_limits<double>::quiet_NaN();
    double handoff_vel_err = std::numeric_limits<double>::quiet_NaN();
    double replan_delay_sec = std::numeric_limits<double>::quiet_NaN();
    double state_jump_norm = std::numeric_limits<double>::quiet_NaN();
    double control_jump_norm = std::numeric_limits<double>::quiet_NaN();
    double hover_z_jump = std::numeric_limits<double>::quiet_NaN();
    std::map<std::string, double> extra_values;
};

class CsvLogger {
public:
    /// Initialize logger.
    /// @param state_names     Column names for raw OCP state vector (from descriptor().state_names)
    /// @param control_names   Column names for raw OCP control vector (from descriptor().control_names)
    /// @param command_mode    Active command mode — determines which commanded log gets written
    bool initialize(const std::string& drone_name,
                    const std::string& ocp_type,
                    const std::string& mode,
                    const std::string& solver_type,
                    const rclcpp::Logger& ros_logger,
                    double mass_kg,
                    int state_dim,
                    const std::vector<std::string>& state_names,
                    const std::vector<std::string>& control_names,
                    const std::vector<std::string>& log_state_headers,
                    OCPDescriptor::CommandMode command_mode,
                    std::function<std::vector<double>(const Eigen::VectorXd&, const TargetSnapshot&, const std::string&)> extract_actual_state) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (initialized_) {
            return true;
        }

        (void)log_state_headers;
        (void)command_mode;
        (void)extract_actual_state;
        mass_kg_      = mass_kg;
        state_dim_    = state_dim;
        state_names_  = state_names;
        control_names_ = control_names;

        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ts;
        ts << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        std::string folder = "./logs/" + drone_name + "_" + ocp_type + "_" +
                             mode + "_" + solver_type + "_" + ts.str();
        std::filesystem::create_directories(folder);
        log_folder_ = folder;

        // ── all_solves.csv ───────────────────────────────────────────────────
        all_solves_log_.open(folder + "/all_solves.csv");
        if (all_solves_log_.is_open()) {
            // Common prefix
            all_solves_log_ << "solve_num,solve_time_ms,solve_iters,coord_mode,node,t,theta";

            // Raw OCP state columns (named from dynamics)
            for (size_t i = 0; i < state_names_.size(); ++i) {
                all_solves_log_ << "," << state_names_[i];
            }
            // Extra state elements beyond named list (e.g. DT appended by variable-dt dynamics)
            // These get auto-named x{N}, x{N+1}, ...
            // We don't know the actual solver vector size at header time, so we write extras
            // dynamically per-row. Header is therefore open-ended — we'll handle header
            // for those via the first row. Actually: we pre-extend the header with
            // a placeholder for DT used by variable-dt OCPs (x[state_dim] = DT).
            // For simplicity: always emit exactly state_names_.size() state cols +
            // one extra "DT" col for variable-dt OCPs (detected if solver vec > state_dim).

            // Raw OCP control columns
            for (size_t i = 0; i < control_names_.size(); ++i) {
                all_solves_log_ << "," << control_names_[i];
            }
            // Extra control (e.g. theta timestep for variable-dt: u[4])
            // Always emit "theta" col in the prefix already; extra u elements would be u4+ only.

            // Target at the end
            all_solves_log_ << ",tgt_x,tgt_y,tgt_z,tgt_vx,tgt_vy,tgt_vz"
                           << ",tgt_ax,tgt_ay,tgt_az"
                           << ",tgt_qw,tgt_qx,tgt_qy,tgt_qz"
                           << ",tgt_wx,tgt_wy,tgt_wz"
                           << ",tgt_alfx,tgt_alfy,tgt_alfz\n";
        }

        // ── solver_events.csv ────────────────────────────────────────────────
        solver_events_log_.open(folder + "/solver_events.csv");
        if (solver_events_log_.is_open()) {
            solver_events_log_
                << "timestamp,solve_num,event,reason,coord_mode,x0_source"
                << ",solve_time_ms,solve_iters,constraint_error"
                << ",active_elapsed_now_sec,activation_elapsed_sec,activation_wall_time_sec"
                << ",solve_lead_sec,solve_finish_late_by_sec"
                << ",handoff_pos_err,handoff_vel_err,replan_delay_sec"
                << ",state_jump_norm,control_jump_norm,hover_z_jump"
                << ",extra_values\n";
        }

        initialized_ = all_solves_log_.is_open() &&
                       solver_events_log_.is_open();
        RCLCPP_INFO(ros_logger, "Logging to: %s", folder.c_str());
        return initialized_;
    }

    bool isInitialized() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return initialized_;
    }

    /// Log raw OCP solve trajectory to all_solves.csv.
    /// Each node row: common prefix | raw x[i] | raw u[i] | tgt_x,tgt_y,tgt_z,tgt_vx,tgt_vy,tgt_vz
    void logSolveTrajectory(const std::vector<Eigen::VectorXd>& traj,
                            const std::vector<Eigen::VectorXd>& ctrls,
                            const SolveLogMeta& meta) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !all_solves_log_.is_open()) {
            return;
        }

        const bool has_reconstructed_target =
            (meta.target_world_pos_trajectory.size() == traj.size()) &&
            (meta.target_world_vel_trajectory.size() == traj.size());

        for (int i = 0; i < static_cast<int>(traj.size()); ++i) {
            const auto& s = traj[i];
            if (s.size() == 0) continue;

            // t: cumulative node time
            double t_node = i * meta.ocp_dt;
            if (s.size() > static_cast<Eigen::Index>(state_dim_)) {
                // Variable-dt: DT stored at s[state_dim]
                t_node = s(state_dim_);
            }

            // theta: timestep (stored in u[control_names_.size()] for variable-dt)
            double theta = meta.ocp_dt;
            Eigen::VectorXd control = Eigen::VectorXd::Zero(control_names_.size());
            if (i < static_cast<int>(ctrls.size())) {
                control = ctrls[i];
                if (control.size() > static_cast<Eigen::Index>(control_names_.size())) {
                    // Variable-dt: theta stored at u[control_names_.size()]
                    theta = control(static_cast<int>(control_names_.size()));
                }
            }

            // Target position/velocity at this node
            Eigen::Vector3d tgt_p = meta.target_snapshot_pos;
            Eigen::Vector3d tgt_v = meta.target_snapshot_vel;

            if (has_reconstructed_target) {
                tgt_p = meta.target_world_pos_trajectory[i];
                tgt_v = meta.target_world_vel_trajectory[i];
            } else if (meta.is_relative_plan) {
                tgt_p = meta.target_snapshot_pos +
                        meta.target_snapshot_vel * t_node +
                        0.5 * meta.target_snapshot_acc * t_node * t_node;
                tgt_v = meta.target_snapshot_vel +
                        meta.target_snapshot_acc * t_node;
            }

            all_solves_log_ << std::fixed << std::setprecision(6)
                            << meta.solve_num << ","
                            << meta.solve_time_ms << ","
                            << meta.solve_iters << ","
                            << meta.coord_mode << ","
                            << i << "," << t_node << "," << theta;

            // Raw OCP state x[i] — write exactly as many elements as named
            const int nx = static_cast<int>(state_names_.size());
            for (int j = 0; j < nx; ++j) {
                if (j < s.size()) {
                    all_solves_log_ << "," << s(j);
                } else {
                    all_solves_log_ << ",0";
                }
            }

            // Raw OCP control u[i] — write exactly as many elements as named
            const int nu = static_cast<int>(control_names_.size());
            for (int j = 0; j < nu; ++j) {
                if (j < control.size()) {
                    all_solves_log_ << "," << control(j);
                } else {
                    all_solves_log_ << ",0";
                }
            }

            // Target columns at end
            all_solves_log_ << ","
                            << tgt_p.x() << "," << tgt_p.y() << "," << tgt_p.z() << ","
                            << tgt_v.x() << "," << tgt_v.y() << "," << tgt_v.z() << ","
                            << meta.target_snapshot_acc.x() << "," << meta.target_snapshot_acc.y() << "," << meta.target_snapshot_acc.z() << ","
                            << meta.target_snapshot_quat(0) << "," << meta.target_snapshot_quat(1) << "," << meta.target_snapshot_quat(2) << "," << meta.target_snapshot_quat(3) << ","
                            << meta.target_snapshot_omega.x() << "," << meta.target_snapshot_omega.y() << "," << meta.target_snapshot_omega.z() << ","
                            << meta.target_snapshot_beta.x() << "," << meta.target_snapshot_beta.y() << "," << meta.target_snapshot_beta.z()
                            << "\n";
        }
        all_solves_log_.flush();
    }

    void logSolverEvent(const SolverEventLogRow& row) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !solver_events_log_.is_open()) {
            return;
        }

        solver_events_log_ << std::fixed << std::setprecision(6)
                           << wallTimeSec() << ","
                           << row.solve_num << ","
                           << csvEscape(row.event) << ","
                           << csvEscape(row.reason) << ","
                           << csvEscape(row.coord_mode) << ","
                           << csvEscape(row.x0_source) << ","
                           << row.solve_time_ms << ","
                           << row.solve_iters << ","
                           << row.constraint_error << ","
                           << row.active_elapsed_now_sec << ","
                           << row.activation_elapsed_sec << ","
                           << row.activation_wall_time_sec << ","
                           << row.solve_lead_sec << ","
                           << row.solve_finish_late_by_sec << ","
                           << row.handoff_pos_err << ","
                           << row.handoff_vel_err << ","
                           << row.replan_delay_sec << ","
                           << row.state_jump_norm << ","
                           << row.control_jump_norm << ","
                           << row.hover_z_jump << ","
                           << csvEscape(formatExtraValues(row.extra_values))
                           << "\n";
        solver_events_log_.flush();
    }

private:
    static double wallTimeSec() {
        using SteadyClock = std::chrono::steady_clock;
        return std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();
    }

    static std::string csvEscape(const std::string& s) {
        if (s.find_first_of(",\"\n\r") == std::string::npos) {
            return s;
        }
        std::string out = "\"";
        for (char c : s) {
            if (c == '"') out += "\"\"";
            else out += c;
        }
        out += "\"";
        return out;
    }

    static std::string formatExtraValues(const std::map<std::string, double>& values) {
        std::ostringstream ss;
        bool first = true;
        for (const auto& [key, value] : values) {
            if (!first) ss << ";";
            first = false;
            ss << key << "=" << value;
        }
        return ss.str();
    }

    mutable std::mutex mutex_;
    bool initialized_ = false;
    double mass_kg_ = 0.0282;
    int state_dim_ = 13;
    std::vector<std::string> state_names_;
    std::vector<std::string> control_names_;
    std::string log_folder_;
    std::ofstream all_solves_log_;
    std::ofstream solver_events_log_;
};

}  // namespace planner_logging
