#pragma once

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include "core/target_snapshot.hpp"
#include "core/ocp_descriptor.hpp"
#include "platform/crazyflie.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
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

        mass_kg_      = mass_kg;
        state_dim_    = state_dim;
        state_names_  = state_names;
        control_names_ = control_names;
        command_mode_ = command_mode;
        extract_actual_state_row_ = extract_actual_state;

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

        // ── commanded_state.csv (CmdFullState only) ──────────────────────────
        commanded_state_log_.open(folder + "/commanded_state.csv");
        if (commanded_state_log_.is_open()) {
            commanded_state_log_ << "timestamp,solve_num";
            if (!log_state_headers.empty() && log_state_headers.size() >= static_cast<size_t>(state_dim_)) {
                for (int i = 0; i < state_dim_; ++i) commanded_state_log_ << "," << log_state_headers[i];
            } else {
                for (int i = 0; i < state_dim_; ++i) commanded_state_log_ << ",x" << i;
            }
            commanded_state_log_ << ",fz,mx,my,mz,acc_x,acc_y,acc_z\n";
        }

        // ── commanded_hover_state.csv (CmdBodyRate + platform=crazyflie) ────────
        commanded_hover_log_.open(folder + "/commanded_hover_state.csv");
        if (commanded_hover_log_.is_open()) {
            commanded_hover_log_
                << "timestamp,solve_num,vx,vy,z_distance,yaw_rate\n";
        }

        // ── commanded_bodyrate_state.csv (CmdBodyRate + platform=mavros/generic) ─
        commanded_bodyrate_log_.open(folder + "/commanded_bodyrate_state.csv");
        if (commanded_bodyrate_log_.is_open()) {
            commanded_bodyrate_log_
                << "timestamp,solve_num,T_ms2,omega_x,omega_y,omega_z\n";
        }

        // ── actual_state.csv ─────────────────────────────────────────────────
        actual_state_log_.open(folder + "/actual_state.csv");
        if (actual_state_log_.is_open()) {
            actual_state_log_ << "timestamp,solve_num,coord_mode";
            for (const auto& h : log_state_headers) {
                actual_state_log_ << "," << h;
            }
            actual_state_log_ << "\n";
        }

        initialized_ = all_solves_log_.is_open() &&
                       commanded_state_log_.is_open() &&
                       commanded_hover_log_.is_open() &&
                       commanded_bodyrate_log_.is_open() &&
                       actual_state_log_.is_open();
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

    void logActualState(const Eigen::VectorXd& state,
                        int solve_num,
                        const Eigen::Vector3d& target_pos = Eigen::Vector3d::Zero(),
                        const Eigen::Vector3d& target_vel = Eigen::Vector3d::Zero(),
                        const std::string& coord_mode = "absolute") {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !actual_state_log_.is_open()) {
            return;
        }

        actual_state_log_ << std::fixed << std::setprecision(6)
                          << wallTimeSec() << "," << solve_num << "," << coord_mode;

        if (extract_actual_state_row_) {
            TargetSnapshot ts;
            ts.position = target_pos;
            ts.velocity = target_vel;
            auto row = extract_actual_state_row_(state, ts, coord_mode);
            for (double v : row) {
                actual_state_log_ << "," << v;
            }
        }
        actual_state_log_ << "\n";
        actual_state_log_.flush();
    }

    /// Log hover command [vx, vy, z_cmd, yaw_rate] (CmdBodyRate + platform=crazyflie).
    void logCommandedHoverState(const std::array<float, 4>& cmd, int solve_num) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !commanded_hover_log_.is_open()) {
            return;
        }
        commanded_hover_log_ << std::fixed << std::setprecision(6)
                             << wallTimeSec() << "," << solve_num << ","
                             << cmd[0] << "," << cmd[1] << ","
                             << cmd[2] << "," << cmd[3] << "\n";
        commanded_hover_log_.flush();
    }

    /// Log body-rate command [T_ms2, ωx, ωy, ωz] (CmdBodyRate + platform=mavros or generic).
    void logCommandedBodyRateState(const Eigen::VectorXd& u, int solve_num) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !commanded_bodyrate_log_.is_open()) return;
        commanded_bodyrate_log_ << std::fixed << std::setprecision(6)
                                << wallTimeSec() << "," << solve_num << ","
                                << (u.size() > 0 ? u(0) : 0.0) << ","   // T_ms2
                                << (u.size() > 1 ? u(1) : 0.0) << ","   // omega_x
                                << (u.size() > 2 ? u(2) : 0.0) << ","   // omega_y
                                << (u.size() > 3 ? u(3) : 0.0)          // omega_z
                                << "\n";
        commanded_bodyrate_log_.flush();
    }

    /// Log full commanded state (CmdFullState mode only — skipped if CmdHover).
    void logCommandedState(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int solve_num) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !commanded_state_log_.is_open()) {
            return;
        }
        if (command_mode_ != OCPDescriptor::CommandMode::CmdFullState) {
            return;  // self-filter: only write for CmdFullState OCPs
        }

        const double fz = (control.size() >= 1) ? control(0) : 0.0;
        const Eigen::Vector3d acc = platform::crazyflie::computeAcc(state, fz, mass_kg_);

        commanded_state_log_ << std::fixed << std::setprecision(6)
                             << wallTimeSec() << "," << solve_num;
        for (int i = 0; i < state_dim_; ++i) {
            commanded_state_log_ << "," << state(i);
        }
        if (control.size() >= 4) {
            commanded_state_log_ << ","
                                 << control(0) << ","
                                 << control(1) << ","
                                 << control(2) << ","
                                 << control(3);
        } else {
            commanded_state_log_ << ",0,0,0,0";
        }
        commanded_state_log_ << "," << acc.x() << "," << acc.y() << "," << acc.z() << "\n";
        commanded_state_log_.flush();
    }

private:
    static double wallTimeSec() {
        using SteadyClock = std::chrono::steady_clock;
        return std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();
    }

    mutable std::mutex mutex_;
    bool initialized_ = false;
    double mass_kg_ = 0.0282;
    int state_dim_ = 13;
    std::vector<std::string> state_names_;
    std::vector<std::string> control_names_;
    OCPDescriptor::CommandMode command_mode_ = OCPDescriptor::CommandMode::CmdFullState;
    std::function<std::vector<double>(const Eigen::VectorXd&, const TargetSnapshot&, const std::string&)> extract_actual_state_row_;
    std::string log_folder_;
    std::ofstream all_solves_log_;
    std::ofstream commanded_state_log_;
    std::ofstream commanded_hover_log_;
    std::ofstream commanded_bodyrate_log_;
    std::ofstream actual_state_log_;
};

}  // namespace planner_logging
