#pragma once

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

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

    // Optional node-wise reconstructed target trajectory (world frame).
    std::vector<Eigen::Vector3d> target_world_pos_trajectory;
    std::vector<Eigen::Vector3d> target_world_vel_trajectory;
};

class CsvLogger {
public:
    bool initialize(const std::string& drone_name,
                    const std::string& ocp_type,
                    const std::string& mode,
                    const std::string& solver_type,
                    const rclcpp::Logger& ros_logger,
                    double mass_kg) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (initialized_) {
            return true;
        }

        mass_kg_ = mass_kg;

        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ts;
        ts << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        std::string folder = "./logs/" + drone_name + "_" + ocp_type + "_" +
                             mode + "_" + solver_type + "_" + ts.str();
        std::filesystem::create_directories(folder);
        log_folder_ = folder;

        all_solves_log_.open(folder + "/all_solves.csv");
        if (all_solves_log_.is_open()) {
            all_solves_log_
                << "solve_num,solve_time_ms,solve_iters,coord_mode,node,t,theta,"
                << "tgt_x,tgt_y,tgt_z,tgt_vx,tgt_vy,tgt_vz,"
                << "abs_x,abs_y,abs_z,abs_vx,abs_vy,abs_vz,abs_qw,abs_qx,abs_qy,abs_qz,abs_wx,abs_wy,abs_wz,"
                << "fz,mx,my,mz,"
                << "rel_x,rel_y,rel_z,rel_vx,rel_vy,rel_vz,rel_qw,rel_qx,rel_qy,rel_qz,rel_wx,rel_wy,rel_wz\n";
        }

        commanded_state_log_.open(folder + "/commanded_state.csv");
        if (commanded_state_log_.is_open()) {
            commanded_state_log_
                << "timestamp,solve_num,"
                << "x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
                << "fz,mx,my,mz,"
                << "acc_x,acc_y,acc_z\n";
        }

        commanded_hover_log_.open(folder + "/commanded_hover_state.csv");
        if (commanded_hover_log_.is_open()) {
            commanded_hover_log_
                << "timestamp,solve_num,vx,vy,z_distance,yaw_rate\n";
        }

        actual_state_log_.open(folder + "/actual_state.csv");
        if (actual_state_log_.is_open()) {
            actual_state_log_
                << "timestamp,solve_num,coord_mode,"
                << "x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
                << "abs_x,abs_y,abs_z,abs_vx,abs_vy,abs_vz,abs_qw,abs_qx,abs_qy,abs_qz,abs_wx,abs_wy,abs_wz,"
                << "rel_x,rel_y,rel_z,rel_vx,rel_vy,rel_vz,rel_qw,rel_qx,rel_qy,rel_qz,rel_wx,rel_wy,rel_wz,"
                << "tgt_x,tgt_y,tgt_z,tgt_vx,tgt_vy,tgt_vz\n";
        }

        initialized_ = all_solves_log_.is_open() &&
                       commanded_state_log_.is_open() &&
                       commanded_hover_log_.is_open() &&
                       actual_state_log_.is_open();
        RCLCPP_INFO(ros_logger, "Logging to: %s", folder.c_str());
        return initialized_;
    }

    bool isInitialized() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return initialized_;
    }

    void logSolveTrajectory(const std::vector<Eigen::VectorXd>& traj,
                            const std::vector<Eigen::VectorXd>& ctrls,
                            const SolveLogMeta& meta) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !all_solves_log_.is_open()) {
            return;
        }

        const double nan = std::numeric_limits<double>::quiet_NaN();
        const bool has_reconstructed_target =
            (meta.target_world_pos_trajectory.size() == traj.size()) &&
            (meta.target_world_vel_trajectory.size() == traj.size());

        for (int i = 0; i < static_cast<int>(traj.size()); ++i) {
            const auto& s = traj[i];
            if (s.size() < 13) {
                continue;
            }

            double t_node = i * meta.ocp_dt;
            if (s.size() > 13) {
                t_node = s(13);
            }

            Eigen::Vector3d tgt_p = meta.target_snapshot_pos;
            Eigen::Vector3d tgt_v = meta.target_snapshot_vel;

            if (has_reconstructed_target) {
                tgt_p = meta.target_world_pos_trajectory[i];
                tgt_v = meta.target_world_vel_trajectory[i];
            } else if (meta.is_relative_plan) {
                // Propagate target using constant-acceleration assumption matching Quad6DOFVarTimeRelative
                tgt_p = meta.target_snapshot_pos +
                        meta.target_snapshot_vel * t_node +
                        0.5 * meta.target_snapshot_acc * t_node * t_node;
                tgt_v = meta.target_snapshot_vel +
                        meta.target_snapshot_acc * t_node;
            }

            Eigen::VectorXd x_abs = Eigen::VectorXd::Zero(13);
            Eigen::VectorXd x_rel = Eigen::VectorXd::Constant(13, nan);

            if (meta.is_relative_plan) {
                x_rel = s.head(13);
                x_abs.segment(0, 3) = tgt_p + x_rel.segment(0, 3);
                x_abs.segment(3, 3) = tgt_v + x_rel.segment(3, 3);
                x_abs.segment(6, 7) = x_rel.segment(6, 7);
            } else {
                x_abs = s.head(13);
                x_rel.segment(0, 3) = x_abs.segment(0, 3) - tgt_p;
                x_rel.segment(3, 3) = x_abs.segment(3, 3) - tgt_v;
                x_rel.segment(6, 7) = x_abs.segment(6, 7);
            }

            double theta = meta.ocp_dt;
            if (i < static_cast<int>(ctrls.size()) && ctrls[i].size() > 4) {
                theta = ctrls[i](4);
            }

            all_solves_log_ << std::fixed << std::setprecision(6)
                            << meta.solve_num << ","
                            << meta.solve_time_ms << ","
                            << meta.solve_iters << ","
                            << meta.coord_mode << ","
                            << i << "," << t_node << "," << theta << ","
                            << tgt_p.x() << ","
                            << tgt_p.y() << ","
                            << tgt_p.z() << ","
                            << tgt_v.x() << ","
                            << tgt_v.y() << ","
                            << tgt_v.z();

        for (int j = 0; j < 13; ++j) {
            all_solves_log_ << "," << x_abs(j);
        }

        if (i < static_cast<int>(ctrls.size()) && ctrls[i].size() >= 4) {
            all_solves_log_ << ","
            << ctrls[i](0) << ","
            << ctrls[i](1) << ","
            << ctrls[i](2) << ","
            << ctrls[i](3);
        } else {
            all_solves_log_ << ",0,0,0,0";
        }

        for (int j = 0; j < 13; ++j) {
            all_solves_log_ << "," << x_rel(j);
        }
        all_solves_log_ << "\n";
        }
        all_solves_log_.flush();
    }

    void logActualState(const Eigen::VectorXd& state,
                        int solve_num,
                        const Eigen::Vector3d& target_pos = Eigen::Vector3d::Zero(),
                        const Eigen::Vector3d& target_vel = Eigen::Vector3d::Zero(),
                        const std::string& coord_mode = "absolute") {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !actual_state_log_.is_open() || state.size() < 13) {
            return;
        }

        Eigen::VectorXd x_abs = state;
        Eigen::VectorXd x_rel = state;
        if (coord_mode == "relative") {
            x_abs.segment(0, 3) = state.segment(0, 3) + target_pos;
            x_abs.segment(3, 3) = state.segment(3, 3) + target_vel;
            x_rel = state;
        } else {
            x_rel.segment(0, 3) = state.segment(0, 3) - target_pos;
            x_rel.segment(3, 3) = state.segment(3, 3) - target_vel;
        }

        actual_state_log_ << std::fixed << std::setprecision(6)
                          << wallTimeSec() << "," << solve_num << "," << coord_mode;
        for (int i = 0; i < 13; ++i) {
            actual_state_log_ << "," << x_abs(i);
        }
        for (int i = 0; i < 13; ++i) {
            actual_state_log_ << "," << x_abs(i);
        }
        for (int i = 0; i < 13; ++i) {
            actual_state_log_ << "," << x_rel(i);
        }
        actual_state_log_ << ","
                          << target_pos.x() << ","
                          << target_pos.y() << ","
                          << target_pos.z() << ","
                          << target_vel.x() << ","
                          << target_vel.y() << ","
                          << target_vel.z();
        actual_state_log_ << "\n";
        actual_state_log_.flush();
    }

    /// Log the hover command fields actually sent to hardware (CmdHover mode only).
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

    void logCommandedState(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int solve_num) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!initialized_ || !commanded_state_log_.is_open() || state.size() < 13) {
            return;
        }

        const double fz = (control.size() >= 1) ? control(0) : 0.0;
        const Eigen::Vector3d acc = platform::crazyflie::computeAcc(state, fz, mass_kg_);

        commanded_state_log_ << std::fixed << std::setprecision(6)
                             << wallTimeSec() << "," << solve_num;
        for (int i = 0; i < 13; ++i) {
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
    std::string log_folder_;
    std::ofstream all_solves_log_;
    std::ofstream commanded_state_log_;
    std::ofstream commanded_hover_log_;
    std::ofstream actual_state_log_;
};

}  // namespace planner_logging
