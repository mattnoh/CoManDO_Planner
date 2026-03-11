/// @file logger.hpp
/// @brief CoManDO logging utility.
///
/// Keeps EXACTLY the same CSV columns and folder structure as the original
/// planner_node.cpp — just moved here so every node can use it without
/// copy-pasting.  No ROS dependencies; plain std::cout / std::cerr.
///
/// Usage:
///   CommandoLogger log;
///   log.setup(drone_name, ocp_type, mode, solver_type);
///   log.logSolveTrajectory(X, U, solve_ms, ocp_dt);
///   log.logActualState(state);
///   log.logCommandedState(x_cmd, u_cmd);
///   log.logPublishedCommand(x_cmd, acc);

#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ── Console print helpers (same [Tag] style as before) ───────────────────────
#define COMANDO_INFO(tag, fmt, ...)  \
    do { printf("[" tag "] " fmt "\n", ##__VA_ARGS__); fflush(stdout); } while(0)

#define COMANDO_WARN(tag, fmt, ...)  \
    do { fprintf(stderr, "[WARN][" tag "] " fmt "\n", ##__VA_ARGS__); } while(0)

#define COMANDO_ERROR(tag, fmt, ...) \
    do { fprintf(stderr, "[ERROR][" tag "] " fmt "\n", ##__VA_ARGS__); } while(0)

// ═══════════════════════════════════════════════════════════════════════════════
class CommandoLogger {
public:
    // ── Setup — call once before any log*() calls ─────────────────────────────
    void setup(const std::string& drone_name,
               const std::string& ocp_type,
               const std::string& mode,
               const std::string& solver_type)
    {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::stringstream ts;
        ts << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");

        std::string folder = "./logs/" + drone_name + "_" + ocp_type
                           + "_" + mode + "_" + solver_type + "_" + ts.str();
        std::filesystem::create_directories(folder);
        log_folder_ = folder;

        commanded_state_log_.open(folder + "/commanded_state.csv");
        if (commanded_state_log_.is_open())
            commanded_state_log_
                << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,fz,mx,my,mz\n";

        actual_state_log_.open(folder + "/actual_state.csv");
        if (actual_state_log_.is_open())
            actual_state_log_
                << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";

        all_solves_log_.open(folder + "/all_solves.csv");
        if (all_solves_log_.is_open())
            all_solves_log_
                << "solve_num,solve_time_ms,node,t,"
                   "x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,fz,mx,my,mz\n";

        published_commands_log_.open(folder + "/published_commands.csv");
        if (published_commands_log_.is_open())
            published_commands_log_
                << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,acc_x,acc_y,acc_z\n";

        initialized_ = true;
        std::cout << "[CommandoLogger] Logging to: " << folder << "\n";
    }

    bool isInitialized() const { return initialized_; }
    const std::string& folder() const { return log_folder_; }

    // ── Per-solve: full trajectory ────────────────────────────────────────────
    void logSolveTrajectory(const std::vector<Eigen::VectorXd>& traj,
                            const std::vector<Eigen::VectorXd>& ctrls,
                            double solve_time_ms,
                            double ocp_dt)
    {
        if (!all_solves_log_.is_open()) return;
        for (int i = 0; i < (int)traj.size(); ++i) {
            const auto& s = traj[i];
            if (s.size() < 13) continue;
            all_solves_log_ << std::fixed << std::setprecision(6)
                << solve_count_ << "," << solve_time_ms << ","
                << i << "," << (i * ocp_dt);
            for (int j = 0; j < 13; ++j) all_solves_log_ << "," << s(j);
            if (i < (int)ctrls.size() && ctrls[i].size() >= 4)
                all_solves_log_ << "," << ctrls[i](0) << "," << ctrls[i](1)
                                << "," << ctrls[i](2) << "," << ctrls[i](3);
            else
                all_solves_log_ << ",0,0,0,0";
            all_solves_log_ << "\n";
        }
        all_solves_log_.flush();
        ++solve_count_;
    }

    // ── Per-replay-tick: actual platform state ────────────────────────────────
    void logActualState(const Eigen::VectorXd& state)
    {
        if (!actual_state_log_.is_open()) return;
        actual_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) actual_state_log_ << "," << state(i);
        actual_state_log_ << "\n";
        actual_state_log_.flush();
    }

    // ── Per-replay-tick: commanded state + control ────────────────────────────
    void logCommandedState(const Eigen::VectorXd& s, const Eigen::VectorXd& u)
    {
        if (!commanded_state_log_.is_open() || s.size() < 13) return;
        commanded_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) commanded_state_log_ << "," << s(i);
        if (u.size() >= 4)
            commanded_state_log_ << "," << u(0) << "," << u(1)
                                 << "," << u(2) << "," << u(3);
        else
            commanded_state_log_ << ",0,0,0,0";
        commanded_state_log_ << "\n";
        commanded_state_log_.flush();
    }

    // ── Per-replay-tick: what was actually sent to the platform ───────────────
    void logPublishedCommand(const Eigen::VectorXd& s,
                             const Eigen::Vector3d& acc)
    {
        if (!published_commands_log_.is_open()) return;
        published_commands_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) published_commands_log_ << "," << s(i);
        published_commands_log_ << "," << acc(0) << "," << acc(1)
                                << "," << acc(2) << "\n";
        published_commands_log_.flush();
    }

private:
    static double wallTimeSec() {
        using Clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(
            Clock::now().time_since_epoch()).count();
    }

    bool         initialized_ = false;
    std::string  log_folder_;
    std::ofstream commanded_state_log_;
    std::ofstream actual_state_log_;
    std::ofstream all_solves_log_;
    std::ofstream published_commands_log_;
    int           solve_count_ = 0;
};