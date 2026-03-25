/// @file planner_node.cpp
/// @brief CoManDO planner node - platform-agnostic MPC planner.
///
/// PLATFORM ABSTRACTION:
/// ─────────────────────
/// Platform-specific code (subscriptions, publishers, message formats)
/// is extracted into include/platform/*.hpp files. This node remains
/// platform-agnostic and delegates to the appropriate platform module.
///
/// LESSONS FROM CoManDO BRIDGE FAILURE (applied here):
/// ─────────────────────────────────────────────────────
/// 1. NO message_filters synchronization - direct callbacks update state
/// 2. NO intermediate bridge node - direct subscribe/publish to hardware
/// 3. Acceleration feedforward is computed and included in commands
/// 4. State mutex protects current_state_ across all callbacks
/// 5. Separate callback groups prevent sensor callbacks from blocking solver

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <Eigen/Dense>

#include "quadrotor_mpc.hpp"
#include "ocp_registry.hpp"
#include "platform/crazyflie.hpp"
#include "platform/px4.hpp"
#include "platform/target_tracker.hpp"
#include "state_monitor.hpp"

#include <chrono>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <atomic>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

// ─────────────────────────────────────────────────────────────────────────────
// Solver result struct
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// PlannerNode - platform-agnostic MPC planner
// ─────────────────────────────────────────────────────────────────────────────
class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        // Declare parameters
        this->declare_parameter("ocp_type", std::string("landing"));
        this->declare_parameter("drone_name", std::string("cf_1"));
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("platform", std::string("crazyflie"));
        this->declare_parameter("solver", std::string("alipddp"));
        this->declare_parameter("mode", std::string("mpc"));
        this->declare_parameter("hover_target_x", 0.0);
        this->declare_parameter("hover_target_y", 0.0);
        this->declare_parameter("hover_target_z", 1.0);
        this->declare_parameter("n_replay", 4);
        this->declare_parameter("mass_kg", 0.027);
        this->declare_parameter("target_odom_topic", std::string("/target/odom"));
        this->declare_parameter("target_accel_topic", std::string("/target/accel"));
        this->declare_parameter("enable_terminal_freeze", true);
        this->declare_parameter("terminal_freeze_enter_pos", 0.20);
        this->declare_parameter("terminal_freeze_enter_vel", 0.10);
        this->declare_parameter("terminal_freeze_require_vel", false);
        this->declare_parameter("terminal_freeze_exit_pos", 0.20);

        // Get parameters
        ocp_type_ = this->get_parameter("ocp_type").as_string();
        drone_name_ = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        platform_ = this->get_parameter("platform").as_string();
        solver_type_ = this->get_parameter("solver").as_string();
        mode_ = this->get_parameter("mode").as_string();
        n_replay_ = this->get_parameter("n_replay").as_int();
        mass_kg_ = this->get_parameter("mass_kg").as_double();
        target_odom_topic_ = this->get_parameter("target_odom_topic").as_string();
        target_accel_topic_ = this->get_parameter("target_accel_topic").as_string();
        enable_terminal_freeze_ = this->get_parameter("enable_terminal_freeze").as_bool();
        terminal_freeze_enter_pos_ = this->get_parameter("terminal_freeze_enter_pos").as_double();
        terminal_freeze_enter_vel_ = this->get_parameter("terminal_freeze_enter_vel").as_double();
        terminal_freeze_require_vel_ = this->get_parameter("terminal_freeze_require_vel").as_bool();
        terminal_freeze_exit_pos_ = this->get_parameter("terminal_freeze_exit_pos").as_double();

        double tx = this->get_parameter("hover_target_x").as_double();
        double ty = this->get_parameter("hover_target_y").as_double();
        double tz = this->get_parameter("hover_target_z").as_double();

        ocp_dt_ = OCPRegistry::getDT(ocp_type_);

        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(13);
        terminal(0) = tx; terminal(1) = ty; terminal(2) = tz;
        terminal(6) = 1.0;
        terminal_position_abs_ << tx, ty, tz;

  // Initialize MPC solver
    if (solver_type_ == "alipddp") {
        QuadrotorMPC::Config cfg;
        cfg.ocp_type = ocp_type_;
        cfg.terminal_state = terminal;
        cfg.n_shift = n_replay_;
        alipddp_mpc_ = std::make_unique<QuadrotorMPC>(cfg);
    }

        // Initialize replay tick counter to n_replay_ so first solve fires immediately
        replay_ticks_since_solve_.store(n_replay_);

        // Create callback groups (separate to prevent blocking)
        sensor_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        solver_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        replay_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

// Setup platform-specific subscriptions and publishers
// IMPORTANT: Direct subscribe/publish, no intermediate bridge node
if (platform_ == "crazyflie") {
    platform::crazyflie::setup(
        this, sensor_cb_group_, drone_name_,
        state_monitor_.crazyflieState(), state_monitor_.stateMutex(), cf_handles_);
} else if (platform_ == "px4") {
    platform::px4::setup(
        this, sensor_cb_group_,
        state_monitor_.px4State(), state_monitor_.stateMutex(), px4_handles_);
} else {
    RCLCPP_ERROR(this->get_logger(), "Unknown platform: %s", platform_.c_str());
    throw std::runtime_error("Unknown platform: " + platform_);
}

if (ocp_type_ == "stateswitch") {
    platform::target_tracker::setup(
        this, sensor_cb_group_, target_odom_topic_, target_accel_topic_,
        state_monitor_.targetState(), state_monitor_.targetMutex(), target_handles_);
}

        // Trajectory publisher (for visualization)
        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);

        // Setup timers based on mode
        if (mode_ == "mpc") {
            solver_timer_ = this->create_wall_timer(
                1ms,
                std::bind(&PlannerNode::solverLoop, this),
                solver_cb_group_);

            const int replay_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
            mpc_replay_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(replay_ms),
                std::bind(&PlannerNode::mpcReplayTick, this),
                replay_cb_group_);
        } else {
            startup_timer_ = this->create_wall_timer(
                50ms,
                std::bind(&PlannerNode::openLoopStartupCheck, this),
                solver_cb_group_);
        }

        RCLCPP_INFO(this->get_logger(),
            "Ready mode=%s platform=%s solver=%s ocp=%s "
            "ocp_dt=%.3fs n_replay=%d replay_period=%.0fms",
            mode_.c_str(), platform_.c_str(), solver_type_.c_str(),
            ocp_type_.c_str(), ocp_dt_, n_replay_, n_replay_ * ocp_dt_ * 1000.0);
        RCLCPP_INFO(this->get_logger(), "Target: [%.3f, %.3f, %.3f]", tx, ty, tz);
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    // Check if state is available (platform-agnostic)
    // ─────────────────────────────────────────────────────────────────────────
bool hasState() const {
    return state_monitor_.hasState(platform_);
}

bool hasFreshTargetState() const {
    return state_monitor_.hasFreshTargetState(ocp_type_, this->now());
}

TargetSnapshot getTargetSnapshot() const {
    return state_monitor_.getTargetSnapshot(ocp_type_, this->now());
}

Eigen::VectorXd getCurrentState() const {
    return state_monitor_.getCurrentState(platform_);
}

    // ─────────────────────────────────────────────────────────────────────────
    // SOLVER LOOP - polls at 1ms but only solves once n_replay_ ticks fired
    // ─────────────────────────────────────────────────────────────────────────
    void solverLoop() {
        if (!hasState()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for state...");
            return;
        }

        if (!hasFreshTargetState()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for fresh target state...");
            return;
        }

        if (replay_ticks_since_solve_.load() < n_replay_) return;

        Eigen::VectorXd x0_abs = getCurrentState();
        TargetSnapshot target_snapshot = getTargetSnapshot();
        Eigen::VectorXd x0 = x0_abs;
        if (ocp_type_ == "stateswitch") {
            x0 = Eigen::VectorXd::Zero(13);
            x0.segment(0, 3) = x0_abs.segment(0, 3) - target_snapshot.position;
            x0.segment(3, 3) = x0_abs.segment(3, 3) - target_snapshot.velocity;
            x0.segment(6, 7) = x0_abs.segment(6, 7);
        }

        const double pos_err = (ocp_type_ == "stateswitch")
            ? x0.segment(0, 3).norm()
            : (x0_abs.segment(0, 3) - terminal_position_abs_).norm();
        const double vel_err = (ocp_type_ == "stateswitch")
            ? x0.segment(3, 3).norm()
            : x0_abs.segment(3, 3).norm();

        if (terminal_freeze_.load()) {
            if (pos_err > terminal_freeze_exit_pos_) {
                terminal_freeze_.store(false);
                RCLCPP_INFO(this->get_logger(),
                    "Terminal freeze released: pos_err=%.3f", pos_err);
            } else {
                return;
            }
        }

        const bool freeze_condition = (pos_err < terminal_freeze_enter_pos_) &&
            (!terminal_freeze_require_vel_ || vel_err < terminal_freeze_enter_vel_);
        if (enable_terminal_freeze_ && is_primed_.load() && freeze_condition) {
            terminal_freeze_.store(true);
            RCLCPP_INFO(this->get_logger(),
                "Terminal freeze engaged: pos_err=%.3f vel_err=%.3f", pos_err, vel_err);
            return;
        }

if (!is_flying_) {
    is_flying_ = true;
    RCLCPP_INFO(this->get_logger(),
        "State received — starting RH MPC (%s)", solver_type_.c_str());
    if (platform_ == "px4") {
        platform::px4::arm(this, px4_handles_);
    }
    if (logging_enabled_ && !logging_initialized_) {
                setupLogging();
                logging_initialized_ = true;
            }
            RCLCPP_INFO(this->get_logger(),
                "x0=[%.3f,%.3f,%.3f | %.3f,%.3f,%.3f]",
                x0(0), x0(1), x0(2), x0(3), x0(4), x0(5));
        }

        const Eigen::Vector3d target_accel =
            (ocp_type_ == "stateswitch") ? target_snapshot.acceleration : Eigen::Vector3d::Zero();
        SolverResult result = callSolver(x0, target_accel);
        if (ocp_type_ == "stateswitch") {
            result.is_relative_plan = true;
            result.target_snapshot_pos = target_snapshot.position;
            result.target_snapshot_vel = target_snapshot.velocity;
            result.target_snapshot_acc = target_snapshot.acceleration;
        }

        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "Solve FAILED — holding");
            Eigen::VectorXd x_hold = x0_abs;
            x_hold.segment(3, 3).setZero();
            x_hold.segment(10, 3).setZero();
            publishCommand(x_hold, Eigen::VectorXd::Zero(4));
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "[RH %d] %.1fms iters=%d x0=[%.3f,%.3f,%.3f]",
            solve_count_, result.solve_time_ms, result.solve_iters,
            x0(0), x0(1), x0(2));

        {
            std::lock_guard<std::mutex> lk(mpc_traj_mutex_);
            mpc_traj_ = result.state_trajectory;
            mpc_ctrl_ = result.control_trajectory;
            const int k = static_cast<int>(std::round(result.solve_time_ms / (ocp_dt_ * 1000.0)));
            const int N = static_cast<int>(result.state_trajectory.size()) - 1;
            mpc_replay_idx_ = std::min(1 + k, N);
            mpc_active_solve_num_ = solve_count_;
            last_plan_is_relative_ = result.is_relative_plan;
            last_target_snapshot_pos_ = result.target_snapshot_pos;
            last_target_snapshot_vel_ = result.target_snapshot_vel;
            last_target_snapshot_acc_ = result.target_snapshot_acc;
            last_solve_wall_time_ = result.solve_timestamp;
        }
        replay_ticks_since_solve_.store(0);
        is_primed_.store(true);

        publishTrajectory(result.state_trajectory);

        if (logging_enabled_ && logging_initialized_) {
            logSolveTrajectory(result.state_trajectory, result.control_trajectory,
                               result.solve_time_ms, result.solve_iters);
        }
        ++solve_count_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // REPLAY TICK - time-indexed replay for variable-step stateswitch
    // ─────────────────────────────────────────────────────────────────────────
    void mpcReplayTick() {
        constexpr int IDX_DT = 13;

        if (!is_primed_.load()) return;

        const Clock::time_point now_wall = Clock::now();
        Eigen::VectorXd x_cmd = Eigen::VectorXd::Zero(13);
        Eigen::VectorXd u_cmd = Eigen::VectorXd::Zero(4);
        int active_solve_num = -1;

        bool plan_is_relative = false;
        double elapsed = 0.0;
        double horizon_end = 0.0;
        Eigen::VectorXd x_rel_k;

        {
            std::lock_guard<std::mutex> lk(mpc_traj_mutex_);
            if (mpc_traj_.empty()) {
                replay_ticks_since_solve_.fetch_add(1);
                return;
            }

            active_solve_num = mpc_active_solve_num_;
            plan_is_relative = last_plan_is_relative_;

            if (!plan_is_relative) {
                const int N = static_cast<int>(mpc_traj_.size()) - 1;
                elapsed = std::chrono::duration<double>(now_wall - last_solve_wall_time_).count();
                int idx = std::min(static_cast<int>(elapsed / ocp_dt_), N);
                idx = std::max(0, idx);
                mpc_replay_idx_ = idx;

                if (mpc_traj_[idx].size() >= 13) x_cmd = mpc_traj_[idx].head(13);
                if (!mpc_ctrl_.empty()) {
                    const int u_idx = std::min(idx, static_cast<int>(mpc_ctrl_.size()) - 1);
                    u_cmd = mpc_ctrl_[u_idx];
                }
                horizon_end = N * ocp_dt_;
            } else {
                elapsed = std::chrono::duration<double>(now_wall - last_solve_wall_time_).count();

                const int N = static_cast<int>(mpc_traj_.size()) - 1;
                int idx = 0;
                while (idx + 1 < static_cast<int>(mpc_traj_.size()) &&
                       mpc_traj_[idx].size() > IDX_DT &&
                       mpc_traj_[idx](IDX_DT) < elapsed) {
                    ++idx;
                }
                idx = std::min(idx, N);
                mpc_replay_idx_ = idx;

                x_rel_k = mpc_traj_[idx];
                if (!mpc_ctrl_.empty()) {
                    const int u_idx = std::min(idx, static_cast<int>(mpc_ctrl_.size()) - 1);
                    u_cmd = mpc_ctrl_[u_idx];
                }

                if (!mpc_traj_.empty() && mpc_traj_.back().size() > IDX_DT) {
                    horizon_end = mpc_traj_.back()(IDX_DT);
                }
            }
        }

        if (plan_is_relative) {
            if (elapsed > horizon_end + 0.2) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Stateswitch trajectory stale (elapsed=%.3fs, horizon=%.3fs), clamping to terminal",
                    elapsed, horizon_end);
            }

            const auto [tgt_pos_now, tgt_vel_now] = state_monitor_.getTargetPositionVelocity();

            if (x_rel_k.size() >= 13) {
                x_cmd.segment(0, 3) = x_rel_k.segment(0, 3) + tgt_pos_now;
                x_cmd.segment(3, 3) = x_rel_k.segment(3, 3) + tgt_vel_now;
                x_cmd.segment(6, 7) = x_rel_k.segment(6, 7);
            }
        } else if (elapsed > horizon_end + 0.2) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Trajectory stale (elapsed=%.3fs, horizon=%.3fs), clamping to terminal",
                elapsed, horizon_end);
        }

        replay_ticks_since_solve_.fetch_add(1);
        publishCommand(x_cmd, u_cmd);

        if (logging_enabled_ && logging_initialized_) {
            Eigen::VectorXd act = getCurrentState();
            logActualState(act, active_solve_num);
            logCommandedState(x_cmd, u_cmd, active_solve_num);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SOLVER CALL
    // ─────────────────────────────────────────────────────────────────────────
    SolverResult callSolver(const Eigen::VectorXd& state,
                            const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero()) {
        SolverResult result;
        if (solver_type_ == "alipddp" && alipddp_mpc_) {
            auto r = alipddp_mpc_->solve(state, target_accel);
            result.success = r.success;
            result.next_state = r.next_state;
            result.state_trajectory = r.state_trajectory;
            result.control_trajectory = r.control_trajectory;
            result.solve_time_ms = r.solve_time_ms;
            result.solve_iters = r.solve_iters;
            result.solve_timestamp = r.solve_timestamp;
            result.is_relative_plan = (ocp_type_ == "stateswitch");
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PUBLISH COMMAND - delegates to platform-specific implementation
    // ─────────────────────────────────────────────────────────────────────────
void publishCommand(const Eigen::VectorXd& s, const Eigen::VectorXd& u) {
    if (platform_ == "crazyflie") {
        platform::crazyflie::publishCommand(this, cf_handles_, s, u, mass_kg_);
    } else if (platform_ == "px4") {
        platform::px4::publishCommand(this, px4_handles_, s);
    }
}

    void publishTrajectory(const std::vector<Eigen::VectorXd>& traj) {
        nav_msgs::msg::Path path;
        path.header.stamp = this->now();
        path.header.frame_id = "world";
        for (const auto& s : traj) {
            geometry_msgs::msg::PoseStamped p;
            p.header.frame_id = "world";
            p.pose.position.x = s(0);
            p.pose.position.y = s(1);
            p.pose.position.z = s(2);
            p.pose.orientation.w = s(6);
            p.pose.orientation.x = s(7);
            p.pose.orientation.y = s(8);
            p.pose.orientation.z = s(9);
            path.poses.push_back(p);
        }
        traj_pub_->publish(path);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // OPEN-LOOP MODE
    // ─────────────────────────────────────────────────────────────────────────
    void openLoopStartupCheck() {
        if (!hasState()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[OpenLoop] Waiting for state...");
            return;
        }
        startup_timer_->cancel();

        Eigen::VectorXd x0 = getCurrentState();
        RCLCPP_INFO(this->get_logger(), "[OpenLoop] Solving...");
        SolverResult result = callSolver(x0);

        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_ERROR(this->get_logger(), "[OpenLoop] FAILED");
            return;
        }

        ol_ref_X_ = result.state_trajectory;
        ol_ref_U_ = result.control_trajectory;

        if (logging_enabled_) {
            setupLogging();
            logging_initialized_ = true;
            logSolveTrajectory(ol_ref_X_, ol_ref_U_, result.solve_time_ms, result.solve_iters);
        }
        publishTrajectory(ol_ref_X_);

        ol_replay_step_ = 0;
        ol_replay_start_time_ = Clock::now();
        const int period_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
        replay_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&PlannerNode::openLoopReplayTick, this),
            solver_cb_group_);

        RCLCPP_INFO(this->get_logger(), "[OpenLoop] Replaying %zu steps", ol_ref_X_.size());
    }

    void openLoopReplayTick() {
        const int N = static_cast<int>(ol_ref_X_.size()) - 1;
        const int step = std::min(ol_replay_step_, N);
        const Eigen::VectorXd& x_cmd = ol_ref_X_[step];
        const Eigen::VectorXd u_cmd = (step < static_cast<int>(ol_ref_U_.size()))
            ? ol_ref_U_[step] : Eigen::VectorXd::Zero(4);

        publishCommand(x_cmd, u_cmd);

        if (logging_enabled_ && logging_initialized_) {
            Eigen::VectorXd act = getCurrentState();
            logActualState(act, 0);
            logCommandedState(x_cmd, u_cmd, 0);
        }

        if (ol_replay_step_ < N) ++ol_replay_step_;
        else if (!ol_done_logged_) {
            RCLCPP_INFO(this->get_logger(), "[OpenLoop] Done. Holding.");
            ol_done_logged_ = true;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LOGGING
    // ─────────────────────────────────────────────────────────────────────────
    void setupLogging() {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ts;
        ts << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        std::string folder = "./logs/" + drone_name_ + "_" + ocp_type_ + "_"
            + mode_ + "_" + solver_type_ + "_" + ts.str();
        std::filesystem::create_directories(folder);
        log_folder_ = folder;

        all_solves_log_.open(folder + "/all_solves.csv");
        if (all_solves_log_.is_open())
            all_solves_log_ << "solve_num,solve_time_ms,solve_iters,node,t,"
                "x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
                "fz,mx,my,mz\n";

        commanded_state_log_.open(folder + "/commanded_state.csv");
        if (commanded_state_log_.is_open())
            commanded_state_log_ << "timestamp,solve_num,"
                "x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
                "fz,mx,my,mz,"
                "acc_x,acc_y,acc_z\n";

        actual_state_log_.open(folder + "/actual_state.csv");
        if (actual_state_log_.is_open())
            actual_state_log_ << "timestamp,solve_num,"
                "x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";

        RCLCPP_INFO(this->get_logger(), "Logging to: %s", folder.c_str());
    }

    double wallTimeSec() {
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    void logSolveTrajectory(const std::vector<Eigen::VectorXd>& traj,
                            const std::vector<Eigen::VectorXd>& ctrls,
                            double solve_time_ms, int solve_iters) {
        if (!all_solves_log_.is_open()) return;
        for (int i = 0; i < static_cast<int>(traj.size()); ++i) {
            const auto& s = traj[i];
            if (s.size() < 13) continue;
            all_solves_log_ << std::fixed << std::setprecision(6)
                << solve_count_ << "," << solve_time_ms << ","
                << solve_iters << "," << i << "," << (i * ocp_dt_);
            for (int j = 0; j < 13; ++j) all_solves_log_ << "," << s(j);
            if (i < static_cast<int>(ctrls.size()) && ctrls[i].size() >= 4)
                all_solves_log_ << "," << ctrls[i](0) << "," << ctrls[i](1)
                    << "," << ctrls[i](2) << "," << ctrls[i](3);
            else
                all_solves_log_ << ",0,0,0,0";
            all_solves_log_ << "\n";
        }
        all_solves_log_.flush();
    }

    void logActualState(const Eigen::VectorXd& state, int solve_num) {
        if (!actual_state_log_.is_open()) return;
        actual_state_log_ << std::fixed << std::setprecision(6)
            << wallTimeSec() << "," << solve_num;
        for (int i = 0; i < 13; ++i) actual_state_log_ << "," << state(i);
        actual_state_log_ << "\n";
        actual_state_log_.flush();
    }

    void logCommandedState(const Eigen::VectorXd& s, const Eigen::VectorXd& u, int solve_num) {
        if (!commanded_state_log_.is_open() || s.size() < 13) return;
        const double fz = (u.size() >= 1) ? u(0) : 0.0;
        Eigen::Vector3d acc = platform::crazyflie::computeAcc(s, fz, mass_kg_);

        commanded_state_log_ << std::fixed << std::setprecision(6)
            << wallTimeSec() << "," << solve_num;
        for (int i = 0; i < 13; ++i) commanded_state_log_ << "," << s(i);
        if (u.size() >= 4)
            commanded_state_log_ << "," << u(0) << "," << u(1)
                << "," << u(2) << "," << u(3);
        else
            commanded_state_log_ << ",0,0,0,0";
        commanded_state_log_ << "," << acc.x() << "," << acc.y() << "," << acc.z();
        commanded_state_log_ << "\n";
        commanded_state_log_.flush();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MEMBERS - Configuration
    // ─────────────────────────────────────────────────────────────────────────
    std::string ocp_type_, platform_, solver_type_, mode_, drone_name_;
    std::string target_odom_topic_ = "/target/odom";
    std::string target_accel_topic_ = "/target/accel";
    bool enable_terminal_freeze_ = true;
    double terminal_freeze_enter_pos_ = 0.20;
    double terminal_freeze_enter_vel_ = 0.10;
    bool terminal_freeze_require_vel_ = false;
    double terminal_freeze_exit_pos_ = 0.20;
    bool logging_enabled_;
    double ocp_dt_ = 0.05;
    double mass_kg_ = 0.027;
    int n_replay_ = 4;

    // MPC solver
    std::unique_ptr<QuadrotorMPC> alipddp_mpc_;

    // Platform handles
    platform::crazyflie::Handles cf_handles_;
    platform::px4::Handles px4_handles_;

    // Timers
    rclcpp::TimerBase::SharedPtr solver_timer_;
    rclcpp::TimerBase::SharedPtr mpc_replay_timer_;
    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr replay_timer_;

    // Trajectory publisher
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;

    // MPC trajectory buffer
    std::mutex mpc_traj_mutex_;
    std::vector<Eigen::VectorXd> mpc_traj_;
    std::vector<Eigen::VectorXd> mpc_ctrl_;
    int mpc_replay_idx_ = 1;
    int mpc_active_solve_num_ = -1;

    // Tick counter
    std::atomic<int> replay_ticks_since_solve_{0};

    // State and target ownership (thread-safe copy-out API)
    StateMonitor state_monitor_;

    // Target-tracker handles
    platform::target_tracker::Handles target_handles_;

    // Callback groups
    rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
    rclcpp::CallbackGroup::SharedPtr solver_cb_group_;
    rclcpp::CallbackGroup::SharedPtr replay_cb_group_;

    // State flags
    bool is_flying_ = false;
    int solve_count_ = 0;
    std::atomic<bool> is_primed_{false};
    std::atomic<bool> terminal_freeze_{false};
    Eigen::Vector3d terminal_position_abs_ = Eigen::Vector3d::Zero();

    // Open-loop state
    std::vector<Eigen::VectorXd> ol_ref_X_, ol_ref_U_;

    // Last solve snapshot (for stateswitch replay/log reconstruction)
    bool last_plan_is_relative_ = false;
    Eigen::Vector3d last_target_snapshot_pos_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_target_snapshot_vel_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_target_snapshot_acc_ = Eigen::Vector3d::Zero();
    Clock::time_point last_solve_wall_time_{};
    int ol_replay_step_ = 0;
    bool ol_done_logged_ = false;
    Clock::time_point ol_replay_start_time_;

    // Logging
    bool logging_initialized_ = false;
    std::string log_folder_;
    std::ofstream commanded_state_log_;
    std::ofstream actual_state_log_;
    std::ofstream all_solves_log_;
};

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
