/// @file comando_planner_node.cpp
/// @brief CoManDO Planner — platform-agnostic MPC solver node.

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "comando_planner/msg/mpc_command.hpp"

#include "solver/alipddp_solver.hpp"
#include "solver/ocp_registry.hpp"
#include "utils/logger.hpp"

#ifdef HAS_ACADOS
#include "solver/acados_solver.hpp"
#endif

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

struct SolverResult {
    bool                         success   = false;
    Eigen::VectorXd              next_state;
    std::vector<Eigen::VectorXd> state_trajectory;
    std::vector<Eigen::VectorXd> control_trajectory;
    double                       solve_time_ms = 0.0;
    std::chrono::steady_clock::time_point solve_timestamp;
};

class CommandoPlannerNode : public rclcpp::Node {
public:
    CommandoPlannerNode() : Node("comando_planner") {

        // ── Parameters ────────────────────────────────────────────────────────
        this->declare_parameter("ocp_type",       std::string("hover"));
        this->declare_parameter("drone_name",     std::string("cf_1"));
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("solver",         std::string("alipddp"));
        this->declare_parameter("mode",           std::string("mpc"));
        this->declare_parameter("hover_target_x", 0.0);
        this->declare_parameter("hover_target_y", 0.0);
        this->declare_parameter("hover_target_z", 1.0);
        this->declare_parameter("sim_n_shift",    0);    // 0=hardware timing, 1=match standalone N_SHIFT=1
        this->declare_parameter("max_solves",     0);    // 0=run forever, >0=stop after N solves (match standalone loop)

        ocp_type_        = this->get_parameter("ocp_type").as_string();
        drone_name_      = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        solver_type_     = this->get_parameter("solver").as_string();
        mode_            = this->get_parameter("mode").as_string();

        // ── Validate ──────────────────────────────────────────────────────────
        if (solver_type_ != "alipddp" && solver_type_ != "acados")
            throw std::runtime_error("Unknown solver: " + solver_type_);
        if (mode_ != "mpc" && mode_ != "open_loop")
            throw std::runtime_error("Unknown mode: " + mode_);
#ifndef HAS_ACADOS
        if (solver_type_ == "acados")
            throw std::runtime_error("Acados not compiled in");
#endif

        // ── Terminal state ────────────────────────────────────────────────────
        // Landing OCP: terminal_state is ignored — make_x_ref() defines it internally.
        // Hover OCP:   terminal_state is the hover setpoint.
        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(13);
        terminal(6) = 1.0;

        if (ocp_type_ == "hover") {
            terminal(0) = this->get_parameter("hover_target_x").as_double();
            terminal(1) = this->get_parameter("hover_target_y").as_double();
            terminal(2) = this->get_parameter("hover_target_z").as_double();
            RCLCPP_INFO(this->get_logger(),
                "[CoManDO Planner] Target: [%.3f, %.3f, %.3f]",
                terminal(0), terminal(1), terminal(2));
        } else {
            RCLCPP_INFO(this->get_logger(),
                "[CoManDO Planner] Target: defined in OCP (z_ref=0.1m)");
        }

        // ── OCP timing ────────────────────────────────────────────────────────
        ocp_dt_ = OCPRegistry::getDT(ocp_type_);

        const int default_solver_rate = static_cast<int>(std::round(1.0 / ocp_dt_));
        this->declare_parameter("solver_rate", default_solver_rate);
        solver_rate_ = this->get_parameter("solver_rate").as_int();

        const double solver_period = 1.0 / static_cast<double>(solver_rate_);
        n_shift_ = std::max(1, static_cast<int>(std::round(solver_period / ocp_dt_)));

        // ── Create solver ─────────────────────────────────────────────────────
        if (solver_type_ == "alipddp") {
            ALIPDDPSolver::Config cfg;
            cfg.ocp_type       = ocp_type_;
            cfg.terminal_state = terminal;
            cfg.n_shift        = n_shift_;
            cfg.sim_n_shift    = this->get_parameter("sim_n_shift").as_int();
            alipddp_mpc_ = std::make_unique<ALIPDDPSolver>(cfg);
        }
#ifdef HAS_ACADOS
        else if (solver_type_ == "acados") {
            AcadosMPC::Config cfg;
            cfg.ocp_type       = ocp_type_;
            cfg.terminal_state = terminal;
            cfg.dt             = ocp_dt_;
            cfg.n_shift        = n_shift_;
            acados_mpc_ = std::make_unique<AcadosMPC>(cfg);
        }
#endif

        max_solves_ = this->get_parameter("max_solves").as_int();

        // ── Initial state ─────────────────────────────────────────────────────
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;

        // ── Callback groups ───────────────────────────────────────────────────
        sensor_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        solver_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        // ── Subscriptions ─────────────────────────────────────────────────────
        rclcpp::SubscriptionOptions sensor_opts;
        sensor_opts.callback_group = sensor_cb_group_;

        state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/mpc/state", 10,
            std::bind(&CommandoPlannerNode::stateCallback, this, std::placeholders::_1),
            sensor_opts);

        // ── Publishers ────────────────────────────────────────────────────────
        cmd_pub_ = this->create_publisher<comando_planner::msg::MpcCommand>(
            "/mpc/command", 10);

        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);

        // ── Timers ────────────────────────────────────────────────────────────
        if (mode_ == "mpc") {
            solver_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(1000 / solver_rate_),
                std::bind(&CommandoPlannerNode::solverLoop, this),
                solver_cb_group_);
        } else {
            startup_timer_ = this->create_wall_timer(
                50ms, std::bind(&CommandoPlannerNode::openLoopStartupCheck, this),
                solver_cb_group_);
        }

        RCLCPP_INFO(this->get_logger(),
            "[CoManDO Planner] mode=%s  solver=%s  ocp=%s  "
            "solver_rate=%dHz  ocp_dt=%.3fs  n_shift=%d",
            mode_.c_str(), solver_type_.c_str(), ocp_type_.c_str(),
            solver_rate_, ocp_dt_, n_shift_);
        RCLCPP_INFO(this->get_logger(),
            "[CoManDO Planner] Waiting for /mpc/state...");
    }

private:
    void stateCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        current_state_(0)  = msg->pose.pose.position.x;
        current_state_(1)  = msg->pose.pose.position.y;
        current_state_(2)  = msg->pose.pose.position.z;
        current_state_(3)  = msg->twist.twist.linear.x;
        current_state_(4)  = msg->twist.twist.linear.y;
        current_state_(5)  = msg->twist.twist.linear.z;
        current_state_(6)  = msg->pose.pose.orientation.w;
        current_state_(7)  = msg->pose.pose.orientation.x;
        current_state_(8)  = msg->pose.pose.orientation.y;
        current_state_(9)  = msg->pose.pose.orientation.z;
        current_state_(10) = msg->twist.twist.angular.x;
        current_state_(11) = msg->twist.twist.angular.y;
        current_state_(12) = msg->twist.twist.angular.z;
        state_received_ = true;
    }

    void solverLoop()
    {
        bool have_state;
        Eigen::VectorXd x0;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            have_state = state_received_;
            x0 = current_state_;
        }
        if (!have_state) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[CoManDO Planner] Waiting for /mpc/state...");
            return;
        }

        if (!is_running_) {
            is_running_ = true;
            RCLCPP_INFO(this->get_logger(),
                "[CoManDO Planner] State received — starting MPC (%s) at %.0f Hz replay",
                solver_type_.c_str(), 1.0 / ocp_dt_);

            if (logging_enabled_ && !logging_initialized_) {
                logger_.setup(drone_name_, ocp_type_, mode_, solver_type_);
                logging_initialized_ = true;
            }
        }

        const auto t_solve_start = this->now();
        SolverResult result = callSolver(x0);
        const double solve_sec = (this->now() - t_solve_start).seconds();

        if (result.success) {
            const auto& X = result.state_trajectory;
            const auto& U = result.control_trajectory;
            const int   N = static_cast<int>(X.size()) - 1;

            const int skip = std::max(1,
                std::min(static_cast<int>(std::round(solve_sec / ocp_dt_)),
                         std::max(1, n_shift_ - 1)));

            RCLCPP_INFO(this->get_logger(),
                "[CoManDO Planner] solve=%.1fms  latency_skip=%d  replay starts at X[%d]",
                solve_sec * 1000.0, skip, skip);

            publishMpcCommand(X, U, N, skip, result.solve_time_ms);
            publishTrajectory(X);

            if (logging_enabled_ && logging_initialized_) {
                logger_.logSolveTrajectory(X, U, result.solve_time_ms, ocp_dt_);
                // X[1] is what sim_bridge publishes as next x0.
                // U[0] is the first control applied.
                // Together = "commanded state" — matches standalone X_exec/U_exec.
                if (X.size() > 1 && !U.empty())
                    logger_.logCommandedState(X[1], U[0]);
            }

            if (++diag_count_ % 10 == 0) printDiagnostics(result);

            // ── Stop after max_solves (matches standalone fixed-step loop) ────
            ++solve_count_;
            if (max_solves_ > 0 && solve_count_ >= max_solves_) {
                RCLCPP_INFO(this->get_logger(),
                    "[CoManDO Planner] Reached max_solves=%d — stopping.", max_solves_);
                solver_timer_->cancel();
            }

        } else {
            RCLCPP_WARN(this->get_logger(),
                "[CoManDO Planner] MPC solve FAILED — bridge replays stale traj");
        }
    }

    void openLoopStartupCheck()
    {
        bool have_state;
        { std::lock_guard<std::mutex> lk(state_mutex_); have_state = state_received_; }
        if (!have_state) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[OpenLoop] Waiting for /mpc/state...");
            return;
        }
        startup_timer_->cancel();

        Eigen::VectorXd x0;
        { std::lock_guard<std::mutex> lk(state_mutex_); x0 = current_state_; }

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoop] Running ONE solve (%s)...", solver_type_.c_str());

        SolverResult result = callSolver(x0);

        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_ERROR(this->get_logger(), "[OpenLoop] Solve FAILED.");
            return;
        }

        const auto& X = result.state_trajectory;
        const auto& U = result.control_trajectory;
        const int   N = static_cast<int>(X.size()) - 1;

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoop] Solve OK in %.1f ms — %zu states, horizon=%.2fs",
            result.solve_time_ms, X.size(), (X.size()-1) * ocp_dt_);

        if (logging_enabled_) {
            logger_.setup(drone_name_, ocp_type_, mode_, solver_type_);
            logging_initialized_ = true;
            logger_.logSolveTrajectory(X, U, result.solve_time_ms, ocp_dt_);
        }

        publishMpcCommand(X, U, N, /*skip=*/1, result.solve_time_ms);
        publishTrajectory(X);
    }

    void publishMpcCommand(const std::vector<Eigen::VectorXd>& X,
                           const std::vector<Eigen::VectorXd>& U,
                           int N, int replay_start_idx, double solve_time_ms)
    {
        auto msg = std::make_unique<comando_planner::msg::MpcCommand>();
        msg->header.stamp     = this->now();
        msg->header.frame_id  = "world";
        msg->success          = true;
        msg->solve_time_ms    = solve_time_ms;
        msg->ocp_dt           = ocp_dt_;
        msg->horizon          = N;
        msg->nx               = 13;
        msg->nu               = 4;
        msg->n_shift          = n_shift_;
        msg->replay_start_idx = replay_start_idx;

        msg->state_trajectory.resize((N + 1) * 13);
        for (int k = 0; k <= N; ++k)
            for (int i = 0; i < 13; ++i)
                msg->state_trajectory[k * 13 + i] = X[k](i);

        msg->control_trajectory.resize(N * 4);
        for (int k = 0; k < N; ++k)
            for (int i = 0; i < 4; ++i)
                msg->control_trajectory[k * 4 + i] = U[k](i);

        if (!U.empty() && U[0].size() >= 4) {
            msg->fz = U[0](0);
            msg->mx = U[0](1);
            msg->my = U[0](2);
            msg->mz = U[0](3);
        }

        cmd_pub_->publish(std::move(msg));
    }

    void publishTrajectory(const std::vector<Eigen::VectorXd>& traj)
    {
        nav_msgs::msg::Path path;
        path.header.stamp    = this->now();
        path.header.frame_id = "world";
        for (const auto& s : traj) {
            geometry_msgs::msg::PoseStamped p;
            p.header.frame_id    = "world";
            p.pose.position.x    = s(0); p.pose.position.y    = s(1); p.pose.position.z    = s(2);
            p.pose.orientation.w = s(6); p.pose.orientation.x = s(7);
            p.pose.orientation.y = s(8); p.pose.orientation.z = s(9);
            path.poses.push_back(p);
        }
        traj_pub_->publish(path);
    }

    SolverResult callSolver(const Eigen::VectorXd& state)
    {
        SolverResult result;
        if (solver_type_ == "alipddp" && alipddp_mpc_) {
            auto r = alipddp_mpc_->solve(state);
            result.success            = r.success;
            result.next_state         = r.next_state;
            result.state_trajectory   = r.state_trajectory;
            result.control_trajectory = r.control_trajectory;
            result.solve_time_ms      = r.solve_time_ms;
            result.solve_timestamp    = r.solve_timestamp;
        }
#ifdef HAS_ACADOS
        else if (solver_type_ == "acados" && acados_mpc_) {
            auto r = acados_mpc_->solve(state);
            result.success            = r.success;
            result.next_state         = r.next_state;
            result.state_trajectory   = r.state_trajectory;
            result.control_trajectory = r.control_trajectory;
            result.solve_time_ms      = r.solve_time_ms;
            result.solve_timestamp    = r.solve_timestamp;
        }
#endif
        return result;
    }

    void printDiagnostics(const SolverResult& result)
    {
        Eigen::VectorXd snap;
        { std::lock_guard<std::mutex> lk(state_mutex_); snap = current_state_; }
        double thrust = result.control_trajectory.empty()
                        ? 0.0 : result.control_trajectory[0](0);
        RCLCPP_INFO(this->get_logger(),
            "[%s] %.1fms | pos[%.3f,%.3f,%.3f] vel[%.3f,%.3f,%.3f] "
            "w[%.3f,%.3f,%.3f]rad/s | T=%.3fN",
            solver_type_.c_str(), result.solve_time_ms,
            snap(0),snap(1),snap(2), snap(3),snap(4),snap(5),
            snap(10),snap(11),snap(12), thrust);
    }

    // ── Members ───────────────────────────────────────────────────────────────
    std::string ocp_type_, solver_type_, mode_, drone_name_;
    bool        logging_enabled_;
    int         solver_rate_;
    double      ocp_dt_  = 0.1;
    int         n_shift_ = 1;

    std::unique_ptr<ALIPDDPSolver> alipddp_mpc_;
#ifdef HAS_ACADOS
    std::unique_ptr<AcadosMPC> acados_mpc_;
#endif

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       state_sub_;
    rclcpp::Publisher<comando_planner::msg::MpcCommand>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr              traj_pub_;
    rclcpp::TimerBase::SharedPtr solver_timer_;
    rclcpp::TimerBase::SharedPtr startup_timer_;

    rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
    rclcpp::CallbackGroup::SharedPtr solver_cb_group_;

    mutable std::mutex state_mutex_;
    Eigen::VectorXd    current_state_;
    bool               state_received_     = false;
    bool               is_running_         = false;

    CommandoLogger logger_;
    bool           logging_initialized_    = false;

    int diag_count_  = 0;
    int max_solves_  = 0;   // 0=run forever
    int solve_count_ = 0;   // incremented each successful solve
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CommandoPlannerNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}