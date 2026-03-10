// planner_node.cpp
// ─────────────────────────────────────────────────────────────────────────────
// CoManDO Planner — supports two platforms and two solvers:
//
//   platform:  "crazyflie"  (Crazyswarm2 topics, ENU)
//              "px4"        (PX4 uORB-over-DDS topics, NED↔ENU conversion)
//
//   solver:    "alipddp"    (ALIPDDP — augmented Lagrangian IPM DDP)
//              "acados"     (acados SQP_RTI — code-generated NLP)
//
// State layout (13-dim, ENU/FLU internally):
//   [0-2]   position        x, y, z          (m)
//   [3-5]   velocity        vx, vy, vz       (m/s)
//   [6-9]   quaternion      qw, qx, qy, qz   (–)
//   [10-12] angular rate    wx, wy, wz        (rad/s)
//
// ── MPC PUBLISH ARCHITECTURE (Zhang et al. 2023 §III, Algorithm 1 Line 9) ──
//
//   Two decoupled timers in MPC mode:
//
//   1. solver_timer_ (solver_rate Hz, e.g. 1-2 Hz)
//      Runs ALIPDDP, on success: locks mpc_traj_mutex_, stores X into
//      mpc_traj_, resets mpc_replay_idx_ = 1.
//      Does NOT publish directly.
//
//   2. mpc_replay_timer_ (ocp_dt Hz, e.g. 10 Hz for ocp_dt=0.1s)
//      Every ocp_dt seconds: reads mpc_traj_[mpc_replay_idx_] under lock,
//      publishes it, increments mpc_replay_idx_.
//      This implements the paper's "û(t) = u°_{i,j} for t in [t_{i,j}, t_{i,j+1})"
//      — the drone receives the next node on the committed trajectory while
//      the solver computes the next solution.
//
//   Effect on warm-start quality (paper Lemma 1):
//      Because the drone is actually moving between solves, by the time
//      solve k+1 runs, actual state ≈ prev_X_[n_shift].  This shrinks
//      vel_err, which raises γ in the γ-blend (quadrotor_mpc.cpp), which
//      lets the warm-start carry more information across solves.
// ─────────────────────────────────────────────────────────────────────────────

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

// ── Crazyflie interface ──────────────────────────────────────────────────────
#include <crazyflie_interfaces/msg/full_state.hpp>

// ── PX4 interface (conditionally compiled) ───────────────────────────────────
#ifdef HAS_PX4_MSGS
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#endif

// ── Solvers ──────────────────────────────────────────────────────────────────
#include "quadrotor_mpc.hpp"
#include "ocp_registry.hpp"
#ifdef HAS_ACADOS
#include "acados_solver.hpp"
#endif

#include <chrono>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <atomic>
#include <Eigen/Dense>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

// ═══════════════════════════════════════════════════════════════════════════════
//  NED ↔ ENU frame conversion helpers
// ═══════════════════════════════════════════════════════════════════════════════
namespace frame_conv {

inline Eigen::Vector3d ned_to_enu(const Eigen::Vector3d& v_ned) {
    return {v_ned.y(), v_ned.x(), -v_ned.z()};
}
inline Eigen::Vector3d enu_to_ned(const Eigen::Vector3d& v_enu) {
    return {v_enu.y(), v_enu.x(), -v_enu.z()};
}
inline Eigen::Quaterniond quat_ned_to_enu(double w, double x, double y, double z) {
    return Eigen::Quaterniond(w, y, x, -z).normalized();
}
inline Eigen::Quaterniond quat_enu_to_ned(const Eigen::Quaterniond& q_enu) {
    return Eigen::Quaterniond(q_enu.w(), q_enu.y(), q_enu.x(), -q_enu.z()).normalized();
}
inline Eigen::Vector3d omega_frd_to_flu(const Eigen::Vector3d& w_frd) {
    return {w_frd.x(), -w_frd.y(), -w_frd.z()};
}
inline Eigen::Vector3d omega_flu_to_frd(const Eigen::Vector3d& w_flu) {
    return {w_flu.x(), -w_flu.y(), -w_flu.z()};
}

} // namespace frame_conv


// ═══════════════════════════════════════════════════════════════════════════════
//  Unified Result type
// ═══════════════════════════════════════════════════════════════════════════════
struct SolverResult {
    bool                          success   = false;
    Eigen::VectorXd               next_state;
    std::vector<Eigen::VectorXd>  state_trajectory;
    std::vector<Eigen::VectorXd>  control_trajectory;
    double                        solve_time_ms = 0.0;
    std::chrono::steady_clock::time_point solve_timestamp;
};


// ═══════════════════════════════════════════════════════════════════════════════
//  Planner Node
// ═══════════════════════════════════════════════════════════════════════════════
class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        // ── Parameters ────────────────────────────────────────────────────────
        this->declare_parameter("ocp_type",       std::string("hover"));
        this->declare_parameter("drone_name",     std::string("cf_1"));
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("platform",       std::string("crazyflie"));
        this->declare_parameter("solver",         std::string("alipddp"));
        this->declare_parameter("mode",           std::string("mpc"));
        this->declare_parameter("hover_target_x", 0.0);
        this->declare_parameter("hover_target_y", 0.0);
        this->declare_parameter("hover_target_z", 1.0);

        ocp_type_        = this->get_parameter("ocp_type").as_string();
        drone_name_      = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        platform_        = this->get_parameter("platform").as_string();
        solver_type_     = this->get_parameter("solver").as_string();
        mode_            = this->get_parameter("mode").as_string();

        double tx = this->get_parameter("hover_target_x").as_double();
        double ty = this->get_parameter("hover_target_y").as_double();
        double tz = this->get_parameter("hover_target_z").as_double();

        // ── Validate ──────────────────────────────────────────────────────────
        if (platform_ != "crazyflie" && platform_ != "px4") {
            RCLCPP_ERROR(this->get_logger(), "Unknown platform '%s'", platform_.c_str());
            throw std::runtime_error("Invalid platform: " + platform_);
        }
        if (solver_type_ != "alipddp" && solver_type_ != "acados") {
            RCLCPP_ERROR(this->get_logger(), "Unknown solver '%s'", solver_type_.c_str());
            throw std::runtime_error("Invalid solver: " + solver_type_);
        }
        if (mode_ != "mpc" && mode_ != "open_loop") {
            RCLCPP_ERROR(this->get_logger(), "Unknown mode '%s'", mode_.c_str());
            throw std::runtime_error("Invalid mode: " + mode_);
        }
#ifndef HAS_ACADOS
        if (solver_type_ == "acados") throw std::runtime_error("Acados not available");
#endif
#ifndef HAS_PX4_MSGS
        if (platform_ == "px4") throw std::runtime_error("PX4 msgs not available");
#endif

        // ── Terminal state ────────────────────────────────────────────────────
        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(13);
        terminal(0) = tx; terminal(1) = ty; terminal(2) = tz;
        terminal(6) = 1.0;

        // ── OCP DT and solver rate ─────────────────────────────────────────────
        ocp_dt_ = OCPRegistry::getDT(ocp_type_);

        // solver_rate defaults to 1/ocp_dt (1 shift per solve).
        // Set lower (e.g. 2 Hz) to let the solver take more time per step.
        // n_shift = round(solver_period / ocp_dt) — how many OCP nodes
        // are consumed between solves, determines warm-start shift amount.
        const int default_solver_rate = static_cast<int>(std::round(1.0 / ocp_dt_));
        this->declare_parameter("solver_rate", default_solver_rate);
        solver_rate_ = this->get_parameter("solver_rate").as_int();

        const double solver_period = 1.0 / static_cast<double>(solver_rate_);
        n_shift_ = std::max(1, static_cast<int>(std::round(solver_period / ocp_dt_)));

        // ── Create solver ─────────────────────────────────────────────────────
        if (solver_type_ == "alipddp") {
            QuadrotorMPC::Config cfg;
            cfg.ocp_type       = ocp_type_;
            cfg.terminal_state = terminal;
            cfg.n_shift        = n_shift_;
            alipddp_mpc_ = std::make_unique<QuadrotorMPC>(cfg);
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

        // ── Initial state ─────────────────────────────────────────────────────
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;

        // ── Callback groups ───────────────────────────────────────────────────
        // sensor_cb_group_: pose/odom callbacks (fast, non-blocking)
        // solver_cb_group_: MPC solve timer (slow, blocking)
        // replay_cb_group_: MPC replay timer (fast, non-blocking) — SEPARATE
        //   from solver so replay continues while solver is running on another
        //   thread. MutuallyExclusive so only one replay fires at a time.
        sensor_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        solver_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        replay_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        // ── Platform-specific pub/sub ─────────────────────────────────────────
        if (platform_ == "crazyflie") {
            setupCrazyflie();
        } else {
            setupPX4();
        }

        // ── Trajectory visualisation ──────────────────────────────────────────
        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);

        // ── Timer setup ───────────────────────────────────────────────────────
        if (mode_ == "mpc") {
            // Solver timer: runs at solver_rate Hz (slow — blocks during solve)
            solver_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(1000 / solver_rate_),
                std::bind(&PlannerNode::solverLoop, this),
                solver_cb_group_);

            // Replay timer: runs at 1/ocp_dt Hz (fast — just reads and publishes)
            // This decouples the publish rate from the solve rate.
            // While the solver is blocked computing the next trajectory,
            // the replay timer continues advancing through the committed trajectory
            // so the drone receives a new setpoint every ocp_dt seconds.
            const int replay_period_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
            mpc_replay_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(replay_period_ms),
                std::bind(&PlannerNode::mpcReplayTick, this),
                replay_cb_group_);

        } else {
            startup_timer_ = this->create_wall_timer(
                50ms, std::bind(&PlannerNode::openLoopStartupCheck, this),
                solver_cb_group_);
        }

        RCLCPP_INFO(this->get_logger(),
            "Planner ready  mode=%s  platform=%s  solver=%s  ocp=%s  "
            "solver_rate=%dHz  ocp_dt=%.3fs  n_shift=%d",
            mode_.c_str(), platform_.c_str(), solver_type_.c_str(), ocp_type_.c_str(),
            solver_rate_, ocp_dt_, n_shift_);
        RCLCPP_INFO(this->get_logger(), "Target: [%.3f, %.3f, %.3f]", tx, ty, tz);
        RCLCPP_INFO(this->get_logger(), "Waiting for state feedback...");
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    //  Platform setup
    // ─────────────────────────────────────────────────────────────────────────

    void setupCrazyflie()
    {
        cf_cmd_pub_ = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);

        rclcpp::SubscriptionOptions sensor_opts;
        sensor_opts.callback_group = sensor_cb_group_;

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            std::bind(&PlannerNode::cfPoseCallback, this, std::placeholders::_1),
            sensor_opts);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + drone_name_ + "/odom", 10,
            std::bind(&PlannerNode::cfOdomCallback, this, std::placeholders::_1),
            sensor_opts);
    }

    void setupPX4()
    {
#ifdef HAS_PX4_MSGS
        px4_setpoint_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
            "/fmu/in/trajectory_setpoint", 10);
        px4_offboard_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
            "/fmu/in/offboard_control_mode", 10);
        px4_cmd_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
            "/fmu/in/vehicle_command", 10);

        rclcpp::SubscriptionOptions sensor_opts;
        sensor_opts.callback_group = sensor_cb_group_;
        px4_odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", 10,
            std::bind(&PlannerNode::px4OdomCallback, this, std::placeholders::_1),
            sensor_opts);

        px4_heartbeat_timer_ = this->create_wall_timer(
            100ms, std::bind(&PlannerNode::px4HeartbeatCallback, this));
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Crazyflie state callbacks
    // ─────────────────────────────────────────────────────────────────────────

    void cfPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        current_state_(0) = msg->pose.position.x;
        current_state_(1) = msg->pose.position.y;
        current_state_(2) = msg->pose.position.z;
        current_state_(6) = msg->pose.orientation.w;
        current_state_(7) = msg->pose.orientation.x;
        current_state_(8) = msg->pose.orientation.y;
        current_state_(9) = msg->pose.orientation.z;
        pose_received_ = true;
    }

    void cfOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        current_state_(3) = msg->twist.twist.linear.x;
        current_state_(4) = msg->twist.twist.linear.y;
        current_state_(5) = msg->twist.twist.linear.z;
        constexpr double DEG2RAD = M_PI / 180.0;
        current_state_(10) = msg->twist.twist.angular.x * DEG2RAD;
        current_state_(11) = msg->twist.twist.angular.y * DEG2RAD;
        current_state_(12) = msg->twist.twist.angular.z * DEG2RAD;
        odom_received_ = true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  PX4 callbacks
    // ─────────────────────────────────────────────────────────────────────────

#ifdef HAS_PX4_MSGS
    void px4OdomCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
    {
        Eigen::Vector3d pos_ned(msg->position[0], msg->position[1], msg->position[2]);
        Eigen::Vector3d vel_ned(msg->velocity[0], msg->velocity[1], msg->velocity[2]);
        Eigen::Vector3d omega_frd(msg->angular_velocity[0], msg->angular_velocity[1], msg->angular_velocity[2]);

        auto pos_enu   = frame_conv::ned_to_enu(pos_ned);
        auto vel_enu   = frame_conv::ned_to_enu(vel_ned);
        auto q_enu     = frame_conv::quat_ned_to_enu(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
        auto omega_flu = frame_conv::omega_frd_to_flu(omega_frd);

        std::lock_guard<std::mutex> lk(state_mutex_);
        current_state_(0)  = pos_enu.x();  current_state_(1)  = pos_enu.y();  current_state_(2)  = pos_enu.z();
        current_state_(3)  = vel_enu.x();  current_state_(4)  = vel_enu.y();  current_state_(5)  = vel_enu.z();
        current_state_(6)  = q_enu.w();    current_state_(7)  = q_enu.x();
        current_state_(8)  = q_enu.y();    current_state_(9)  = q_enu.z();
        current_state_(10) = omega_flu.x(); current_state_(11) = omega_flu.y(); current_state_(12) = omega_flu.z();
        pose_received_ = odom_received_ = true;
    }

    void px4HeartbeatCallback()
    {
        px4_msgs::msg::OffboardControlMode msg;
        msg.timestamp    = this->now().nanoseconds() / 1000;
        msg.position     = true;
        msg.velocity     = true;
        msg.acceleration = false;
        msg.attitude     = false;
        msg.body_rate    = false;
        px4_offboard_pub_->publish(msg);
    }

    void px4Arm()
    {
        px4_msgs::msg::VehicleCommand msg;
        msg.timestamp = this->now().nanoseconds() / 1000;
        msg.command   = 400; msg.param1 = 1.0;
        msg.target_system = msg.source_system = 1;
        msg.target_component = msg.source_component = 1;
        msg.from_external = true;
        px4_cmd_pub_->publish(msg);
    }

    void px4SetOffboardMode()
    {
        px4_msgs::msg::VehicleCommand msg;
        msg.timestamp = this->now().nanoseconds() / 1000;
        msg.command   = 176; msg.param1 = 1.0; msg.param2 = 6.0;
        msg.target_system = msg.source_system = 1;
        msg.target_component = msg.source_component = 1;
        msg.from_external = true;
        px4_cmd_pub_->publish(msg);
    }
#endif

    // ─────────────────────────────────────────────────────────────────────────
    //  MPC replay tick (runs at ocp_dt rate, separate from solver)
    //
    //  Implements Algorithm 1 line 9 from Zhang et al.:
    //    "output û(t) = u°_{i,j} for t_{i,j} ≤ t < min(t_{i,j+1}, t_{i+1,0})"
    //
    //  Between solver re-runs, the drone steps sequentially through the last
    //  committed trajectory.  When a new trajectory arrives (solver succeeded),
    //  mpc_replay_idx_ resets to 1 and we start replaying the fresh solution.
    // ─────────────────────────────────────────────────────────────────────────
    void mpcReplayTick()
    {
        Eigen::VectorXd x_cmd;
        Eigen::VectorXd u_cmd;
        bool have_traj = false;

        {
            std::lock_guard<std::mutex> lk(mpc_traj_mutex_);
            if (!mpc_traj_.empty()) {
                const int N = static_cast<int>(mpc_traj_.size()) - 1;
                // Clamp at last node — hold terminal position if replay runs out
                const int idx = std::min(mpc_replay_idx_, N);
                x_cmd      = mpc_traj_[idx];
                u_cmd      = (idx < (int)mpc_ctrl_.size()) ? mpc_ctrl_[idx]
                                                            : Eigen::VectorXd::Zero(4);
                have_traj  = true;
                if (mpc_replay_idx_ < N) {
                    ++mpc_replay_idx_;
                }
            }
        }

        if (!have_traj) {
            // No trajectory yet — hold hover at current measured position
            Eigen::VectorXd x_hover;
            {
                std::lock_guard<std::mutex> lk(state_mutex_);
                x_hover = current_state_;
            }
            x_hover.segment(3, 3).setZero();
            x_hover.segment(10, 3).setZero();
            if (platform_ == "crazyflie") publishCrazyflieCommand(x_hover, Eigen::Vector3d::Zero());
            else                          publishPX4Command(x_hover, Eigen::Vector3d::Zero());
            return;
        }

        if (platform_ == "crazyflie") publishCrazyflieCommand(x_cmd, Eigen::Vector3d::Zero());
        else                          publishPX4Command(x_cmd, Eigen::Vector3d::Zero());

        // Logging: actual state vs commanded
        if (logging_enabled_ && logging_initialized_) {
            Eigen::VectorXd act;
            { std::lock_guard<std::mutex> lk(state_mutex_); act = current_state_; }
            logActualState(act);
            logCommandedState(x_cmd, u_cmd);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Solver loop (runs at solver_rate Hz — may block for 100-500 ms)
    // ─────────────────────────────────────────────────────────────────────────
    void solverLoop()
    {
        bool have_pose, have_odom;
        Eigen::VectorXd x0;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            have_pose = pose_received_;
            have_odom = odom_received_;
            x0 = current_state_;
        }
        if (!have_pose || !have_odom) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for state: /pose [%s] /odom [%s]...",
                have_pose ? "OK" : "MISSING",
                have_odom ? "OK" : "MISSING");
            return;
        }

        if (!is_flying_) {
            is_flying_ = true;
            RCLCPP_INFO(this->get_logger(),
                "State feedback received — starting MPC (%s)  replay at %.0f Hz",
                solver_type_.c_str(), 1.0 / ocp_dt_);

#ifdef HAS_PX4_MSGS
            if (platform_ == "px4") { px4SetOffboardMode(); px4Arm(); }
#endif
            if (logging_enabled_ && !logging_initialized_) {
                setupLogging();
                logging_initialized_ = true;
            }

            RCLCPP_INFO(this->get_logger(),
                "x0=[%.3f,%.3f,%.3f | %.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f]",
                x0(0),x0(1),x0(2), x0(3),x0(4),x0(5),
                x0(6),x0(7),x0(8),x0(9), x0(10),x0(11),x0(12));
        }

        // ── Solve ─────────────────────────────────────────────────────────────
        SolverResult result = callSolver(x0);

        if (result.success) {
            const auto& X = result.state_trajectory;
            const auto& U = result.control_trajectory;

            // ── Update committed trajectory for replay timer ───────────────────
            // Reset replay index to 1 so the replay timer starts at the freshly
            // computed X[1] on its next tick.
            {
                std::lock_guard<std::mutex> lk(mpc_traj_mutex_);
                mpc_traj_ = X;
                mpc_ctrl_ = U;
                mpc_replay_idx_ = 1;   // start at X[1]; X[0] = x0_ocp (already passed)
            }

            publishTrajectory(X);

            if (logging_enabled_ && logging_initialized_) {
                const Eigen::VectorXd& u0 = U.empty() ? Eigen::VectorXd::Zero(4) : U[0];
                logSolveTrajectory(X, U, result.solve_time_ms);
            }

            if (++diag_count_ % 10 == 0) printDiagnostics(result);

        } else {
            RCLCPP_WARN(this->get_logger(), "MPC solve FAILED (%s) — replay continues on stale traj",
                solver_type_.c_str());
            // mpc_traj_ is NOT updated — replay timer continues on last good trajectory.
            // This gives graceful degradation: the drone keeps moving on the previous
            // plan rather than freezing.
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Unified solver call
    // ─────────────────────────────────────────────────────────────────────────
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

    // ═════════════════════════════════════════════════════════════════════════
    //  MODE: Open-loop — single solve then replay (unchanged)
    // ═════════════════════════════════════════════════════════════════════════

    void openLoopStartupCheck()
    {
        bool have_odom;
        { std::lock_guard<std::mutex> lk(state_mutex_); have_odom = odom_received_; }
        if (!have_odom) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[OpenLoop] Waiting for /odom...");
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

        ol_ref_X_ = result.state_trajectory;
        ol_ref_U_ = result.control_trajectory;

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoop] Solve OK in %.1f ms — %zu states, horizon=%.2fs",
            result.solve_time_ms, ol_ref_X_.size(),
            (ol_ref_X_.size()-1) * ocp_dt_);

        if (logging_enabled_) {
            setupLogging(); logging_initialized_ = true;
            logSolveTrajectory(ol_ref_X_, ol_ref_U_, result.solve_time_ms);
        }
        publishTrajectory(ol_ref_X_);

        ol_replay_step_       = 0;
        ol_replay_start_time_ = Clock::now();

        const int period_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
        replay_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&PlannerNode::openLoopReplayTick, this),
            solver_cb_group_);

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoop] Replay started at %dms/step (%zu steps)",
            period_ms, ol_ref_X_.size());
    }

    void openLoopReplayTick()
    {
        const int N    = static_cast<int>(ol_ref_X_.size()) - 1;
        const int step = std::min(ol_replay_step_, N);
        const bool done = (ol_replay_step_ >= N);

        const Eigen::VectorXd& x_cmd = ol_ref_X_[step];

        if (platform_ == "crazyflie") publishCrazyflieCommand(x_cmd, Eigen::Vector3d::Zero());
        else                          publishPX4Command(x_cmd, Eigen::Vector3d::Zero());

        Eigen::VectorXd act;
        { std::lock_guard<std::mutex> lk(state_mutex_); act = current_state_; }

        if (logging_enabled_ && logging_initialized_) {
            logActualState(act);
            const Eigen::VectorXd u = (step < (int)ol_ref_U_.size())
                ? ol_ref_U_[step] : Eigen::VectorXd::Zero(4);
            logCommandedState(x_cmd, u);
        }

        if (ol_replay_step_ % 10 == 0 || done) {
            double t_el = std::chrono::duration<double>(Clock::now() - ol_replay_start_time_).count();
            Eigen::Vector3d err = act.head(3) - x_cmd.head(3);
            RCLCPP_INFO(this->get_logger(),
                "[Step %3d/%d | t=%.2fs] cmd=[%.3f,%.3f,%.3f] act=[%.3f,%.3f,%.3f] |err|=%.3fm",
                step, N, t_el,
                x_cmd(0),x_cmd(1),x_cmd(2), act(0),act(1),act(2), err.norm());
        }

        if (done) {
            if (!ol_done_logged_) {
                RCLCPP_INFO(this->get_logger(), "[OpenLoop] Trajectory done. Holding X[%d].", N);
                ol_done_logged_ = true;
            }
        } else {
            ++ol_replay_step_;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Publishing
    // ─────────────────────────────────────────────────────────────────────────

    void publishCrazyflieCommand(const Eigen::VectorXd& s, const Eigen::Vector3d& acc_cmd)
    {
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";
        msg.pose.position.x     = s(0); msg.pose.position.y     = s(1); msg.pose.position.z     = s(2);
        msg.twist.linear.x      = s(3); msg.twist.linear.y      = s(4); msg.twist.linear.z      = s(5);
        msg.pose.orientation.w  = s(6); msg.pose.orientation.x  = s(7);
        msg.pose.orientation.y  = s(8); msg.pose.orientation.z  = s(9);
        msg.twist.angular.x     = s(10); msg.twist.angular.y    = s(11); msg.twist.angular.z    = s(12);
        msg.acc.x = acc_cmd(0); msg.acc.y = acc_cmd(1); msg.acc.z = acc_cmd(2);
        cf_cmd_pub_->publish(msg);

        if (published_commands_log_.is_open()) {
            published_commands_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
            for (int i = 0; i < 13; ++i) published_commands_log_ << "," << s(i);
            published_commands_log_ << "," << acc_cmd(0) << "," << acc_cmd(1) << "," << acc_cmd(2) << "\n";
            published_commands_log_.flush();
        }
    }

    void publishPX4Command(const Eigen::VectorXd& s,
                           [[maybe_unused]] const Eigen::Vector3d& acc_cmd)
    {
#ifdef HAS_PX4_MSGS
        Eigen::Vector3d pos_ned = frame_conv::enu_to_ned({s(0),s(1),s(2)});
        Eigen::Vector3d vel_ned = frame_conv::enu_to_ned({s(3),s(4),s(5)});
        Eigen::Quaterniond q_ned = frame_conv::quat_enu_to_ned({s(6),s(7),s(8),s(9)});
        double yaw_ned = std::atan2(
            2.0*(q_ned.w()*q_ned.z() + q_ned.x()*q_ned.y()),
            1.0 - 2.0*(q_ned.y()*q_ned.y() + q_ned.z()*q_ned.z()));

        px4_msgs::msg::TrajectorySetpoint sp;
        sp.timestamp   = this->now().nanoseconds() / 1000;
        sp.position[0] = pos_ned.x(); sp.position[1] = pos_ned.y(); sp.position[2] = pos_ned.z();
        sp.velocity[0] = vel_ned.x(); sp.velocity[1] = vel_ned.y(); sp.velocity[2] = vel_ned.z();
        sp.acceleration[0] = sp.acceleration[1] = sp.acceleration[2] = std::numeric_limits<float>::quiet_NaN();
        sp.jerk[0] = sp.jerk[1] = sp.jerk[2] = std::numeric_limits<float>::quiet_NaN();
        sp.yaw     = static_cast<float>(yaw_ned);
        sp.yawspeed = std::numeric_limits<float>::quiet_NaN();
        px4_setpoint_pub_->publish(sp);
#endif
    }

    void publishTrajectory(const std::vector<Eigen::VectorXd>& traj)
    {
        nav_msgs::msg::Path path;
        path.header.stamp = this->now(); path.header.frame_id = "world";
        for (const auto& s : traj) {
            geometry_msgs::msg::PoseStamped p;
            p.header.frame_id = "world";
            p.pose.position.x = s(0); p.pose.position.y = s(1); p.pose.position.z = s(2);
            p.pose.orientation.w = s(6); p.pose.orientation.x = s(7);
            p.pose.orientation.y = s(8); p.pose.orientation.z = s(9);
            path.poses.push_back(p);
        }
        traj_pub_->publish(path);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Logging
    // ─────────────────────────────────────────────────────────────────────────

    void setupLogging()
    {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::stringstream ts;
        ts << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");

        std::string folder = "./logs/" + drone_name_ + "_" + ocp_type_ + "_"
                           + mode_ + "_" + solver_type_ + "_" + ts.str();
        std::filesystem::create_directories(folder);
        log_folder_ = folder;

        commanded_state_log_.open(folder + "/commanded_state.csv");
        if (commanded_state_log_.is_open())
            commanded_state_log_ << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,fz,mx,my,mz\n";

        actual_state_log_.open(folder + "/actual_state.csv");
        if (actual_state_log_.is_open())
            actual_state_log_ << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";

        all_solves_log_.open(folder + "/all_solves.csv");
        if (all_solves_log_.is_open())
            all_solves_log_ << "solve_num,solve_time_ms,node,t,"
                               "x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,fz,mx,my,mz\n";

        published_commands_log_.open(folder + "/published_commands.csv");
        if (published_commands_log_.is_open())
            published_commands_log_ << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,acc_x,acc_y,acc_z\n";

        RCLCPP_INFO(this->get_logger(), "Logging to: %s", folder.c_str());
    }

    double wallTimeSec() {
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    void logSolveTrajectory(const std::vector<Eigen::VectorXd>& traj,
                            const std::vector<Eigen::VectorXd>& ctrls,
                            double solve_time_ms)
    {
        if (!all_solves_log_.is_open()) return;
        for (int i = 0; i < (int)traj.size(); ++i) {
            const auto& s = traj[i];
            if (s.size() < 13) continue;
            all_solves_log_ << std::fixed << std::setprecision(6)
                << solve_count_ << "," << solve_time_ms << "," << i << "," << (i * ocp_dt_);
            for (int j = 0; j < 13; ++j) all_solves_log_ << "," << s(j);
            if (i < (int)ctrls.size() && ctrls[i].size() >= 4)
                all_solves_log_ << "," << ctrls[i](0) << "," << ctrls[i](1)
                                << "," << ctrls[i](2) << "," << ctrls[i](3);
            else all_solves_log_ << ",0,0,0,0";
            all_solves_log_ << "\n";
        }
        all_solves_log_.flush();
        ++solve_count_;
    }

    void logActualState(const Eigen::VectorXd& state)
    {
        if (!actual_state_log_.is_open()) return;
        actual_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) actual_state_log_ << "," << state(i);
        actual_state_log_ << "\n";
        actual_state_log_.flush();
    }

    void logCommandedState(const Eigen::VectorXd& s, const Eigen::VectorXd& u)
    {
        if (!commanded_state_log_.is_open() || s.size() < 13) return;
        commanded_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) commanded_state_log_ << "," << s(i);
        if (u.size() >= 4) commanded_state_log_ << "," << u(0) << "," << u(1) << "," << u(2) << "," << u(3);
        else commanded_state_log_ << ",0,0,0,0";
        commanded_state_log_ << "\n";
        commanded_state_log_.flush();
    }

    void printDiagnostics(const SolverResult& result)
    {
        Eigen::VectorXd snap;
        { std::lock_guard<std::mutex> lk(state_mutex_); snap = current_state_; }
        int replay_idx;
        { std::lock_guard<std::mutex> lk(mpc_traj_mutex_); replay_idx = mpc_replay_idx_; }
        double thrust = result.control_trajectory.empty() ? 0.0 : result.control_trajectory[0](0);
        RCLCPP_INFO(this->get_logger(),
            "[%s] %.1fms | pos[%.3f,%.3f,%.3f] vel[%.3f,%.3f,%.3f] "
            "w[%.3f,%.3f,%.3f]rad/s | T=%.3fN | replay_idx=%d",
            solver_type_.c_str(), result.solve_time_ms,
            snap(0),snap(1),snap(2), snap(3),snap(4),snap(5),
            snap(10),snap(11),snap(12), thrust, replay_idx);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Members
    // ─────────────────────────────────────────────────────────────────────────

    // Config
    std::string ocp_type_, platform_, solver_type_, mode_, drone_name_;
    bool        logging_enabled_;
    int         solver_rate_;
    double      ocp_dt_  = 0.1;
    int         n_shift_ = 1;

    // Solvers
    std::unique_ptr<QuadrotorMPC> alipddp_mpc_;
#ifdef HAS_ACADOS
    std::unique_ptr<AcadosMPC>    acados_mpc_;
#endif

    // Crazyflie
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cf_cmd_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr   pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           odom_sub_;

    // PX4
#ifdef HAS_PX4_MSGS
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr  px4_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr px4_offboard_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr      px4_cmd_pub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr  px4_odom_sub_;
    rclcpp::TimerBase::SharedPtr                                     px4_heartbeat_timer_;
#endif

    // Shared publishers/timers
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
    rclcpp::TimerBase::SharedPtr solver_timer_;
    rclcpp::TimerBase::SharedPtr mpc_replay_timer_;   // NEW — replay at ocp_dt rate
    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr replay_timer_;       // open_loop mode only

    // State (sensor callbacks write, solver reads)
    Eigen::VectorXd current_state_;
    bool pose_received_ = false;
    bool odom_received_ = false;
    bool is_flying_     = false;

    // MPC committed trajectory — written by solverLoop, read by mpcReplayTick
    // Protected by mpc_traj_mutex_ (separate from state_mutex_ to avoid
    // blocking sensor callbacks during replay reads).
    std::mutex                   mpc_traj_mutex_;
    std::vector<Eigen::VectorXd> mpc_traj_;   // X[0..N]
    std::vector<Eigen::VectorXd> mpc_ctrl_;   // U[0..N-1]
    int                          mpc_replay_idx_ = 0;  // next node to publish

    // Threading
    mutable std::mutex               state_mutex_;
    rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
    rclcpp::CallbackGroup::SharedPtr solver_cb_group_;
    rclcpp::CallbackGroup::SharedPtr replay_cb_group_;   // NEW — replay group

    // MPC diagnostics
    int diag_count_ = 0;

    // Open-loop
    std::vector<Eigen::VectorXd> ol_ref_X_, ol_ref_U_;
    int ol_replay_step_ = 0;
    bool ol_done_logged_ = false;
    Clock::time_point ol_replay_start_time_;

    // Logging
    bool          logging_initialized_ = false;
    std::string   log_folder_;
    std::ofstream commanded_state_log_, actual_state_log_, all_solves_log_, published_commands_log_;
    int           solve_count_ = 0;
};

// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}