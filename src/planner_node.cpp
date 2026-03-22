#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>

#ifdef HAS_PX4_MSGS
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#endif

#include "quadrotor_mpc.hpp"
#include "ocp_registry.hpp"
#ifdef HAS_ACADOS
#include "acados_solver.hpp"
#endif

#include <chrono>
#include <thread>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <Eigen/Dense>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace frame_conv {
inline Eigen::Vector3d ned_to_enu(const Eigen::Vector3d& v) { return {v.y(), v.x(), -v.z()}; }
inline Eigen::Vector3d enu_to_ned(const Eigen::Vector3d& v) { return {v.y(), v.x(), -v.z()}; }
inline Eigen::Quaterniond quat_ned_to_enu(double w, double x, double y, double z) {
    return Eigen::Quaterniond(w, y, x, -z).normalized();
}
inline Eigen::Quaterniond quat_enu_to_ned(const Eigen::Quaterniond& q) {
    return Eigen::Quaterniond(q.w(), q.y(), q.x(), -q.z()).normalized();
}
inline Eigen::Vector3d omega_frd_to_flu(const Eigen::Vector3d& w) { return {w.x(), -w.y(), -w.z()}; }
}

struct SolverResult {
    bool                          success   = false;
    Eigen::VectorXd               next_state;
    std::vector<Eigen::VectorXd>  state_trajectory;
    std::vector<Eigen::VectorXd>  control_trajectory;
    double                        solve_time_ms = 0.0;
    std::chrono::steady_clock::time_point solve_timestamp;
};

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        this->declare_parameter("ocp_type",       std::string("landing"));
        this->declare_parameter("drone_name",     std::string("cf_1"));
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("platform",       std::string("crazyflie"));
        this->declare_parameter("solver",         std::string("alipddp"));
        this->declare_parameter("mode",           std::string("mpc"));
        this->declare_parameter("hover_target_x", 0.0);
        this->declare_parameter("hover_target_y", 0.0);
        this->declare_parameter("hover_target_z", 1.0);
        // Fixed number of trajectory nodes to replay between solves.
        // During replay the drone tracks X[1..n_replay] at ocp_dt spacing.
        // n_shift in QuadrotorMPC is set to the same value so warm-start
        // U alignment is exact.
        // At ocp_dt=0.05s: n_replay=4 → 200ms of tracking per solve cycle.
        this->declare_parameter("n_replay", 4);

        ocp_type_        = this->get_parameter("ocp_type").as_string();
        drone_name_      = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        platform_        = this->get_parameter("platform").as_string();
        solver_type_     = this->get_parameter("solver").as_string();
        mode_            = this->get_parameter("mode").as_string();
        n_replay_        = this->get_parameter("n_replay").as_int();

        double tx = this->get_parameter("hover_target_x").as_double();
        double ty = this->get_parameter("hover_target_y").as_double();
        double tz = this->get_parameter("hover_target_z").as_double();

        ocp_dt_ = OCPRegistry::getDT(ocp_type_);

        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(13);
        terminal(0) = tx; terminal(1) = ty; terminal(2) = tz;
        terminal(6) = 1.0;

        if (solver_type_ == "alipddp") {
            QuadrotorMPC::Config cfg;
            cfg.ocp_type       = ocp_type_;
            cfg.terminal_state = terminal;
            cfg.n_shift        = n_replay_;   // fixed: matches replay count
            alipddp_mpc_ = std::make_unique<QuadrotorMPC>(cfg);
        }
#ifdef HAS_ACADOS
        else if (solver_type_ == "acados") {
            AcadosMPC::Config cfg;
            cfg.ocp_type       = ocp_type_;
            cfg.terminal_state = terminal;
            cfg.dt             = ocp_dt_;
            cfg.n_shift        = n_replay_;
            acados_mpc_ = std::make_unique<AcadosMPC>(cfg);
        }
#endif

        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;

        sensor_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        solver_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        replay_cb_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        if (platform_ == "crazyflie") setupCrazyflie();
        else                          setupPX4();

        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);

        if (mode_ == "mpc") {
            // Solver timer: runs as fast as possible, blocks on each solve.
            // When done swaps mpc_traj_ and resets replay idx to 1.
            solver_timer_ = this->create_wall_timer(
                1ms,
                std::bind(&PlannerNode::solverLoop, this),
                solver_cb_group_);

            // Replay timer: fires every ocp_dt (50ms) independently.
            // Streams mpc_traj_[idx++] with no gaps — no sleep in solver.
            const int replay_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
            mpc_replay_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(replay_ms),
                std::bind(&PlannerNode::mpcReplayTick, this),
                replay_cb_group_);
        } else {
            startup_timer_ = this->create_wall_timer(
                50ms, std::bind(&PlannerNode::openLoopStartupCheck, this),
                solver_cb_group_);
        }

        RCLCPP_INFO(this->get_logger(),
            "Ready  mode=%s  platform=%s  solver=%s  ocp=%s  "
            "ocp_dt=%.3fs  n_replay=%d  replay_period=%.0fms",
            mode_.c_str(), platform_.c_str(), solver_type_.c_str(),
            ocp_type_.c_str(), ocp_dt_, n_replay_, n_replay_ * ocp_dt_ * 1000.0);
        RCLCPP_INFO(this->get_logger(), "Target: [%.3f, %.3f, %.3f]", tx, ty, tz);
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    void setupCrazyflie()
    {
        cf_cmd_pub_ = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);

        rclcpp::SubscriptionOptions opts;
        opts.callback_group = sensor_cb_group_;

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            std::bind(&PlannerNode::cfPoseCallback, this, std::placeholders::_1), opts);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + drone_name_ + "/odom", 10,
            std::bind(&PlannerNode::cfOdomCallback, this, std::placeholders::_1), opts);
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

        rclcpp::SubscriptionOptions opts;
        opts.callback_group = sensor_cb_group_;
        px4_odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", 10,
            std::bind(&PlannerNode::px4OdomCallback, this, std::placeholders::_1), opts);

        px4_heartbeat_timer_ = this->create_wall_timer(
            100ms, std::bind(&PlannerNode::px4HeartbeatCallback, this));
#endif
    }

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

#ifdef HAS_PX4_MSGS
    void px4OdomCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
    {
        auto pos = frame_conv::ned_to_enu({msg->position[0], msg->position[1], msg->position[2]});
        auto vel = frame_conv::ned_to_enu({msg->velocity[0], msg->velocity[1], msg->velocity[2]});
        auto q   = frame_conv::quat_ned_to_enu(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
        auto w   = frame_conv::omega_frd_to_flu({msg->angular_velocity[0], msg->angular_velocity[1], msg->angular_velocity[2]});
        std::lock_guard<std::mutex> lk(state_mutex_);
        current_state_ << pos.x(), pos.y(), pos.z(),
                          vel.x(), vel.y(), vel.z(),
                          q.w(), q.x(), q.y(), q.z(),
                          w.x(), w.y(), w.z();
        pose_received_ = odom_received_ = true;
    }
    void px4HeartbeatCallback()
    {
        px4_msgs::msg::OffboardControlMode msg;
        msg.timestamp = this->now().nanoseconds() / 1000;
        msg.position = msg.velocity = true;
        px4_offboard_pub_->publish(msg);
    }
    void px4Arm()
    {
        px4_msgs::msg::VehicleCommand msg;
        msg.timestamp = this->now().nanoseconds() / 1000;
        msg.command = 400; msg.param1 = 1.0;
        msg.target_system = msg.source_system = 1;
        msg.target_component = msg.source_component = 1;
        msg.from_external = true;
        px4_cmd_pub_->publish(msg);
    }
    void px4SetOffboardMode()
    {
        px4_msgs::msg::VehicleCommand msg;
        msg.timestamp = this->now().nanoseconds() / 1000;
        msg.command = 176; msg.param1 = 1.0; msg.param2 = 6.0;
        msg.target_system = msg.source_system = 1;
        msg.target_component = msg.source_component = 1;
        msg.from_external = true;
        px4_cmd_pub_->publish(msg);
    }
#endif

    // ─────────────────────────────────────────────────────────────────────────
    //  SOLVER LOOP — blocks on solve, then swaps trajectory buffer.
    //  No sleep. Replay timer handles all command streaming independently.
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
                "Waiting for state: pose[%s] odom[%s]",
                have_pose ? "OK" : "MISSING", have_odom ? "OK" : "MISSING");
            return;
        }

        if (!is_flying_) {
            is_flying_ = true;
            RCLCPP_INFO(this->get_logger(),
                "State received — starting RH MPC (%s)", solver_type_.c_str());
#ifdef HAS_PX4_MSGS
            if (platform_ == "px4") { px4SetOffboardMode(); px4Arm(); }
#endif
            if (logging_enabled_ && !logging_initialized_) {
                setupLogging();
                logging_initialized_ = true;
            }
            RCLCPP_INFO(this->get_logger(),
                "x0=[%.3f,%.3f,%.3f | %.3f,%.3f,%.3f]",
                x0(0),x0(1),x0(2), x0(3),x0(4),x0(5));
        }

        // Blocks for solve duration. Replay timer keeps firing during this time.
        SolverResult result = callSolver(x0);

        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "Solve FAILED — holding");
            Eigen::VectorXd x_hold = x0;
            x_hold.segment(3,3).setZero();
            x_hold.segment(10,3).setZero();
            publishCommand(x_hold, Eigen::VectorXd::Zero(4));
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "[RH %d] %.1fms  x0=[%.3f,%.3f,%.3f]",
            solve_count_, result.solve_time_ms,
            x0(0), x0(1), x0(2));

        // Swap — replay timer picks up from idx=1 on next tick.
        {
            std::lock_guard<std::mutex> lk(mpc_traj_mutex_);
            mpc_traj_       = result.state_trajectory;
            mpc_ctrl_       = result.control_trajectory;
            mpc_replay_idx_ = 1;
        }

        publishTrajectory(result.state_trajectory);

        if (logging_enabled_ && logging_initialized_)
            logSolveTrajectory(result.state_trajectory,
                               result.control_trajectory,
                               result.solve_time_ms);

        ++solve_count_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  REPLAY TICK — fires every ocp_dt (50ms), streams mpc_traj_[idx++].
    //  Runs independent of solver — no gaps ever.
    // ─────────────────────────────────────────────────────────────────────────
    void mpcReplayTick()
    {
        Eigen::VectorXd x_cmd, u_cmd;
        {
            std::lock_guard<std::mutex> lk(mpc_traj_mutex_);
            if (mpc_traj_.empty()) return;

            const int N   = (int)mpc_traj_.size() - 1;
            const int idx = std::min(mpc_replay_idx_, N);
            x_cmd = mpc_traj_[idx];
            u_cmd = (idx < (int)mpc_ctrl_.size())
                ? mpc_ctrl_[idx] : Eigen::VectorXd::Zero(4);

            if (mpc_replay_idx_ < N) ++mpc_replay_idx_;
        }

        publishCommand(x_cmd, u_cmd);

        if (logging_enabled_ && logging_initialized_) {
            Eigen::VectorXd act;
            { std::lock_guard<std::mutex> lk(state_mutex_); act = current_state_; }
            logActualState(act);
            logCommandedState(x_cmd, u_cmd);
        }
    }

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

    // ─────────────────────────────────────────────────────────────────────────
    void publishCommand(const Eigen::VectorXd& s, const Eigen::VectorXd& u)
    {
        if (platform_ == "crazyflie") publishCrazyflieCommand(s, u);
        else                          publishPX4Command(s);
    }

    void publishCrazyflieCommand(const Eigen::VectorXd& s, const Eigen::VectorXd& u)
    {
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";
        msg.pose.position.x    = s(0); msg.pose.position.y    = s(1); msg.pose.position.z    = s(2);
        msg.twist.linear.x     = s(3); msg.twist.linear.y     = s(4); msg.twist.linear.z     = s(5);
        msg.pose.orientation.w = s(6); msg.pose.orientation.x = s(7);
        msg.pose.orientation.y = s(8); msg.pose.orientation.z = s(9);
        msg.twist.angular.x    = s(10); msg.twist.angular.y   = s(11); msg.twist.angular.z   = s(12);
        msg.acc.x = 0; msg.acc.y = 0; msg.acc.z = 0;
        cf_cmd_pub_->publish(msg);

        if (published_commands_log_.is_open()) {
            published_commands_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
            for (int i = 0; i < 13; ++i) published_commands_log_ << "," << s(i);
            published_commands_log_ << ",0,0,0\n";
            published_commands_log_.flush();
        }
    }

    void publishPX4Command(const Eigen::VectorXd& s)
    {
#ifdef HAS_PX4_MSGS
        auto pos = frame_conv::enu_to_ned({s(0),s(1),s(2)});
        auto vel = frame_conv::enu_to_ned({s(3),s(4),s(5)});
        auto q   = frame_conv::quat_enu_to_ned(Eigen::Quaterniond(s(6),s(7),s(8),s(9)));
        double yaw = std::atan2(2*(q.w()*q.z()+q.x()*q.y()),
                                1-2*(q.y()*q.y()+q.z()*q.z()));
        px4_msgs::msg::TrajectorySetpoint sp;
        sp.timestamp   = this->now().nanoseconds() / 1000;
        sp.position[0] = pos.x(); sp.position[1] = pos.y(); sp.position[2] = pos.z();
        sp.velocity[0] = vel.x(); sp.velocity[1] = vel.y(); sp.velocity[2] = vel.z();
        sp.acceleration[0] = sp.acceleration[1] = sp.acceleration[2] = std::numeric_limits<float>::quiet_NaN();
        sp.jerk[0] = sp.jerk[1] = sp.jerk[2] = std::numeric_limits<float>::quiet_NaN();
        sp.yaw = static_cast<float>(yaw);
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
    //  Open-loop mode (unchanged)
    // ─────────────────────────────────────────────────────────────────────────
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
        RCLCPP_INFO(this->get_logger(), "[OpenLoop] Solving...");
        SolverResult result = callSolver(x0);
        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_ERROR(this->get_logger(), "[OpenLoop] FAILED"); return;
        }
        ol_ref_X_ = result.state_trajectory;
        ol_ref_U_ = result.control_trajectory;
        if (logging_enabled_) { setupLogging(); logging_initialized_ = true;
            logSolveTrajectory(ol_ref_X_, ol_ref_U_, result.solve_time_ms); }
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

    void openLoopReplayTick()
    {
        const int N = (int)ol_ref_X_.size() - 1;
        const int step = std::min(ol_replay_step_, N);
        publishCommand(ol_ref_X_[step],
            (step < (int)ol_ref_U_.size()) ? ol_ref_U_[step] : Eigen::VectorXd::Zero(4));
        if (logging_enabled_ && logging_initialized_) {
            Eigen::VectorXd act; { std::lock_guard<std::mutex> lk(state_mutex_); act = current_state_; }
            logActualState(act); logCommandedState(ol_ref_X_[step],
                (step < (int)ol_ref_U_.size()) ? ol_ref_U_[step] : Eigen::VectorXd::Zero(4));
        }
        if (ol_replay_step_ < N) ++ol_replay_step_;
        else if (!ol_done_logged_) {
            RCLCPP_INFO(this->get_logger(), "[OpenLoop] Done. Holding.");
            ol_done_logged_ = true;
        }
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
        if (u.size() >= 4) commanded_state_log_ << "," << u(0) << "," << u(1)
                                                 << "," << u(2) << "," << u(3);
        else commanded_state_log_ << ",0,0,0,0";
        commanded_state_log_ << "\n";
        commanded_state_log_.flush();
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::string ocp_type_, platform_, solver_type_, mode_, drone_name_;
    bool        logging_enabled_;
    double      ocp_dt_ = 0.05;
    int         n_replay_ = 4;   // nodes replayed between solves == n_shift

    std::unique_ptr<QuadrotorMPC> alipddp_mpc_;
#ifdef HAS_ACADOS
    std::unique_ptr<AcadosMPC>    acados_mpc_;
#endif

    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cf_cmd_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr   pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           odom_sub_;

#ifdef HAS_PX4_MSGS
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr  px4_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr px4_offboard_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr      px4_cmd_pub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr  px4_odom_sub_;
    rclcpp::TimerBase::SharedPtr                                     px4_heartbeat_timer_;
#endif

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
    rclcpp::TimerBase::SharedPtr solver_timer_;
    rclcpp::TimerBase::SharedPtr mpc_replay_timer_;
    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr replay_timer_;

    // Shared trajectory buffer between solver and replay timer
    std::mutex                   mpc_traj_mutex_;
    std::vector<Eigen::VectorXd> mpc_traj_;
    std::vector<Eigen::VectorXd> mpc_ctrl_;
    int                          mpc_replay_idx_ = 1;

    Eigen::VectorXd current_state_;
    bool pose_received_ = false;
    bool odom_received_ = false;
    bool is_flying_     = false;

    mutable std::mutex               state_mutex_;
    rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
    rclcpp::CallbackGroup::SharedPtr solver_cb_group_;
    rclcpp::CallbackGroup::SharedPtr replay_cb_group_;

    int solve_count_ = 0;

    std::vector<Eigen::VectorXd> ol_ref_X_, ol_ref_U_;
    int ol_replay_step_ = 0;
    bool ol_done_logged_ = false;
    Clock::time_point ol_replay_start_time_;

    bool          logging_initialized_ = false;
    std::string   log_folder_;
    std::ofstream commanded_state_log_, actual_state_log_,
                  all_solves_log_, published_commands_log_;
};

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