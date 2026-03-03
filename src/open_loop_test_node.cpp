// open_loop_test_node.cpp
// ─────────────────────────────────────────────────────────────────────────────
// OPEN-LOOP DIVERGENCE TEST
//
// PURPOSE:
//   The receding-horizon MPC is diverging. This node isolates whether the
//   problem is in the MPC feedback loop (state estimation noise, warm-start
//   corruption, etc.) or whether the first solve itself is physically wrong.
//
// HOW IT WORKS:
//   1. Wait for the first valid state from the drone.
//   2. Run the MPC solver EXACTLY ONCE on that initial state.
//   3. Store the full open-loop trajectory  X[0..N]  and controls  U[0..N-1].
//   4. At every ocp_dt tick, replay:
//        - Publish X[k] as the commanded setpoint (position, velocity,
//          attitude, angular rate, feedforward acc) — same FullState msg
//          the real planner sends.
//        - Record the actual drone state from /odom at that same tick.
//   5. Write a single comparison CSV:
//        step, t, cmd_x..cmd_wz, act_x..act_wz
//      so you can plot commanded vs actual side-by-side.
//
// INTERPRETATION:
//   - If actual ≈ commanded  → first-solve trajectory is physically correct;
//     the divergence is in the receding-horizon feedback logic.
//   - If actual diverges from commanded even on the first open-loop pass  →
//     the OCP dynamics model, mass, or frame convention is wrong; fix the
//     model before debugging the feedback.
//
// IMPORTANT:
//   - The node fires at ocp_dt, NOT at solver_rate. This ensures one command
//     is published per OCP time step, matching what the MPC assumes.
//   - After the trajectory ends the last state X[N] is held indefinitely so
//     the drone does not drop.
//   - Only the FIRST solve is ever executed; the solver timer is cancelled
//     immediately after.
//
// State layout (13-dim, ENU/FLU):
//   [0-2]   position        x, y, z       (m)
//   [3-5]   velocity        vx, vy, vz    (m/s)
//   [6-9]   quaternion      qw, qx, qy, qz
//   [10-12] angular rate    wx, wy, wz    (rad/s)
// ─────────────────────────────────────────────────────────────────────────────

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

#include <chrono>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

// ─────────────────────────────────────────────────────────────────────────────
// Frame conversion helpers (copied from planner_node.cpp)
// ─────────────────────────────────────────────────────────────────────────────
namespace frame_conv {

inline Eigen::Vector3d ned_to_enu(const Eigen::Vector3d& v) {
    return {v.y(), v.x(), -v.z()};
}
inline Eigen::Vector3d enu_to_ned(const Eigen::Vector3d& v) {
    return {v.y(), v.x(), -v.z()};
}
inline Eigen::Quaterniond quat_ned_to_enu(double w, double x, double y, double z) {
    return Eigen::Quaterniond(w, y, x, -z).normalized();
}
inline Eigen::Quaterniond quat_enu_to_ned(const Eigen::Quaterniond& q) {
    return Eigen::Quaterniond(q.w(), q.y(), q.x(), -q.z()).normalized();
}
inline Eigen::Vector3d omega_frd_to_flu(const Eigen::Vector3d& w) {
    return {w.x(), -w.y(), -w.z()};
}

} // namespace frame_conv


// ═══════════════════════════════════════════════════════════════════════════════
class OpenLoopTestNode : public rclcpp::Node {
public:
    OpenLoopTestNode() : Node("open_loop_test_node")
    {
        // ── Parameters (same names as planner_node for easy reuse) ────────────
        this->declare_parameter("ocp_type",       std::string("hover"));
        this->declare_parameter("drone_name",     std::string("cf_1"));
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("platform",       std::string("crazyflie"));
        this->declare_parameter("hover_target_x", 0.0);
        this->declare_parameter("hover_target_y", 0.0);
        this->declare_parameter("hover_target_z", 1.0);

        ocp_type_        = this->get_parameter("ocp_type").as_string();
        drone_name_      = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        platform_        = this->get_parameter("platform").as_string();

        double tx = this->get_parameter("hover_target_x").as_double();
        double ty = this->get_parameter("hover_target_y").as_double();
        double tz = this->get_parameter("hover_target_z").as_double();

        // ── OCP time step drives the replay rate ──────────────────────────────
        ocp_dt_ = OCPRegistry::getDT(ocp_type_);

        // ── Terminal state ────────────────────────────────────────────────────
        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(13);
        terminal(0) = tx;
        terminal(1) = ty;
        terminal(2) = tz;
        terminal(6) = 1.0;  // qw = 1 (identity quaternion)

        // ── MPC solver (used for one shot only) ───────────────────────────────
        QuadrotorMPC::Config cfg;
        cfg.ocp_type       = ocp_type_;
        cfg.terminal_state = terminal;
        cfg.n_shift        = 1;  // irrelevant — only one solve
        mpc_ = std::make_unique<QuadrotorMPC>(cfg);

        // ── Initial state placeholder ─────────────────────────────────────────
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;  // identity quaternion

        // ── Set up state subscribers ──────────────────────────────────────────
        if (platform_ == "crazyflie") {
            setupCrazyflie();
        }
#ifdef HAS_PX4_MSGS
        else if (platform_ == "px4") {
            setupPX4();
        }
#endif
        else {
            throw std::runtime_error("Unknown platform: " + platform_);
        }

        // ── Trajectory visualisation ──────────────────────────────────────────
        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/open_loop_trajectory", 10);

        // ── Startup timer: waits for first state, then fires ONE solve ─────────
        // We poll at 20 Hz until odom arrives; once it does we cancel this
        // timer and start the replay timer.
        startup_timer_ = this->create_wall_timer(
            50ms, std::bind(&OpenLoopTestNode::startupCheck, this));

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoopTest] platform=%s  ocp=%s  ocp_dt=%.3fs  target=[%.2f,%.2f,%.2f]",
            platform_.c_str(), ocp_type_.c_str(), ocp_dt_, tx, ty, tz);
        RCLCPP_INFO(this->get_logger(),
            "[OpenLoopTest] Waiting for first state feedback...");
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    //  Platform setup
    // ─────────────────────────────────────────────────────────────────────────

    void setupCrazyflie()
    {
        cf_cmd_pub_ = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);

        // /pose as fallback until /odom arrives
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                if (!odom_received_) {
                    current_state_(0) = msg->pose.position.x;
                    current_state_(1) = msg->pose.position.y;
                    current_state_(2) = msg->pose.position.z;
                    current_state_(6) = msg->pose.orientation.w;
                    current_state_(7) = msg->pose.orientation.x;
                    current_state_(8) = -msg->pose.orientation.y;  // FRD→FLU
                    current_state_(9) = -msg->pose.orientation.z;  // FRD→FLU
                    pose_received_ = true;
                }
            });

        // /odom: authoritative source for all 13 states
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + drone_name_ + "/odom", 10,
            std::bind(&OpenLoopTestNode::cfOdomCallback, this, std::placeholders::_1));
    }

#ifdef HAS_PX4_MSGS
    void setupPX4()
    {
        px4_setpoint_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
            "/fmu/in/trajectory_setpoint", 10);
        px4_offboard_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
            "/fmu/in/offboard_control_mode", 10);
        px4_cmd_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
            "/fmu/in/vehicle_command", 10);

        px4_odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", 10,
            std::bind(&OpenLoopTestNode::px4OdomCallback, this, std::placeholders::_1));

        // Heartbeat required to stay in offboard mode
        px4_heartbeat_timer_ = this->create_wall_timer(
            100ms, [this]() {
                px4_msgs::msg::OffboardControlMode msg;
                msg.timestamp = this->now().nanoseconds() / 1000;
                msg.position  = true;
                msg.velocity  = true;
                px4_offboard_pub_->publish(msg);
            });
    }
#endif

    // ─────────────────────────────────────────────────────────────────────────
    //  State callbacks
    // ─────────────────────────────────────────────────────────────────────────

    void cfOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // All 13 states from one message — temporally consistent
        current_state_(0) = msg->pose.pose.position.x;
        current_state_(1) = msg->pose.pose.position.y;
        current_state_(2) = msg->pose.pose.position.z;
        current_state_(3) = msg->twist.twist.linear.x;
        current_state_(4) = msg->twist.twist.linear.y;
        current_state_(5) = msg->twist.twist.linear.z;
        current_state_(6) = msg->pose.pose.orientation.w;
        current_state_(7) = msg->pose.pose.orientation.x;
        current_state_(8) = -msg->pose.pose.orientation.y;   // FRD→FLU
        current_state_(9) = -msg->pose.pose.orientation.z;   // FRD→FLU
        constexpr double DEG2RAD = M_PI / 180.0;
        current_state_(10) = msg->twist.twist.angular.x * DEG2RAD;
        current_state_(11) = -msg->twist.twist.angular.y * DEG2RAD;  // FRD→FLU
        current_state_(12) = -msg->twist.twist.angular.z * DEG2RAD;  // FRD→FLU
        pose_received_ = true;
        odom_received_ = true;
    }

#ifdef HAS_PX4_MSGS
    void px4OdomCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
    {
        Eigen::Vector3d pos_enu  = frame_conv::ned_to_enu({msg->position[0], msg->position[1], msg->position[2]});
        Eigen::Vector3d vel_enu  = frame_conv::ned_to_enu({msg->velocity[0], msg->velocity[1], msg->velocity[2]});
        Eigen::Quaterniond q_enu = frame_conv::quat_ned_to_enu(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
        Eigen::Vector3d omega    = frame_conv::omega_frd_to_flu({msg->angular_velocity[0],
                                                                  msg->angular_velocity[1],
                                                                  msg->angular_velocity[2]});
        current_state_ << pos_enu, vel_enu,
                          q_enu.w(), q_enu.x(), q_enu.y(), q_enu.z(),
                          omega;
        pose_received_ = true;
        odom_received_ = true;
    }
#endif

    // ─────────────────────────────────────────────────────────────────────────
    //  Startup: wait for first state, then solve once
    // ─────────────────────────────────────────────────────────────────────────

    void startupCheck()
    {
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[OpenLoopTest] Waiting for /odom...");
            return;
        }

        // Cancel the polling timer — we won't need it again
        startup_timer_->cancel();

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoopTest] First state received. Running ONE MPC solve...");
        RCLCPP_INFO(this->get_logger(),
            "[OpenLoopTest] x0 = [%.3f, %.3f, %.3f | %.3f, %.3f, %.3f | "
            "%.3f, %.3f, %.3f, %.3f | %.3f, %.3f, %.3f]",
            current_state_(0), current_state_(1), current_state_(2),
            current_state_(3), current_state_(4), current_state_(5),
            current_state_(6), current_state_(7), current_state_(8), current_state_(9),
            current_state_(10), current_state_(11), current_state_(12));

        // ── Single solve ──────────────────────────────────────────────────────
        auto result = mpc_->solve(current_state_);

        if (!result.success || result.state_trajectory.size() < 2) {
            RCLCPP_ERROR(this->get_logger(),
                "[OpenLoopTest] MPC solve FAILED. Cannot run test.");
            return;
        }

        // Store the open-loop plan — these are NEVER updated again
        ref_X_ = result.state_trajectory;   // X[0..N]
        ref_U_ = result.control_trajectory; // U[0..N-1]
        solve_time_ms_ = result.solve_time_ms;

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoopTest] Solve OK in %.1f ms — %zu states, %zu controls, "
            "horizon = %.2f s",
            solve_time_ms_,
            ref_X_.size(), ref_U_.size(),
            (ref_X_.size() - 1) * ocp_dt_);

        // ── Log the planned trajectory before we start executing ──────────────
        if (logging_enabled_) {
            setupLogging();
            logPlannedTrajectory();
        }

        // ── Publish the full trajectory path for RViz ─────────────────────────
        publishTrajectoryViz(ref_X_);

        // ── Start the replay timer at exactly ocp_dt ──────────────────────────
        // Each tick publishes ref_X_[step_] and records current_state_.
        replay_step_ = 0;
        replay_start_time_ = Clock::now();

        const int period_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
        replay_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&OpenLoopTestNode::replayTick, this));

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoopTest] Replay started at %d ms / step (%zu steps total)",
            period_ms, ref_X_.size());
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Replay loop — fires at ocp_dt
    // ─────────────────────────────────────────────────────────────────────────

    void replayTick()
    {
        // Clamp step to last state so we hold the final position
        const int N = static_cast<int>(ref_X_.size()) - 1;  // last valid index
        const int step = std::min(replay_step_, N);
        const bool trajectory_done = (replay_step_ >= N);

        // ── Commanded state at this step ──────────────────────────────────────
        const Eigen::VectorXd& x_cmd = ref_X_[step];

        // Feedforward acceleration from the corresponding control
        // u = [fz_B, Mx, My, Mz] — only thrust (u(0)) contributes to acceleration
        Eigen::Vector3d acc_ff = Eigen::Vector3d::Zero();
        if (step < static_cast<int>(ref_U_.size()) && ref_U_[step].size() >= 1) {
            constexpr double CF_MASS = 0.027;  // kg
            Eigen::Quaterniond q_cmd(x_cmd(6), x_cmd(7), x_cmd(8), x_cmd(9));
            q_cmd.normalize();
            Eigen::Vector3d f_body(0.0, 0.0, ref_U_[step](0));  // thrust along body-z only
            acc_ff = q_cmd.toRotationMatrix() * f_body / CF_MASS;
            acc_ff.z() -= 9.81;  // Mellinger adds gravity back
        }

        // ── Publish command ───────────────────────────────────────────────────
        if (platform_ == "crazyflie") {
            publishCrazyflieCommand(x_cmd, acc_ff);
        }
#ifdef HAS_PX4_MSGS
        else if (platform_ == "px4") {
            publishPX4Command(x_cmd, acc_ff);
        }
#endif

        // ── Log comparison: commanded vs actual ───────────────────────────────
        if (logging_enabled_ && comparison_log_.is_open()) {
            logComparison(step, x_cmd);
        }

        // ── Progress ──────────────────────────────────────────────────────────
        if (replay_step_ % 10 == 0 || trajectory_done) {
            double t_elapsed = std::chrono::duration<double>(
                Clock::now() - replay_start_time_).count();
            Eigen::Vector3d pos_err = current_state_.head(3) - x_cmd.head(3);
            RCLCPP_INFO(this->get_logger(),
                "[Step %3d/%d | t=%.2fs] "
                "cmd=[%.3f,%.3f,%.3f]  act=[%.3f,%.3f,%.3f]  err=[%.3f,%.3f,%.3f]  |err|=%.3fm",
                step, N, t_elapsed,
                x_cmd(0), x_cmd(1), x_cmd(2),
                current_state_(0), current_state_(1), current_state_(2),
                pos_err(0), pos_err(1), pos_err(2), pos_err.norm());
        }

        if (trajectory_done) {
            if (!done_logged_) {
                RCLCPP_INFO(this->get_logger(),
                    "[OpenLoopTest] Trajectory complete. Holding final position X[%d]. "
                    "Check logs for comparison.", N);
                done_logged_ = true;
                if (comparison_log_.is_open()) comparison_log_.flush();
            }
            // Keep publishing the final state indefinitely so the drone holds
        } else {
            ++replay_step_;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Publishing
    // ─────────────────────────────────────────────────────────────────────────

    void publishCrazyflieCommand(const Eigen::VectorXd& s, const Eigen::Vector3d& acc_ff)
    {
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";
        msg.pose.position.x    = s(0);
        msg.pose.position.y    = s(1);
        msg.pose.position.z    = s(2);
        msg.twist.linear.x     = s(3);
        msg.twist.linear.y     = s(4);
        msg.twist.linear.z     = s(5);
        msg.pose.orientation.w =  s(6);      // qw unchanged
        msg.pose.orientation.x =  s(7);      // qx unchanged
        msg.pose.orientation.y = -s(8);      // qy: FLU→FRD flip
        msg.pose.orientation.z = -s(9);      // qz: FLU→FRD flip
        msg.twist.angular.x    =  s(10);     // wx unchanged
        msg.twist.angular.y    = -s(11);     // wy: FLU→FRD flip
        msg.twist.angular.z    = -s(12);     // wz: FLU→FRD flip
        msg.acc.x = acc_ff(0);
        msg.acc.y = acc_ff(1);
        msg.acc.z = acc_ff(2);
        cf_cmd_pub_->publish(msg);
    }

#ifdef HAS_PX4_MSGS
    void publishPX4Command(const Eigen::VectorXd& s,
                           [[maybe_unused]] const Eigen::Vector3d& acc_ff)
    {
        Eigen::Vector3d pos_ned = frame_conv::enu_to_ned(s.head(3));
        Eigen::Vector3d vel_ned = frame_conv::enu_to_ned(s.segment(3, 3));
        Eigen::Quaterniond q_enu(s(6), s(7), s(8), s(9));
        Eigen::Quaterniond q_ned = frame_conv::quat_enu_to_ned(q_enu);
        double yaw_ned = std::atan2(
            2.0 * (q_ned.w() * q_ned.z() + q_ned.x() * q_ned.y()),
            1.0 - 2.0 * (q_ned.y() * q_ned.y() + q_ned.z() * q_ned.z()));

        px4_msgs::msg::TrajectorySetpoint sp;
        sp.timestamp    = this->now().nanoseconds() / 1000;
        sp.position[0]  = static_cast<float>(pos_ned.x());
        sp.position[1]  = static_cast<float>(pos_ned.y());
        sp.position[2]  = static_cast<float>(pos_ned.z());
        sp.velocity[0]  = static_cast<float>(vel_ned.x());
        sp.velocity[1]  = static_cast<float>(vel_ned.y());
        sp.velocity[2]  = static_cast<float>(vel_ned.z());
        sp.yaw          = static_cast<float>(yaw_ned);
        for (int i = 0; i < 3; ++i) {
            sp.acceleration[i] = std::numeric_limits<float>::quiet_NaN();
            sp.jerk[i]         = std::numeric_limits<float>::quiet_NaN();
        }
        sp.yawspeed = std::numeric_limits<float>::quiet_NaN();
        px4_setpoint_pub_->publish(sp);
    }
#endif

    void publishTrajectoryViz(const std::vector<Eigen::VectorXd>& traj)
    {
        nav_msgs::msg::Path path;
        path.header.stamp    = this->now();
        path.header.frame_id = "world";
        for (const auto& s : traj) {
            geometry_msgs::msg::PoseStamped p;
            p.header.frame_id    = "world";
            p.pose.position.x    = s(0);
            p.pose.position.y    = s(1);
            p.pose.position.z    = s(2);
            p.pose.orientation.w = s(6);
            p.pose.orientation.x = s(7);
            p.pose.orientation.y = s(8);
            p.pose.orientation.z = s(9);
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
        log_folder_ = "./logs/" + drone_name_ + "_open_loop_test_" + ts.str();
        std::filesystem::create_directories(log_folder_);

        // ── 1. planned_trajectory.csv — what the solver produced ─────────────
        planned_log_.open(log_folder_ + "/planned_trajectory.csv");
        if (planned_log_.is_open())
            planned_log_ << "node,t,"
                "cmd_x,cmd_y,cmd_z,cmd_vx,cmd_vy,cmd_vz,"
                "cmd_qw,cmd_qx,cmd_qy,cmd_qz,cmd_wx,cmd_wy,cmd_wz,"
                "fx,fy,fz,mx,my,mz,thrust_norm\n";

        // ── 2. comparison.csv — commanded vs actual at each replay step ───────
        comparison_log_.open(log_folder_ + "/commanded_vs_actual.csv");
        if (comparison_log_.is_open())
            comparison_log_ <<
                "step,t,"
                // commanded (from MPC solution)
                "cmd_x,cmd_y,cmd_z,cmd_vx,cmd_vy,cmd_vz,"
                "cmd_qw,cmd_qx,cmd_qy,cmd_qz,cmd_wx,cmd_wy,cmd_wz,"
                // actual (from drone odom)
                "act_x,act_y,act_z,act_vx,act_vy,act_vz,"
                "act_qw,act_qx,act_qy,act_qz,act_wx,act_wy,act_wz,"
                // derived error
                "err_x,err_y,err_z,pos_err_norm\n";

        RCLCPP_INFO(this->get_logger(),
            "[OpenLoopTest] Logging to: %s", log_folder_.c_str());
    }

    void logPlannedTrajectory()
    {
        if (!planned_log_.is_open()) return;
        for (int i = 0; i < static_cast<int>(ref_X_.size()); ++i) {
            const auto& s = ref_X_[i];
            if (s.size() < 13) continue;
            planned_log_ << std::fixed << std::setprecision(6)
                << i << "," << (i * ocp_dt_);
            for (int j = 0; j < 13; ++j)
                planned_log_ << "," << s(j);
            if (i < static_cast<int>(ref_U_.size()) && ref_U_[i].size() >= 4) {
                const auto& u = ref_U_[i];
                // u = [fz, Mx, My, Mz]
                planned_log_ << ",0,0," << u(0)  // fx=0, fy=0, fz=u(0)
                             << "," << u(1) << "," << u(2) << "," << u(3)  // Mx, My, Mz
                             << "," << u(0);  // thrust_norm = fz
            } else {
                planned_log_ << ",0,0,0,0,0,0,0";
            }
            planned_log_ << "\n";
        }
        planned_log_.flush();
        RCLCPP_INFO(this->get_logger(),
            "[OpenLoopTest] Planned trajectory written (%zu nodes)", ref_X_.size());
    }

    void logComparison(int step, const Eigen::VectorXd& x_cmd)
    {
        double t_rel = step * ocp_dt_;
        Eigen::Vector3d pos_err = current_state_.head(3) - x_cmd.head(3);

        comparison_log_ << std::fixed << std::setprecision(6)
            << step << "," << t_rel;
        // commanded
        for (int j = 0; j < 13; ++j)
            comparison_log_ << "," << x_cmd(j);
        // actual
        for (int j = 0; j < 13; ++j)
            comparison_log_ << "," << current_state_(j);
        // error
        comparison_log_ << "," << pos_err(0) << "," << pos_err(1) << "," << pos_err(2)
                        << "," << pos_err.norm() << "\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Members
    // ─────────────────────────────────────────────────────────────────────────

    // Config
    std::string ocp_type_;
    std::string drone_name_;
    std::string platform_;
    bool        logging_enabled_;
    double      ocp_dt_ = 0.1;

    // MPC (used once only)
    std::unique_ptr<QuadrotorMPC> mpc_;

    // Stored open-loop plan
    std::vector<Eigen::VectorXd> ref_X_;   // X[0..N]
    std::vector<Eigen::VectorXd> ref_U_;   // U[0..N-1]
    double solve_time_ms_ = 0.0;

    // Replay state
    int          replay_step_ = 0;
    bool         done_logged_ = false;
    Clock::time_point replay_start_time_;

    // Drone state
    Eigen::VectorXd current_state_;
    bool pose_received_ = false;
    bool odom_received_ = false;

    // ── Crazyflie ─────────────────────────────────────────────────────────────
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cf_cmd_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr   pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           odom_sub_;

    // ── PX4 ───────────────────────────────────────────────────────────────────
#ifdef HAS_PX4_MSGS
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr     px4_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr    px4_offboard_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr         px4_cmd_pub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr     px4_odom_sub_;
    rclcpp::TimerBase::SharedPtr                                        px4_heartbeat_timer_;
#endif

    // ── Shared ────────────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr replay_timer_;

    // Logging
    std::string   log_folder_;
    std::ofstream planned_log_;
    std::ofstream comparison_log_;
};

// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OpenLoopTestNode>());
    rclcpp::shutdown();
    return 0;
}