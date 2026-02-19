// planner_node.cpp
// ─────────────────────────────────────────────────────────────────────────────
// State layout (13-dimensional, matches your OCP):
//   [0-2]   position        x, y, z          (m)      — from /pose (MoCap)
//   [3-5]   velocity        vx, vy, vz       (m/s)    — from /odom (Kalman)
//   [6-9]   quaternion      qw, qx, qy, qz   (–)      — from /pose (MoCap)
//   [10-12] angular rate    wx, wy, wz        (rad/s)  — from /odom (gyro, deg→rad)
//
// Sources:
//   /cf_1/pose   PoseStamped   — Motion-capture position + orientation, 100 Hz
//   /cf_1/odom   Odometry      — Kalman velocity + gyro rates,           100 Hz
//
// Why two topics instead of odom alone?
//   Motion-capture quaternion is more accurate than the Kalman/stabilizer
//   euler-reconstructed quaternion that odom carries. We mix the best of both.
// ─────────────────────────────────────────────────────────────────────────────

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
#include "quadrotor_mpc.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>

using namespace std::chrono_literals;
using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;

// ─────────────────────────────────────────────────────────────────────────────
class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        // ── Parameters ────────────────────────────────────────────────────────
        this->declare_parameter("drone_name",     "cf_1");
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("ocp_type",       "hover");
        this->declare_parameter("control_rate",   10);
        this->declare_parameter("solver_rate",    10);

        drone_name_      = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        std::string ocp  = this->get_parameter("ocp_type").as_string();
        control_rate_    = this->get_parameter("control_rate").as_int();
        solver_rate_     = this->get_parameter("solver_rate").as_int();

        // ── MPC ───────────────────────────────────────────────────────────────
        QuadrotorMPC::Config cfg;
        cfg.ocp_type = ocp;
        mpc_.reset(new QuadrotorMPC(cfg));
        ocp_dt_ = mpc_->getOcpDt();

        // ── Initial state — identity quaternion, everything else zero ─────────
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;   // qw = 1

        // ── Publishers ────────────────────────────────────────────────────────
        cmd_pub_  = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);
        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);

        // ── Subscribers ───────────────────────────────────────────────────────
        // 1) Motion-capture pose — position + quaternion at 100 Hz
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            std::bind(&PlannerNode::poseCallback, this, std::placeholders::_1));

        // 2) Kalman odom — world-frame velocity + body gyro rates at 100 Hz
        //    Requires `odom` in firmware_logging.default_topics (cflib backend).
        //    twist.linear:  vx,vy,vz  in m/s  (Kalman, world frame)
        //    twist.angular: wx,wy,wz  in rad/s (gyro — see unit note below)
        //
        //    UNIT NOTE: crazyflie_server publishes gyro.x/y/z from firmware
        //    which logs in deg/s.  Crazyswarm2's _log_odom_data_callback does
        //    NOT convert them before stuffing into twist.angular, so we apply
        //    DEG2RAD here.  Verify once: print twist.angular.x while spinning
        //    — if values are ~100-500 during fast spin they are still deg/s.
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + drone_name_ + "/odom", 10,
            std::bind(&PlannerNode::odomCallback, this, std::placeholders::_1));

        // ── Timers ────────────────────────────────────────────────────────────
        // Solver: re-solves the OCP and updates the stored trajectory.
        solver_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / solver_rate_),
            std::bind(&PlannerNode::solverLoop, this));

        // Control: publishes the current reference setpoint to Mellinger.
        // Can run faster than the solver — it just indexes into the latest
        // stored trajectory.
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / control_rate_),
            std::bind(&PlannerNode::controlLoop, this));

        RCLCPP_INFO(this->get_logger(),
            "Planner ready  drone=%s  ocp=%s  ctrl=%dHz  solver=%dHz  ocp_dt=%.3fs",
            drone_name_.c_str(), ocp.c_str(), control_rate_, solver_rate_, ocp_dt_);
        RCLCPP_INFO(this->get_logger(),
            "Waiting for /pose AND /odom before starting MPC...");
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    // Callbacks
    // ─────────────────────────────────────────────────────────────────────────

    // /pose — Motion-capture position + quaternion
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        current_state_(0) = msg->pose.position.x;
        current_state_(1) = msg->pose.position.y;
        current_state_(2) = msg->pose.position.z;
        // Quaternion: MoCap gives the most accurate orientation
        current_state_(6) = msg->pose.orientation.w;  // qw
        current_state_(7) = msg->pose.orientation.x;  // qx
        current_state_(8) = msg->pose.orientation.y;  // qy
        current_state_(9) = msg->pose.orientation.z;  // qz
        pose_received_ = true;
    }

    // /odom — Kalman velocity + gyro angular rates
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // World-frame velocity from Kalman filter (kalman.statePX/Y/Z)
        current_state_(3) = msg->twist.twist.linear.x;
        current_state_(4) = msg->twist.twist.linear.y;
        current_state_(5) = msg->twist.twist.linear.z;

        // Body-frame angular rates from gyro (gyro.x/y/z)
        // Crazyswarm2 publishes these in deg/s — convert to rad/s for the OCP.
        // If you have confirmed they are already rad/s, remove DEG2RAD.
        constexpr double DEG2RAD = M_PI / 180.0;
        current_state_(10) = msg->twist.twist.angular.x * DEG2RAD;
        current_state_(11) = msg->twist.twist.angular.y * DEG2RAD;
        current_state_(12) = msg->twist.twist.angular.z * DEG2RAD;

        odom_received_ = true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Solver loop — runs at solver_rate_ Hz
    // Solves the OCP from the current state and stores the new trajectory.
    // ─────────────────────────────────────────────────────────────────────────
    void solverLoop()
    {
        if (!pose_received_ || !odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for /pose [%s] and /odom [%s]...",
                pose_received_ ? "OK" : "MISSING",
                odom_received_ ? "OK" : "MISSING");
            return;
        }

        if (!is_flying_) {
            is_flying_ = true;
            RCLCPP_INFO(this->get_logger(), "Both topics received — starting MPC");
            if (logging_enabled_ && !logging_initialized_) {
                setupLogging();
                logging_initialized_ = true;
            }
            // Log initial state for debugging
            RCLCPP_INFO(this->get_logger(),
                "Initial state: pos=[%.3f,%.3f,%.3f] vel=[%.3f,%.3f,%.3f] "
                "q=[%.3f,%.3f,%.3f,%.3f] w=[%.3f,%.3f,%.3f]",
                current_state_(0), current_state_(1), current_state_(2),
                current_state_(3), current_state_(4), current_state_(5),
                current_state_(6), current_state_(7), current_state_(8), current_state_(9),
                current_state_(10), current_state_(11), current_state_(12));
        }

        auto result = mpc_->solve(current_state_);

        if (result.success) {
            {
                std::lock_guard<std::mutex> lock(traj_mutex_);
                state_traj_   = result.state_trajectory;
                control_traj_ = result.control_trajectory;
                traj_valid_   = true;
            }
            publishTrajectory(result.state_trajectory);

            static int diag_count = 0;
            if (++diag_count % 10 == 0) printDiagnostics(result);
        } else {
            RCLCPP_WARN(this->get_logger(), "MPC solve failed");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Control loop — runs at control_rate_ Hz
    //
    // Always commands state_traj_[1]: the first predicted future step.
    //
    // Why state[1] and not state[0]?
    //   state[0] is the current state (where we already are — no feedforward).
    //   state[1] is one OCP timestep ahead — exactly what Mellinger should be
    //   driving toward right now.  By the time the message arrives at Mellinger
    //   (~1-5ms) we are already a few ms into that step, so this is correct.
    //
    // If control_rate_ > solver_rate_ you can walk idx forward between solves,
    // but for equal rates always commanding state[1] is the simplest, correct
    // approach with no timing artefacts.
    // ─────────────────────────────────────────────────────────────────────────
    void controlLoop()
    {
        if (!is_flying_) return;

        std::lock_guard<std::mutex> lock(traj_mutex_);
        if (!traj_valid_ || state_traj_.size() < 2) return;

        // ── Always command the first future predicted step ────────────────────
        const Eigen::VectorXd& cmd = state_traj_[1];

        // ── Acceleration feedforward from velocity finite difference ──────────
        // a ≈ (v[1] - v[0]) / dt
        // This is the expected acceleration along the planned trajectory and
        // gives Mellinger's acceleration feedforward term meaningful content.
        Eigen::Vector3d acc_cmd =
            (state_traj_[1].segment(3, 3) - state_traj_[0].segment(3, 3)) / ocp_dt_;

        publishCommand(cmd, acc_cmd);

        if (logging_enabled_ && logging_initialized_) {
            logActualState();
            logCommandedState(cmd);
            if (!control_traj_.empty()) {
                logControl(control_traj_[0]);  // control[0] drives state[0]→state[1]
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Publishing
    // ─────────────────────────────────────────────────────────────────────────
    void publishCommand(const Eigen::VectorXd& s, const Eigen::Vector3d& acc_cmd)
    {
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";

        // Position
        msg.pose.position.x = s(0);
        msg.pose.position.y = s(1);
        msg.pose.position.z = s(2);

        // Linear velocity
        msg.twist.linear.x = s(3);
        msg.twist.linear.y = s(4);
        msg.twist.linear.z = s(5);

        // Orientation — solver order: qw(6), qx(7), qy(8), qz(9)
        msg.pose.orientation.w = s(6);
        msg.pose.orientation.x = s(7);
        msg.pose.orientation.y = s(8);
        msg.pose.orientation.z = s(9);

        // Angular velocity — solver output is already in rad/s
        msg.twist.angular.x = s(10);
        msg.twist.angular.y = s(11);
        msg.twist.angular.z = s(12);

        // Acceleration feedforward
        msg.acc.x = acc_cmd(0);
        msg.acc.y = acc_cmd(1);
        msg.acc.z = acc_cmd(2);

        cmd_pub_->publish(msg);
    }

    void publishTrajectory(const std::vector<Eigen::VectorXd>& traj)
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
    // Logging
    // ─────────────────────────────────────────────────────────────────────────
    void setupLogging()
    {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::stringstream ts;
        ts << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        std::string folder = "./logs/" + drone_name_ + "_flight_" + ts.str();
        std::filesystem::create_directories(folder);

        commanded_state_log_.open(folder + "/commanded_state.csv");
        actual_state_log_.open(folder    + "/actual_state.csv");
        control_log_.open(folder         + "/control.csv");

        if (commanded_state_log_.is_open())
            commanded_state_log_ << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";
        if (actual_state_log_.is_open())
            actual_state_log_    << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";
        if (control_log_.is_open())
            control_log_         << "timestamp,fx,fy,fz,mx,my,mz,thrust_norm\n";

        RCLCPP_INFO(this->get_logger(), "Logging to: %s", folder.c_str());
    }

    double wallTimeSec()
    {
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    void logActualState()
    {
        if (!actual_state_log_.is_open()) return;
        actual_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) actual_state_log_ << "," << current_state_(i);
        actual_state_log_ << "\n";
        actual_state_log_.flush();
    }

    void logCommandedState(const Eigen::VectorXd& s)
    {
        if (!commanded_state_log_.is_open() || s.size() < 13) return;
        commanded_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) commanded_state_log_ << "," << s(i);
        commanded_state_log_ << "\n";
        commanded_state_log_.flush();
    }

    void logControl(const Eigen::VectorXd& u)
    {
        if (!control_log_.is_open() || u.size() < 6) return;
        Eigen::Vector3d f = u.segment(0, 3);
        Eigen::Vector3d m = u.segment(3, 3);
        control_log_ << std::fixed << std::setprecision(6) << wallTimeSec()
                     << "," << f(0) << "," << f(1) << "," << f(2)
                     << "," << m(0) << "," << m(1) << "," << m(2)
                     << "," << f.norm() << "\n";
        control_log_.flush();
    }

    void printDiagnostics(const QuadrotorMPC::Result& result)
    {
        Eigen::Vector3d pos = current_state_.segment(0, 3);
        Eigen::Vector3d vel = current_state_.segment(3, 3);
        Eigen::Vector3d ww  = current_state_.segment(10, 3);
        double thrust = result.control_trajectory.empty() ? 0.0
            : result.control_trajectory[0].segment(0, 3).norm();
        RCLCPP_INFO(this->get_logger(),
            "MPC %.1fms | pos[%.3f,%.3f,%.3f] vel[%.3f,%.3f,%.3f] "
            "w[%.3f,%.3f,%.3f]rad/s | T=%.3fN",
            result.solve_time_ms,
            pos.x(), pos.y(), pos.z(),
            vel.x(), vel.y(), vel.z(),
            ww.x(), ww.y(), ww.z(),
            thrust);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Members
    // ─────────────────────────────────────────────────────────────────────────
    std::unique_ptr<QuadrotorMPC> mpc_;
    std::string                   drone_name_;
    bool                          logging_enabled_;
    int                           control_rate_;
    int                           solver_rate_;
    double                        ocp_dt_ = 0.05;

    // Publishers
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  traj_pub_;

    // Subscribers — only pose (MoCap) and odom (Kalman vel + gyro)
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          odom_sub_;

    // Timers
    rclcpp::TimerBase::SharedPtr solver_timer_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    // State
    Eigen::VectorXd current_state_;   // 13-dim: pos, vel, quat, omega
    bool pose_received_ = false;
    bool odom_received_ = false;
    bool is_flying_     = false;

    // Logging
    bool logging_initialized_ = false;
    std::ofstream commanded_state_log_;
    std::ofstream actual_state_log_;
    std::ofstream control_log_;

    // Trajectory (written by solver, read by control loop)
    std::mutex                   traj_mutex_;
    std::vector<Eigen::VectorXd> state_traj_;
    std::vector<Eigen::VectorXd> control_traj_;
    bool                         traj_valid_ = false;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlannerNode>());
    rclcpp::shutdown();
    return 0;
}