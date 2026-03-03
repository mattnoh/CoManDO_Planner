// planner_node.cpp
// ─────────────────────────────────────────────────────────────────────────────
// State layout (13-dimensional, matches the OCP):
//   [0-2]   position        x, y, z          (m)
//   [3-5]   velocity        vx, vy, vz       (m/s)
//   [6-9]   quaternion      qw, qx, qy, qz   (–)
//   [10-12] angular rate    wx, wy, wz        (rad/s)
//
// Sources:
//   /cf_1/pose   PoseStamped   — Motion-capture position + orientation, 100 Hz
//   /cf_1/odom   Odometry      — Kalman velocity + gyro rates,           100 Hz
//
// ─────────────────────────────────────────────────────────────────────────────

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
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
using Clock     = std::chrono::steady_clock;

// ─────────────────────────────────────────────────────────────────────────────
class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        // ── Parameters (ROS2 ordering: declare ALL before getting ANY) ────────
        // Step 1: Declare ocp_type first, get it immediately
        this->declare_parameter("ocp_type", std::string("hover"));
        std::string ocp = this->get_parameter("ocp_type").as_string();

        // Step 2: Now compute default_solver_rate from the known ocp
        const double ocp_dt_temp = OCPRegistry::getDT(ocp);
        const int default_solver_rate = static_cast<int>(std::round(1.0 / ocp_dt_temp));

        // Step 3: Declare remaining params (solver_rate default is now correct)
        this->declare_parameter("drone_name",     std::string("cf_1"));
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("solver_rate",    default_solver_rate);
        this->declare_parameter("hover_target_x", 0.0);
        this->declare_parameter("hover_target_y", 0.0);
        this->declare_parameter("hover_target_z", 1.0);

        // Step 4: Get all parameters
        drone_name_      = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        solver_rate_     = this->get_parameter("solver_rate").as_int();

        double tx = this->get_parameter("hover_target_x").as_double();
        double ty = this->get_parameter("hover_target_y").as_double();
        double tz = this->get_parameter("hover_target_z").as_double();

        // ── MPC ───────────────────────────────────────────────────────────────
        QuadrotorMPC::Config cfg;
        cfg.ocp_type = ocp;

        // Build a valid 13-dim terminal state for the hover OCP.
        // qw=1 (identity quaternion), everything else zero except position.
        // An uninitialized or all-zero terminal_state has qw=0 (invalid
        // quaternion) — HoverOCP receives a degenerate target and produces
        // a zero-effort "stay in place" solution.
        cfg.terminal_state = Eigen::VectorXd::Zero(13);
        cfg.terminal_state(0) = tx;   // px
        cfg.terminal_state(1) = ty;   // py
        cfg.terminal_state(2) = tz;   // pz
        // vel (3-5) = 0
        cfg.terminal_state(6) = 1.0;  // qw = 1  (identity, upright hover)
        // qx qy qz (7-9) = 0
        // angular rates (10-12) = 0

        // ── Shift count ───────────────────────────────────────────────────────
        // PX4 pattern: set solver_rate = 1/ocp_dt so n_shift = 1 always.
        // n_shift is stored in the MPC config — not passed per solve call.
        const double solver_period = 1.0 / static_cast<double>(solver_rate_);
        n_shift_ = std::max(1, static_cast<int>(std::round(solver_period / ocp_dt_temp)));
        cfg.n_shift = n_shift_;

        mpc_.reset(new QuadrotorMPC(cfg));
        ocp_dt_ = mpc_->getOcpDt();

        // ── Alignment check ───────────────────────────────────────────────────
        // PX4 MPC runs at exactly 1/ocp_dt Hz so n_shift=1 always. Warn if we
        // deviate — a mismatch means x[1] is not exactly "one ocp_dt step ahead
        // of the current moment" and accumulated drift will cause the reference
        // to lag or lead the real trajectory.
        const double ideal_rate = 1.0 / ocp_dt_;
        if (std::abs(solver_period - ocp_dt_) > 1e-4) {
            RCLCPP_WARN(this->get_logger(),
                "solver_rate (%d Hz, period=%.4fs) != 1/ocp_dt (%.1f Hz, ocp_dt=%.4fs). "
                "n_shift=%d. Best practice: set solver_rate=%.0f to match ocp_dt.",
                solver_rate_, solver_period, ideal_rate, ocp_dt_, n_shift_, ideal_rate);
        }

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
        //    twist.linear:  vx,vy,vz  in m/s  (Kalman, world frame)
        //    twist.angular: wx,wy,wz  in deg/s (gyro — converted to rad/s below)
        //
        //    UNIT NOTE: crazyflie_server publishes gyro.x/y/z from firmware
        //    which logs in deg/s. Crazyswarm2's _log_odom_data_callback does
        //    NOT convert them before stuffing into twist.angular, so we apply
        //    DEG2RAD here. Verify: print twist.angular.x while spinning —
        //    if values are ~100-500 during fast spin they are still deg/s.
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + drone_name_ + "/odom", 10,
            std::bind(&PlannerNode::odomCallback, this, std::placeholders::_1));

        // ── Solver timer ──────────────────────────────────────────────────────
        // This is the only timer. On every tick it solves the OCP, then
        // immediately publishes x[1] as the new reference for Mellinger.
        //
        // PREVIOUSLY there was also a separate control_timer_ that walked the
        // trajectory index forward between solves using:
        //   idx = 1 + floor(elapsed_since_last_solve / ocp_dt)
        // WHY WE REMOVED IT:
        //   Mellinger is a full-state feedback controller at ~500 Hz on the
        //   firmware — it continuously corrects toward whatever reference we
        //   give it. Walking the index from our side added mutex/thread
        //   complexity with no practical gain.
        solver_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / solver_rate_),
            std::bind(&PlannerNode::solverLoop, this));

        RCLCPP_INFO(this->get_logger(),
            "Planner ready  drone=%s  ocp=%s  solver=%dHz  ocp_dt=%.3fs  n_shift=%d",
            drone_name_.c_str(), ocp.c_str(), solver_rate_, ocp_dt_, n_shift_);
        RCLCPP_INFO(this->get_logger(),
            "Hover target: [%.3f, %.3f, %.3f]", tx, ty, tz);
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
        current_state_(6) = msg->pose.orientation.w;  // qw
        current_state_(7) = msg->pose.orientation.x;  // qx
        current_state_(8) = msg->pose.orientation.y;  // qy
        current_state_(9) = msg->pose.orientation.z;  // qz
        pose_received_ = true;
    }

    // /odom — Kalman velocity + gyro angular rates
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // World-frame velocity from Kalman filter
        current_state_(3) = msg->twist.twist.linear.x;
        current_state_(4) = msg->twist.twist.linear.y;
        current_state_(5) = msg->twist.twist.linear.z;

        // Body-frame angular rates from gyro — firmware logs in deg/s
        constexpr double DEG2RAD = M_PI / 180.0;
        current_state_(10) = msg->twist.twist.angular.x * DEG2RAD;
        current_state_(11) = msg->twist.twist.angular.y * DEG2RAD;
        current_state_(12) = msg->twist.twist.angular.z * DEG2RAD;

        odom_received_ = true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Solver loop — runs at solver_rate_ Hz
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
            RCLCPP_INFO(this->get_logger(),
                "Initial state: pos=[%.3f,%.3f,%.3f] vel=[%.3f,%.3f,%.3f] "
                "q=[%.3f,%.3f,%.3f,%.3f] w=[%.3f,%.3f,%.3f]",
                current_state_(0), current_state_(1), current_state_(2),
                current_state_(3), current_state_(4), current_state_(5),
                current_state_(6), current_state_(7), current_state_(8), current_state_(9),
                current_state_(10), current_state_(11), current_state_(12));
        }

        // n_shift_ is baked into config_.n_shift at construction — no per-call arg needed.
        auto result = mpc_->solve(current_state_);

        if (result.success) {
            const auto& X = result.state_trajectory;
            const auto& U = result.control_trajectory;

            // ── Publish x[1] as the reference setpoint ────────────────────────
            // x[0] = current measured state (initial condition, not commanded).
            // x[1] = first planned future state — Mellinger tracks this until
            //        the next solve fires.
            //
            // Acceleration feedforward:
            //   a_ff = ( x[2][3:5] - x[1][3:5] ) / ocp_dt
            // This is the planned net world-frame acceleration at node 1.
            // NOTE: currently disabled (zero) because the finite-difference
            // gives the net acceleration INCLUDING gravity, which double-counts
            // gravity in Mellinger's thrust computation. Re-enable only after
            // verifying the frame convention with the firmware.
            Eigen::Vector3d acc_ff = Eigen::Vector3d::Zero();
            // if (X.size() > 2)
            //     acc_ff = (X[2].segment(3, 3) - X[1].segment(3, 3)) / ocp_dt_;

            publishCommand(X[1], acc_ff);

            // ── Publish full trajectory for visualisation ─────────────────────
            publishTrajectory(X);

            // ── Logging ───────────────────────────────────────────────────────
            if (logging_enabled_ && logging_initialized_) {
                logActualState();
                const Eigen::VectorXd& u0 = U.empty() ? Eigen::VectorXd::Zero(6) : U[0];
                logCommandedState(X[1], u0);
                if (!first_solve_logged_) {
                    logFirstTrajectory(X, U);
                    first_solve_logged_ = true;
                }
            }

            static int diag_count = 0;
            if (++diag_count % 10 == 0) printDiagnostics(result);

        } else {
            RCLCPP_WARN(this->get_logger(), "MPC solve failed");
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

        msg.pose.position.x = s(0);
        msg.pose.position.y = s(1);
        msg.pose.position.z = s(2);

        msg.twist.linear.x = s(3);
        msg.twist.linear.y = s(4);
        msg.twist.linear.z = s(5);

        // Solver quaternion order: qw(6), qx(7), qy(8), qz(9)
        msg.pose.orientation.w = s(6);
        msg.pose.orientation.x = s(7);
        msg.pose.orientation.y = s(8);
        msg.pose.orientation.z = s(9);

        // Angular velocity — solver output already in rad/s
        msg.twist.angular.x = s(10);
        msg.twist.angular.y = s(11);
        msg.twist.angular.z = s(12);

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

        commanded_state_log_.open(folder  + "/commanded_state.csv");
        actual_state_log_.open(folder     + "/actual_state.csv");
        first_trajectory_log_.open(folder + "/first_solve_trajectory.csv");

        // commanded_state: state + corresponding control at each MPC tick
        if (commanded_state_log_.is_open())
            commanded_state_log_ << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
                                    "fx,fy,fz,mx,my,mz,thrust_norm\n";
        // actual_state: sensor data only — no paired control
        if (actual_state_log_.is_open())
            actual_state_log_    << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";
        // first_solve_trajectory: full X[0..N] and U[0..N-1] from the cold-start solve
        if (first_trajectory_log_.is_open())
            first_trajectory_log_ << "node,t,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
                                     "fx,fy,fz,mx,my,mz,thrust_norm\n";

        RCLCPP_INFO(this->get_logger(), "Logging to: %s", folder.c_str());
    }

    double wallTimeSec()
    {
        return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    }

    void logFirstTrajectory(const std::vector<Eigen::VectorXd>& traj,
                            const std::vector<Eigen::VectorXd>& ctrls)
    {
        if (!first_trajectory_log_.is_open()) return;
        // Each row: node index, time along horizon, full 13-dim state, 6-dim control + thrust_norm.
        // The terminal node (i == N) has no paired control — write zeros for the control columns.
        for (int i = 0; i < (int)traj.size(); ++i) {
            const auto& s = traj[i];
            if (s.size() < 13) continue;
            double t = i * ocp_dt_;
            first_trajectory_log_ << std::fixed << std::setprecision(6)
                << i << "," << t;
            for (int j = 0; j < 13; ++j)
                first_trajectory_log_ << "," << s(j);
            // Control: use U[i] if available, otherwise zeros for terminal node
            if (i < (int)ctrls.size() && ctrls[i].size() >= 6) {
                const auto& u = ctrls[i];
                Eigen::Vector3d f = u.segment(0, 3);
                first_trajectory_log_ << "," << f(0) << "," << f(1) << "," << f(2)
                                      << "," << u(3) << "," << u(4) << "," << u(5)
                                      << "," << f.norm();
            } else {
                first_trajectory_log_ << ",0,0,0,0,0,0,0";
            }
            first_trajectory_log_ << "\n";
        }
        first_trajectory_log_.flush();
        RCLCPP_INFO(this->get_logger(),
            "First solve trajectory saved (%zu nodes, %.2fs horizon)",
            traj.size(), (traj.size() - 1) * ocp_dt_);
    }

    void logActualState()
    {
        if (!actual_state_log_.is_open()) return;
        actual_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) actual_state_log_ << "," << current_state_(i);
        actual_state_log_ << "\n";
        actual_state_log_.flush();
    }

    void logCommandedState(const Eigen::VectorXd& s, const Eigen::VectorXd& u)
    {
        if (!commanded_state_log_.is_open() || s.size() < 13) return;
        commanded_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) commanded_state_log_ << "," << s(i);
        // Append paired control (U[0] — the control applied at this tick)
        if (u.size() >= 6) {
            Eigen::Vector3d f = u.segment(0, 3);
            commanded_state_log_ << "," << f(0) << "," << f(1) << "," << f(2)
                                 << "," << u(3) << "," << u(4) << "," << u(5)
                                 << "," << f.norm();
        } else {
            commanded_state_log_ << ",0,0,0,0,0,0,0";
        }
        commanded_state_log_ << "\n";
        commanded_state_log_.flush();
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
    int                           solver_rate_;
    double                        ocp_dt_ = 0.05;
    int                           n_shift_ = 1;   // steps to shift warm-start per tick

    // Publishers
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  traj_pub_;

    // Subscribers
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          odom_sub_;

    // Timer — single solver timer replaces the old solver+control pair
    rclcpp::TimerBase::SharedPtr solver_timer_;

    // State — assembled from /pose and /odom callbacks
    Eigen::VectorXd current_state_;
    bool pose_received_ = false;
    bool odom_received_ = false;
    bool is_flying_     = false;

    // Logging
    bool logging_initialized_ = false;
    std::ofstream commanded_state_log_;   // state[1..] + paired U[0] per tick
    std::ofstream actual_state_log_;      // sensor state only — no paired control
    std::ofstream first_trajectory_log_;  // full X[0..N] + U[0..N-1] from first solve
    bool          first_solve_logged_ = false;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlannerNode>());
    rclcpp::shutdown();
    return 0;
}