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
#include <Eigen/Dense>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

// ═══════════════════════════════════════════════════════════════════════════════
//  NED ↔ ENU frame conversion helpers (for PX4 support)
// ═══════════════════════════════════════════════════════════════════════════════
namespace frame_conv {

// Position / velocity: NED → ENU
inline Eigen::Vector3d ned_to_enu(const Eigen::Vector3d& v_ned) {
    return {v_ned.y(), v_ned.x(), -v_ned.z()};
}
// Position / velocity: ENU → NED
inline Eigen::Vector3d enu_to_ned(const Eigen::Vector3d& v_enu) {
    return {v_enu.y(), v_enu.x(), -v_enu.z()};
}

// Quaternion: PX4 (FRD body → NED world) → ROS (FLU body → ENU world)
inline Eigen::Quaterniond quat_ned_to_enu(double w, double x, double y, double z) {
    return Eigen::Quaterniond(w, y, x, -z).normalized();
}

// Quaternion: ENU → NED (same rotation, inverse mapping)
inline Eigen::Quaterniond quat_enu_to_ned(const Eigen::Quaterniond& q_enu) {
    return Eigen::Quaterniond(q_enu.w(), q_enu.y(), q_enu.x(), -q_enu.z()).normalized();
}

// Angular velocity: FRD body → FLU body
inline Eigen::Vector3d omega_frd_to_flu(const Eigen::Vector3d& w_frd) {
    return {w_frd.x(), -w_frd.y(), -w_frd.z()};
}

// Angular velocity: FLU body → FRD body
inline Eigen::Vector3d omega_flu_to_frd(const Eigen::Vector3d& w_flu) {
    return {w_flu.x(), -w_flu.y(), -w_flu.z()};
}

} // namespace frame_conv


// ═══════════════════════════════════════════════════════════════════════════════
//  Unified Result type that both solvers produce
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
        this->declare_parameter("hover_target_x", 0.0);
        this->declare_parameter("hover_target_y", 0.0);
        this->declare_parameter("hover_target_z", 1.0);

        std::string ocp = this->get_parameter("ocp_type").as_string();
        drone_name_      = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        platform_        = this->get_parameter("platform").as_string();
        solver_type_     = this->get_parameter("solver").as_string();

        double tx = this->get_parameter("hover_target_x").as_double();
        double ty = this->get_parameter("hover_target_y").as_double();
        double tz = this->get_parameter("hover_target_z").as_double();

        // ── Validate ──────────────────────────────────────────────────────────
        if (platform_ != "crazyflie" && platform_ != "px4") {
            RCLCPP_ERROR(this->get_logger(), "Unknown platform '%s'. Use 'crazyflie' or 'px4'", platform_.c_str());
            throw std::runtime_error("Invalid platform: " + platform_);
        }
        if (solver_type_ != "alipddp" && solver_type_ != "acados") {
            RCLCPP_ERROR(this->get_logger(), "Unknown solver '%s'. Use 'alipddp' or 'acados'", solver_type_.c_str());
            throw std::runtime_error("Invalid solver: " + solver_type_);
        }
#ifndef HAS_ACADOS
        if (solver_type_ == "acados") {
            RCLCPP_ERROR(this->get_logger(), "Acados not available (built without HAS_ACADOS)");
            throw std::runtime_error("Acados not available");
        }
#endif
#ifndef HAS_PX4_MSGS
        if (platform_ == "px4") {
            RCLCPP_ERROR(this->get_logger(), "PX4 not available (built without HAS_PX4_MSGS)");
            throw std::runtime_error("PX4 msgs not available");
        }
#endif

        // ── Terminal state ────────────────────────────────────────────────────
        Eigen::VectorXd terminal = Eigen::VectorXd::Zero(13);
        terminal(0) = tx;
        terminal(1) = ty;
        terminal(2) = tz;
        terminal(6) = 1.0;  // qw = 1

        // ── Solver rate: match 1/ocp_dt for n_shift=1 ────────────────────────
        double ocp_dt_temp = OCPRegistry::getDT(ocp);
        const int default_solver_rate = static_cast<int>(std::round(1.0 / ocp_dt_temp));
        this->declare_parameter("solver_rate", default_solver_rate);
        solver_rate_ = this->get_parameter("solver_rate").as_int();

        const double solver_period = 1.0 / static_cast<double>(solver_rate_);
        n_shift_ = std::max(1, static_cast<int>(std::round(solver_period / ocp_dt_temp)));

        // ── Create solver ─────────────────────────────────────────────────────
        if (solver_type_ == "alipddp") {
            QuadrotorMPC::Config cfg;
            cfg.ocp_type       = ocp;
            cfg.terminal_state = terminal;
            cfg.n_shift        = n_shift_;
            alipddp_mpc_ = std::make_unique<QuadrotorMPC>(cfg);
            ocp_dt_ = alipddp_mpc_->getOcpDt();
        }
#ifdef HAS_ACADOS
        else if (solver_type_ == "acados") {
            AcadosMPC::Config cfg;
            cfg.ocp_type       = ocp;
            cfg.terminal_state = terminal;
            cfg.dt             = ocp_dt_temp;
            cfg.n_shift        = n_shift_;
            acados_mpc_ = std::make_unique<AcadosMPC>(cfg);
            ocp_dt_ = acados_mpc_->getOcpDt();
        }
#endif

        // ── Initial state ─────────────────────────────────────────────────────
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;

        // ── Set up platform-specific pub/sub ──────────────────────────────────
        if (platform_ == "crazyflie") {
            setupCrazyflie();
        } else {
            setupPX4();
        }

        // ── Trajectory visualisation (shared) ─────────────────────────────────
        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);

        // ── Solver timer ──────────────────────────────────────────────────────
        solver_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / solver_rate_),
            std::bind(&PlannerNode::solverLoop, this));

        RCLCPP_INFO(this->get_logger(),
            "Planner ready  platform=%s  solver=%s  ocp=%s  rate=%dHz  ocp_dt=%.3fs  n_shift=%d",
            platform_.c_str(), solver_type_.c_str(), ocp.c_str(),
            solver_rate_, ocp_dt_, n_shift_);
        RCLCPP_INFO(this->get_logger(),
            "Target: [%.3f, %.3f, %.3f]", tx, ty, tz);
        RCLCPP_INFO(this->get_logger(),
            "Waiting for state feedback before starting MPC...");
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    //  Platform setup
    // ─────────────────────────────────────────────────────────────────────────

    void setupCrazyflie()
    {
        cf_cmd_pub_ = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            std::bind(&PlannerNode::cfPoseCallback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + drone_name_ + "/odom", 10,
            std::bind(&PlannerNode::cfOdomCallback, this, std::placeholders::_1));
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

        px4_odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", 10,
            std::bind(&PlannerNode::px4OdomCallback, this, std::placeholders::_1));

        // PX4 requires continuous heartbeat to stay in offboard mode
        px4_heartbeat_timer_ = this->create_wall_timer(
            100ms, std::bind(&PlannerNode::px4HeartbeatCallback, this));

        RCLCPP_INFO(this->get_logger(),
            "PX4 interface: /fmu/out/vehicle_odometry -> /fmu/in/trajectory_setpoint");
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Crazyflie callbacks
    // ─────────────────────────────────────────────────────────────────────────

    void cfPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        // Motion-capture position + quaternion — always authoritative
        current_state_(0) = msg->pose.position.x;
        current_state_(1) = msg->pose.position.y;
        current_state_(2) = msg->pose.position.z;
        current_state_(6) = msg->pose.orientation.w;  // qw
        current_state_(7) = msg->pose.orientation.x;  // qx
        current_state_(8) = msg->pose.orientation.y;  // qy
        current_state_(9) = msg->pose.orientation.z;  // qz
        pose_received_ = true;
    }

    void cfOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // World-frame velocity from Kalman filter (m/s)
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
    //  PX4 callbacks
    // ─────────────────────────────────────────────────────────────────────────

#ifdef HAS_PX4_MSGS
    void px4OdomCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
    {
        // PX4 VehicleOdometry is in NED frame — convert to ENU for solver
        Eigen::Vector3d pos_ned(msg->position[0], msg->position[1], msg->position[2]);
        Eigen::Vector3d vel_ned(msg->velocity[0], msg->velocity[1], msg->velocity[2]);
        Eigen::Vector3d omega_frd(msg->angular_velocity[0], msg->angular_velocity[1], msg->angular_velocity[2]);

        Eigen::Vector3d pos_enu   = frame_conv::ned_to_enu(pos_ned);
        Eigen::Vector3d vel_enu   = frame_conv::ned_to_enu(vel_ned);
        Eigen::Quaterniond q_enu  = frame_conv::quat_ned_to_enu(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
        Eigen::Vector3d omega_flu = frame_conv::omega_frd_to_flu(omega_frd);

        current_state_(0)  = pos_enu.x();
        current_state_(1)  = pos_enu.y();
        current_state_(2)  = pos_enu.z();
        current_state_(3)  = vel_enu.x();
        current_state_(4)  = vel_enu.y();
        current_state_(5)  = vel_enu.z();
        current_state_(6)  = q_enu.w();
        current_state_(7)  = q_enu.x();
        current_state_(8)  = q_enu.y();
        current_state_(9)  = q_enu.z();
        current_state_(10) = omega_flu.x();
        current_state_(11) = omega_flu.y();
        current_state_(12) = omega_flu.z();

        pose_received_ = true;
        odom_received_ = true;  // PX4 odom has everything in one message
    }

    void px4HeartbeatCallback()
    {
        px4_msgs::msg::OffboardControlMode msg;
        msg.timestamp     = this->now().nanoseconds() / 1000;
        msg.position      = true;
        msg.velocity      = true;
        msg.acceleration  = false;
        msg.attitude      = false;
        msg.body_rate     = false;
        px4_offboard_pub_->publish(msg);
    }

    void px4Arm()
    {
        px4_msgs::msg::VehicleCommand msg;
        msg.timestamp        = this->now().nanoseconds() / 1000;
        msg.command          = 400;     // VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1           = 1.0;     // 1 = arm
        msg.target_system    = 1;
        msg.target_component = 1;
        msg.source_system    = 1;
        msg.source_component = 1;
        msg.from_external    = true;
        px4_cmd_pub_->publish(msg);
    }

    void px4SetOffboardMode()
    {
        px4_msgs::msg::VehicleCommand msg;
        msg.timestamp        = this->now().nanoseconds() / 1000;
        msg.command          = 176;     // VEHICLE_CMD_DO_SET_MODE
        msg.param1           = 1.0;     // base mode
        msg.param2           = 6.0;     // PX4_CUSTOM_MAIN_MODE_OFFBOARD
        msg.target_system    = 1;
        msg.target_component = 1;
        msg.source_system    = 1;
        msg.source_component = 1;
        msg.from_external    = true;
        px4_cmd_pub_->publish(msg);
    }
#endif

    // ─────────────────────────────────────────────────────────────────────────
    //  Solver loop
    // ─────────────────────────────────────────────────────────────────────────

    void solverLoop()
    {
        if (!pose_received_ || !odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for state: /pose [%s] /odom [%s]...",
                pose_received_ ? "OK" : "MISSING",
                odom_received_ ? "OK" : "MISSING");
            return;
        }

        if (!is_flying_) {
            is_flying_ = true;
            RCLCPP_INFO(this->get_logger(), "State feedback received — starting MPC (%s)", solver_type_.c_str());

#ifdef HAS_PX4_MSGS
            if (platform_ == "px4") {
                px4SetOffboardMode();
                px4Arm();
                RCLCPP_INFO(this->get_logger(), "PX4: Offboard mode + Arm commands sent");
            }
#endif

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

        // ── Solve with selected solver ────────────────────────────────────────
        SolverResult result;

        if (solver_type_ == "alipddp" && alipddp_mpc_) {
            auto r = alipddp_mpc_->solve(current_state_);
            result.success            = r.success;
            result.next_state         = r.next_state;
            result.state_trajectory   = r.state_trajectory;
            result.control_trajectory = r.control_trajectory;
            result.solve_time_ms      = r.solve_time_ms;
            result.solve_timestamp    = r.solve_timestamp;
        }
#ifdef HAS_ACADOS
        else if (solver_type_ == "acados" && acados_mpc_) {
            auto r = acados_mpc_->solve(current_state_);
            result.success            = r.success;
            result.next_state         = r.next_state;
            result.state_trajectory   = r.state_trajectory;
            result.control_trajectory = r.control_trajectory;
            result.solve_time_ms      = r.solve_time_ms;
            result.solve_timestamp    = r.solve_timestamp;
        }
#endif

        if (result.success) {
            const auto& X = result.state_trajectory;
            const auto& U = result.control_trajectory;

            Eigen::Vector3d acc_ff = Eigen::Vector3d::Zero();

            // ── Platform-specific command publishing ──────────────────────────
            if (platform_ == "crazyflie") {
                publishCrazyflieCommand(X[1], acc_ff);
            } else {
                publishPX4Command(X[1], acc_ff);
            }

            publishTrajectory(X);

            // ── Logging ──────────────────────────────────────────────────────
            if (logging_enabled_ && logging_initialized_) {
                logActualState();
                const Eigen::VectorXd& u0 = U.empty() ? Eigen::VectorXd::Zero(4) : U[0];
                logCommandedState(X[1], u0);
                if (!first_solve_logged_) {
                    logFirstTrajectory(X, U);
                    first_solve_logged_ = true;
                }
            }

            static int diag_count = 0;
            if (++diag_count % 10 == 0) printDiagnostics(result);

        } else {
            RCLCPP_WARN(this->get_logger(), "MPC solve failed (%s)", solver_type_.c_str());
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Publishing — Crazyflie
    // ─────────────────────────────────────────────────────────────────────────

    void publishCrazyflieCommand(const Eigen::VectorXd& s, const Eigen::Vector3d& acc_cmd)
    {
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";

        msg.pose.position.x    = s(0);
        msg.pose.position.y    = s(1);
        msg.pose.position.z    = s(2);
        msg.twist.linear.x    = s(3);
        msg.twist.linear.y    = s(4);
        msg.twist.linear.z    = s(5);
        msg.pose.orientation.w = s(6);
        msg.pose.orientation.x = s(7);
        msg.pose.orientation.y = s(8);
        msg.pose.orientation.z = s(9);
        msg.twist.angular.x   = s(10);
        msg.twist.angular.y   = s(11);
        msg.twist.angular.z   = s(12);
        msg.acc.x = acc_cmd(0);
        msg.acc.y = acc_cmd(1);
        msg.acc.z = acc_cmd(2);

        cf_cmd_pub_->publish(msg);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Publishing — PX4
    // ─────────────────────────────────────────────────────────────────────────

    void publishPX4Command(const Eigen::VectorXd& s,
                           [[maybe_unused]] const Eigen::Vector3d& acc_cmd)
    {
#ifdef HAS_PX4_MSGS
        // Convert solver ENU state to PX4 NED frame
        Eigen::Vector3d pos_enu(s(0), s(1), s(2));
        Eigen::Vector3d vel_enu(s(3), s(4), s(5));

        Eigen::Vector3d pos_ned = frame_conv::enu_to_ned(pos_enu);
        Eigen::Vector3d vel_ned = frame_conv::enu_to_ned(vel_enu);

        // Yaw from quaternion (ENU → NED)
        Eigen::Quaterniond q_enu(s(6), s(7), s(8), s(9));
        Eigen::Quaterniond q_ned = frame_conv::quat_enu_to_ned(q_enu);

        // Extract yaw in NED frame
        double yaw_ned = std::atan2(
            2.0 * (q_ned.w() * q_ned.z() + q_ned.x() * q_ned.y()),
            1.0 - 2.0 * (q_ned.y() * q_ned.y() + q_ned.z() * q_ned.z()));

        px4_msgs::msg::TrajectorySetpoint sp;
        sp.timestamp     = this->now().nanoseconds() / 1000;
        sp.position[0]   = static_cast<float>(pos_ned.x());
        sp.position[1]   = static_cast<float>(pos_ned.y());
        sp.position[2]   = static_cast<float>(pos_ned.z());
        sp.velocity[0]   = static_cast<float>(vel_ned.x());
        sp.velocity[1]   = static_cast<float>(vel_ned.y());
        sp.velocity[2]   = static_cast<float>(vel_ned.z());
        sp.acceleration[0] = std::numeric_limits<float>::quiet_NaN();
        sp.acceleration[1] = std::numeric_limits<float>::quiet_NaN();
        sp.acceleration[2] = std::numeric_limits<float>::quiet_NaN();
        sp.jerk[0]       = std::numeric_limits<float>::quiet_NaN();
        sp.jerk[1]       = std::numeric_limits<float>::quiet_NaN();
        sp.jerk[2]       = std::numeric_limits<float>::quiet_NaN();
        sp.yaw           = static_cast<float>(yaw_ned);
        sp.yawspeed      = std::numeric_limits<float>::quiet_NaN();

        px4_setpoint_pub_->publish(sp);
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Trajectory visualisation (shared)
    // ─────────────────────────────────────────────────────────────────────────

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
    //  Logging
    // ─────────────────────────────────────────────────────────────────────────

    void setupLogging()
    {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::stringstream ts;
        ts << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        std::string folder = "./logs/" + drone_name_ + "_" + solver_type_ + "_flight_" + ts.str();
        std::filesystem::create_directories(folder);

        commanded_state_log_.open(folder  + "/commanded_state.csv");
        actual_state_log_.open(folder     + "/actual_state.csv");
        first_trajectory_log_.open(folder + "/first_solve_trajectory.csv");

        if (commanded_state_log_.is_open())
            commanded_state_log_ << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,"
                                    "fx,fy,fz,mx,my,mz,thrust_norm\n";
        if (actual_state_log_.is_open())
            actual_state_log_    << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";
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
        for (int i = 0; i < (int)traj.size(); ++i) {
            const auto& s = traj[i];
            if (s.size() < 13) continue;
            double t = i * ocp_dt_;
            first_trajectory_log_ << std::fixed << std::setprecision(6)
                << i << "," << t;
            for (int j = 0; j < 13; ++j)
                first_trajectory_log_ << "," << s(j);
            if (i < (int)ctrls.size() && ctrls[i].size() >= 4) {
                const auto& u = ctrls[i];
                // u = [fz, Mx, My, Mz]
                first_trajectory_log_ << ",0,0," << u(0)  // fx=0, fy=0, fz=u(0)
                                      << "," << u(1) << "," << u(2) << "," << u(3)  // Mx, My, Mz
                                      << "," << u(0);  // thrust_norm = fz
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
        if (u.size() >= 4) {
            // u = [fz, Mx, My, Mz]
            commanded_state_log_ << ",0,0," << u(0)  // fx=0, fy=0, fz=u(0)
                                 << "," << u(1) << "," << u(2) << "," << u(3)  // Mx, My, Mz
                                 << "," << u(0);  // thrust_norm = fz
        } else {
            commanded_state_log_ << ",0,0,0,0,0,0,0";
        }
        commanded_state_log_ << "\n";
        commanded_state_log_.flush();
    }

    void printDiagnostics(const SolverResult& result)
    {
        Eigen::Vector3d pos = current_state_.segment(0, 3);
        Eigen::Vector3d vel = current_state_.segment(3, 3);
        Eigen::Vector3d ww  = current_state_.segment(10, 3);
        double thrust = result.control_trajectory.empty() ? 0.0
            : result.control_trajectory[0](0);  // u(0) = fz only
        RCLCPP_INFO(this->get_logger(),
            "[%s] %.1fms | pos[%.3f,%.3f,%.3f] vel[%.3f,%.3f,%.3f] "
            "w[%.3f,%.3f,%.3f]rad/s | T=%.3fN",
            solver_type_.c_str(),
            result.solve_time_ms,
            pos.x(), pos.y(), pos.z(),
            vel.x(), vel.y(), vel.z(),
            ww.x(), ww.y(), ww.z(),
            thrust);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Members
    // ─────────────────────────────────────────────────────────────────────────

    // Config
    std::string platform_;
    std::string solver_type_;
    std::string drone_name_;
    bool        logging_enabled_;
    int         solver_rate_;
    double      ocp_dt_  = 0.1;
    int         n_shift_ = 1;

    // Solvers (only one is active)
    std::unique_ptr<QuadrotorMPC> alipddp_mpc_;
#ifdef HAS_ACADOS
    std::unique_ptr<AcadosMPC>    acados_mpc_;
#endif

    // ── Crazyflie publishers / subscribers ────────────────────────────────────
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cf_cmd_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr   pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           odom_sub_;

    // ── PX4 publishers / subscribers ──────────────────────────────────────────
#ifdef HAS_PX4_MSGS
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr     px4_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr    px4_offboard_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr         px4_cmd_pub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr    px4_odom_sub_;
    rclcpp::TimerBase::SharedPtr                                        px4_heartbeat_timer_;
#endif

    // ── Shared ────────────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
    rclcpp::TimerBase::SharedPtr                      solver_timer_;

    // State
    Eigen::VectorXd current_state_;
    Eigen::Vector3d filtered_omega_ = Eigen::Vector3d::Zero();
    static constexpr double OMEGA_FILTER_ALPHA = 0.3;
    bool pose_received_ = false;
    bool odom_received_ = false;
    bool is_flying_     = false;

    // Logging
    bool logging_initialized_ = false;
    std::ofstream commanded_state_log_;
    std::ofstream actual_state_log_;
    std::ofstream first_trajectory_log_;
    bool          first_solve_logged_ = false;
};

// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlannerNode>());
    rclcpp::shutdown();
    return 0;
}
