#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
#include <crazyflie_interfaces/msg/log_data_generic.hpp>
#include "quadrotor_mpc.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

using namespace std::chrono_literals;
using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;

// ─────────────────────────────────────────────────────────────────────────────
class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        this->declare_parameter("drone_name",     "cf_1");
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("ocp_type",       "hover");
        this->declare_parameter("control_rate",   50);
        this->declare_parameter("solver_rate",    10);

        drone_name_      = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        std::string ocp  = this->get_parameter("ocp_type").as_string();
        control_rate_    = this->get_parameter("control_rate").as_int();
        solver_rate_     = this->get_parameter("solver_rate").as_int();

        QuadrotorMPC::Config cfg;
        cfg.ocp_type = ocp;
        mpc_.reset(new QuadrotorMPC(cfg));
        ocp_dt_ = mpc_->getOcpDt();

        if (logging_enabled_) setupLogging();

        cmd_pub_  = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);
        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            std::bind(&PlannerNode::poseCallback, this, std::placeholders::_1));

        vel_sub_ = this->create_subscription<crazyflie_interfaces::msg::LogDataGeneric>(
            "/" + drone_name_ + "/velocity", 10,
            std::bind(&PlannerNode::velocityCallback, this, std::placeholders::_1));

        // Angular velocity topic — when this arrives the actual state will
        // have real omega values; until then the fields stay at the zero
        // set once in the constructor below.
        ang_vel_sub_ = this->create_subscription<crazyflie_interfaces::msg::LogDataGeneric>(
            "/" + drone_name_ + "/angular_velocity", 10,
            std::bind(&PlannerNode::angularVelocityCallback, this, std::placeholders::_1));

        // Initial state — identity quaternion, everything else zero.
        // Angular velocity fields (10-12) start at zero here and are updated
        // only by angularVelocityCallback.  They are NEVER overwritten in the
        // solver loop — doing so would corrupt the actual state log.
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;

        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / control_rate_),
            std::bind(&PlannerNode::controlLoop, this));
        solver_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / solver_rate_),
            std::bind(&PlannerNode::solverLoop, this));

        RCLCPP_INFO(this->get_logger(),
            "Planner ready  drone=%s  ocp=%s  ctrl=%dHz  solver=%dHz  ocp_dt=%.3fs",
            drone_name_.c_str(), ocp.c_str(), control_rate_, solver_rate_, ocp_dt_);
    }

private:
    // ── Logging ───────────────────────────────────────────────────────────────
    void setupLogging() {
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

    // ── Subscribers ───────────────────────────────────────────────────────────
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        current_state_(0) = msg->pose.position.x;
        current_state_(1) = msg->pose.position.y;
        current_state_(2) = msg->pose.position.z;
        current_state_(6) = msg->pose.orientation.w;
        current_state_(7) = msg->pose.orientation.x;
        current_state_(8) = msg->pose.orientation.y;
        current_state_(9) = msg->pose.orientation.z;
        pose_received_    = true;
    }

    void velocityCallback(const crazyflie_interfaces::msg::LogDataGeneric::SharedPtr msg) {
        if (msg->values.size() >= 3) {
            current_state_(3) = msg->values[0];
            current_state_(4) = msg->values[1];
            current_state_(5) = msg->values[2];
            vel_received_     = true;
        }
    }

    // Receives body-frame angular velocity from the Crazyflie log system.
    // The three values map directly to omega_x, omega_y, omega_z (rad/s).
    // Once this callback fires the actual-state log will contain real data.
    void angularVelocityCallback(
        const crazyflie_interfaces::msg::LogDataGeneric::SharedPtr msg)
    {
        if (msg->values.size() >= 3) {
            current_state_(10) = msg->values[0];
            current_state_(11) = msg->values[1];
            current_state_(12) = msg->values[2];
        }
    }

    // ── Solver loop ───────────────────────────────────────────────────────────
    void solverLoop() {
        if (!pose_received_ || !vel_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Waiting for pose and velocity...");
            return;
        }
        if (!is_flying_) {
            is_flying_ = true;
            RCLCPP_INFO(this->get_logger(), "State received — starting MPC");
        }

        // NOTE: angular velocity fields (10-12) are NOT zeroed here.
        // They hold whatever angularVelocityCallback last wrote (or the
        // constructor zero if that subscriber hasn't fired yet).
        // Zeroing them here would corrupt both the MPC input and the log.

        auto result = mpc_->solve(current_state_);

        if (result.success) {
            {
                std::lock_guard<std::mutex> lock(traj_mutex_);
                state_traj_   = result.state_trajectory;
                control_traj_ = result.control_trajectory;
                solve_time_   = result.solve_timestamp;
                traj_valid_   = true;
            }
            publishTrajectory(result.state_trajectory);

            static int count = 0;
            if (++count % 10 == 0) printDiagnostics(result);
        } else {
            RCLCPP_WARN(this->get_logger(), "MPC solve failed");
        }
    }

    // ── Control loop ──────────────────────────────────────────────────────────
    void controlLoop() {
        if (!is_flying_) return;

        std::lock_guard<std::mutex> lock(traj_mutex_);
        if (!traj_valid_ || state_traj_.empty()) return;

        // ── Wall-clock trajectory indexing ───────────────────────────────────
        // elapsed is measured from solve_time_.  We add ONE ocp_dt_ before
        // dividing so that idx starts at 1 (the first predicted future state)
        // rather than 0 (the current/past state that was just measured).
        //
        // Why this matters:
        //   idx=0 is X[0] = current_state_ at solve time — a position the
        //   drone has already been at (or is just leaving).  Sending it as a
        //   command causes a sawtooth jump every time the solver refreshes,
        //   because the new X[0] is the new measured position, not a
        //   continuation of the previous trajectory.
        //   Starting at idx=1 means we always command the *next* predicted
        //   state, which is a smooth continuation across solver updates.
        double elapsed_s = std::chrono::duration<double>(
            Clock::now() - solve_time_).count();

        // +1 offset: start playback from the first future step, not the
        // current measurement.
        int idx = static_cast<int>(elapsed_s / ocp_dt_) + 1;

        int max_idx = static_cast<int>(state_traj_.size()) - 1;
        idx = std::max(1, std::min(idx, max_idx));

        Eigen::VectorXd cmd = state_traj_[idx];
        publishCommand(cmd);

        if (logging_enabled_) {
            logActualState();
            logCommandedState(cmd);
            if (!control_traj_.empty()) {
                int u_idx = std::min(idx - 1, (int)control_traj_.size() - 1);
                logControl(control_traj_[u_idx]);
            }
        }
    }

    // ── Publishing ────────────────────────────────────────────────────────────
    void publishCommand(const Eigen::VectorXd& s) {
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";
        msg.pose.position.x    = s(0);
        msg.pose.position.y    = s(1);
        msg.pose.position.z    = s(2);
        msg.twist.linear.x     = s(3);
        msg.twist.linear.y     = s(4);
        msg.twist.linear.z     = s(5);
        msg.pose.orientation.w = s(6);
        msg.pose.orientation.x = s(7);
        msg.pose.orientation.y = s(8);
        msg.pose.orientation.z = s(9);
        msg.twist.angular.x    = s(10);
        msg.twist.angular.y    = s(11);
        msg.twist.angular.z    = s(12);
        cmd_pub_->publish(msg);
    }

    void publishTrajectory(const std::vector<Eigen::VectorXd>& traj) {
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

    // ── Logging ───────────────────────────────────────────────────────────────
    double wallTimeSec() {
        return std::chrono::duration<double>(
            Clock::now().time_since_epoch()).count();
    }

    void logActualState() {
        if (!actual_state_log_.is_open()) return;
        actual_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) actual_state_log_ << "," << current_state_(i);
        actual_state_log_ << "\n";
        actual_state_log_.flush();
    }

    void logCommandedState(const Eigen::VectorXd& s) {
        if (!commanded_state_log_.is_open() || s.size() < 13) return;
        commanded_state_log_ << std::fixed << std::setprecision(6) << wallTimeSec();
        for (int i = 0; i < 13; ++i) commanded_state_log_ << "," << s(i);
        commanded_state_log_ << "\n";
        commanded_state_log_.flush();
    }

    void logControl(const Eigen::VectorXd& u) {
        if (!control_log_.is_open() || u.size() < 6) return;
        Eigen::Vector3d f = u.segment(0, 3);
        Eigen::Vector3d m = u.segment(3, 3);
        control_log_ << std::fixed << std::setprecision(6) << wallTimeSec()
                     << "," << f(0) << "," << f(1) << "," << f(2)
                     << "," << m(0) << "," << m(1) << "," << m(2)
                     << "," << f.norm() << "\n";
        control_log_.flush();
    }

    void printDiagnostics(const QuadrotorMPC::Result& result) {
        Eigen::Vector3d pos = current_state_.segment(0, 3);
        Eigen::Vector3d vel = current_state_.segment(3, 3);
        double thrust = result.control_trajectory.empty() ? 0.0
            : result.control_trajectory[0].segment(0, 3).norm();
        double elapsed = std::chrono::duration<double>(
            Clock::now() - result.solve_timestamp).count();
        RCLCPP_INFO(this->get_logger(),
            "MPC %.1fms | Pos[%.3f,%.3f,%.3f] Vel[%.3f,%.3f,%.3f] | T=%.3fN idx~%d/%zu",
            result.solve_time_ms,
            pos.x(), pos.y(), pos.z(),
            vel.x(), vel.y(), vel.z(),
            thrust,
            (int)(elapsed / ocp_dt_) + 1,
            result.state_trajectory.size());
    }

    // ── Members ───────────────────────────────────────────────────────────────
    std::unique_ptr<QuadrotorMPC>  mpc_;
    std::string                    drone_name_;
    bool                           logging_enabled_;
    int                            control_rate_;
    int                            solver_rate_;
    double                         ocp_dt_ = 0.05;

    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr  cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                   traj_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr    pose_sub_;
    rclcpp::Subscription<crazyflie_interfaces::msg::LogDataGeneric>::SharedPtr vel_sub_;
    rclcpp::Subscription<crazyflie_interfaces::msg::LogDataGeneric>::SharedPtr ang_vel_sub_;
    rclcpp::TimerBase::SharedPtr   control_timer_;
    rclcpp::TimerBase::SharedPtr   solver_timer_;

    Eigen::VectorXd  current_state_;
    bool             pose_received_ = false;
    bool             vel_received_  = false;
    bool             is_flying_     = false;

    std::mutex                    traj_mutex_;
    std::vector<Eigen::VectorXd>  state_traj_;
    std::vector<Eigen::VectorXd>  control_traj_;
    TimePoint                     solve_time_;
    bool                          traj_valid_ = false;

    std::ofstream  commanded_state_log_;
    std::ofstream  actual_state_log_;
    std::ofstream  control_log_;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlannerNode>());
    rclcpp::shutdown();
    return 0;
}