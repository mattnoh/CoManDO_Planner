/// @file sim_bridge.cpp
/// @brief Standalone sim bridge — async ROS wrapper of quad_cf_rh_noise.cpp loop.
///
/// Standalone loop (synchronous):
///   x_true = simStepPerfect(x_true, U[0])   ← x_true becomes X[1]
///   x_measured = x_true + noise
///   warmStart(x_measured, U[1:])
///   solve
///
/// This does the same thing over ROS messages:
///   on MpcCommand → publish X[1] + noise → planner warms from X[1] → solves
///
/// IDX is ALWAYS 1. Not n_shift. Not replay_start_idx.
/// Perfect dynamics: X[1] = dynamics(X[0], U[0]) exactly.
/// Using X[n_shift] here (n_shift=5 at 2Hz/ocp_dt=0.05) is identical to
/// running the standalone with N_SHIFT=5 — which diverges.

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "comando_planner/msg/mpc_command.hpp"

#include <Eigen/Dense>
#include <random>

struct NoiseProfile {
    double pos = 0.0, vel = 0.0, att = 0.0, omega = 0.0;
};

static NoiseProfile noiseFromString(const std::string& mode) {
    NoiseProfile p;
    if      (mode == "moderate") { p.pos=0.005; p.vel=0.010; p.att=0.005; p.omega=0.010; }
    else if (mode == "large")    { p.pos=0.020; p.vel=0.050; p.att=0.020; p.omega=0.050; }
    return p;
}

class SimBridge : public rclcpp::Node {
public:
    SimBridge() : Node("sim_bridge"), rng_(std::random_device{}())
    {
        this->declare_parameter("noise_mode", std::string("none"));
        this->declare_parameter("x0_x", 0.0);
        this->declare_parameter("x0_y", 0.0);
        this->declare_parameter("x0_z", 1.0);

        noise_ = noiseFromString(this->get_parameter("noise_mode").as_string());

        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(0) = this->get_parameter("x0_x").as_double();
        current_state_(1) = this->get_parameter("x0_y").as_double();
        current_state_(2) = this->get_parameter("x0_z").as_double();
        current_state_(6) = 1.0;

        state_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/mpc/state", 10);

        cmd_sub_ = this->create_subscription<comando_planner::msg::MpcCommand>(
            "/mpc/command", 10,
            std::bind(&SimBridge::onCommand, this, std::placeholders::_1));

        // Publish initial state at 10Hz until first command arrives
        init_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            [this]() { publish(current_state_); });

        RCLCPP_INFO(this->get_logger(), "[SimBridge] x0=[%.2f,%.2f,%.2f] noise=%s",
            current_state_(0), current_state_(1), current_state_(2),
            this->get_parameter("noise_mode").as_string().c_str());
    }

private:
    void onCommand(const comando_planner::msg::MpcCommand::SharedPtr msg)
    {
        if (!msg->success) return;

        // Kill the init timer on first command
        if (init_timer_) { init_timer_->cancel(); init_timer_.reset(); }

        const int nx = msg->nx;

        // Always X[1] — matches standalone N_SHIFT=1 exactly.
        // X[1] = dynamics(X[0], U[0]) under perfect dynamics.
        // Do NOT use n_shift or replay_start_idx — those compensate for
        // solve latency on hardware and have nothing to do with the plant.
        constexpr int IDX = 1;

        if (static_cast<int>(msg->state_trajectory.size()) < (IDX + 1) * nx) {
            RCLCPP_ERROR(this->get_logger(), "[SimBridge] Trajectory too short");
            return;
        }

        Eigen::VectorXd x_next(nx);
        for (int i = 0; i < nx; ++i)
            x_next(i) = msg->state_trajectory[IDX * nx + i];

        current_state_ = addNoise(x_next);
        publish(current_state_);

        RCLCPP_INFO(this->get_logger(), "[SimBridge] step=%d  x0=[%.4f,%.4f,%.4f]",
            ++step_, current_state_(0), current_state_(1), current_state_(2));
    }

    void publish(const Eigen::VectorXd& s)
    {
        nav_msgs::msg::Odometry msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";
        msg.pose.pose.position.x    = s(0);
        msg.pose.pose.position.y    = s(1);
        msg.pose.pose.position.z    = s(2);
        msg.twist.twist.linear.x    = s(3);
        msg.twist.twist.linear.y    = s(4);
        msg.twist.twist.linear.z    = s(5);
        msg.pose.pose.orientation.w = s(6);
        msg.pose.pose.orientation.x = s(7);
        msg.pose.pose.orientation.y = s(8);
        msg.pose.pose.orientation.z = s(9);
        msg.twist.twist.angular.x   = s(10);
        msg.twist.twist.angular.y   = s(11);
        msg.twist.twist.angular.z   = s(12);
        state_pub_->publish(msg);
    }

    Eigen::VectorXd addNoise(const Eigen::VectorXd& s)
    {
        if (noise_.pos == 0.0) return s;
        Eigen::VectorXd n = s;
        auto g = [&](double sig){ return std::normal_distribution<double>(0,sig)(rng_); };
        n(0)+=g(noise_.pos); n(1)+=g(noise_.pos); n(2)+=g(noise_.pos);
        n(3)+=g(noise_.vel); n(4)+=g(noise_.vel); n(5)+=g(noise_.vel);
        n(7)+=g(noise_.att); n(8)+=g(noise_.att); n(9)+=g(noise_.att);
        double qn = n.segment(6,4).norm();
        if (qn > 1e-6) n.segment(6,4) /= qn;
        n(10)+=g(noise_.omega); n(11)+=g(noise_.omega); n(12)+=g(noise_.omega);
        return n;
    }

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr             state_pub_;
    rclcpp::Subscription<comando_planner::msg::MpcCommand>::SharedPtr cmd_sub_;
    rclcpp::TimerBase::SharedPtr init_timer_;

    Eigen::VectorXd current_state_;
    NoiseProfile    noise_;
    std::mt19937    rng_;
    int             step_ = 0;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimBridge>());
    rclcpp::shutdown();
    return 0;
}