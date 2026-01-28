#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
#include <crazyflie_interfaces/msg/log_data_generic.hpp>
#include "quadrotor_mpc.hpp"

#include <chrono>
#include <memory>
#include <Eigen/Dense>

using namespace std::chrono_literals;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        // ROS parameters
        this->declare_parameter("drone_name", "cf_1");
        this->declare_parameter("world_frame", "world");
        this->declare_parameter("control_frequency", 100.0);
        
        // Get parameters
        drone_name_ = this->get_parameter("drone_name").as_string();
        world_frame_ = this->get_parameter("world_frame").as_string();
        double control_freq = this->get_parameter("control_frequency").as_double();
        
        // Setup MPC
        QuadrotorMPC::Config mpc_config;
        mpc_config.horizon = 50;
        mpc_config.dt = 1.0 / control_freq;
        
        // Set terminal state PROPERLY - with ZERO VELOCITY!
        mpc_config.terminal_state = Eigen::VectorXd::Zero(13);
        mpc_config.terminal_state(0) = 0.0;  // x = 0
        mpc_config.terminal_state(1) = 0.0;  // y = 0
        mpc_config.terminal_state(2) = 1.0;  // z = 1m
        mpc_config.terminal_state(6) = 1.0;  // upright quaternion (w=1)
        
        RCLCPP_INFO(this->get_logger(), "MPC Config:");
        RCLCPP_INFO(this->get_logger(), "  Horizon: %d", mpc_config.horizon);
        RCLCPP_INFO(this->get_logger(), "  DT: %.3f", mpc_config.dt);
        RCLCPP_INFO(this->get_logger(), "  Terminal state: [%.2f, %.2f, %.2f]", 
                   mpc_config.terminal_state(0), mpc_config.terminal_state(1), mpc_config.terminal_state(2));
        
        mpc_.reset(new QuadrotorMPC(mpc_config));
        
        // Setup publishers
        cmd_pub_ = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);
        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);
        
        // Subscribe to pose topic
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            std::bind(&PlannerNode::poseCallback, this, std::placeholders::_1));
        
        // Subscribe to velocity topic
        vel_sub_ = this->create_subscription<crazyflie_interfaces::msg::LogDataGeneric>(
            "/" + drone_name_ + "/velocity", 10,
            std::bind(&PlannerNode::velocityCallback, this, std::placeholders::_1));
        
        // Subscribe to acceleration topic (optional, for future use)
        acc_sub_ = this->create_subscription<crazyflie_interfaces::msg::LogDataGeneric>(
            "/" + drone_name_ + "/acceleration", 10,
            std::bind(&PlannerNode::accelerationCallback, this, std::placeholders::_1));
        
        // Initialize state properly
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;  // Identity quaternion
        pose_received_ = false;
        vel_received_ = false;
        
        // Control timer
        double period = 1.0 / control_freq;
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(period),
            std::bind(&PlannerNode::controlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "CoManDO Planner started for %s", drone_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "Control frequency: %.1f Hz", control_freq);
    }

private:
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        // Position
        current_state_(0) = msg->pose.position.x;
        current_state_(1) = msg->pose.position.y;
        current_state_(2) = msg->pose.position.z;
        
        // Quaternion
        current_state_(6) = msg->pose.orientation.w;
        current_state_(7) = msg->pose.orientation.x;
        current_state_(8) = msg->pose.orientation.y;
        current_state_(9) = msg->pose.orientation.z;
        
        pose_received_ = true;
    }
    
    void velocityCallback(const crazyflie_interfaces::msg::LogDataGeneric::SharedPtr msg) {
        // Velocity data is in msg->values array: [vx, vy, vz]
        if (msg->values.size() >= 3) {
            current_state_(3) = msg->values[0];  // vx
            current_state_(4) = msg->values[1];  // vy
            current_state_(5) = msg->values[2];  // vz
            
            vel_received_ = true;
        }
    }
    
    void accelerationCallback(const crazyflie_interfaces::msg::LogDataGeneric::SharedPtr msg) {
        // Store for future use if needed
        // Acceleration data is in msg->values array: [ax, ay, az]
        if (msg->values.size() >= 3) {
            // Currently not used in MPC, but available
        }
    }
    
    void controlLoop() {
        // Check if we have received both pose and velocity
        if (!pose_received_ || !vel_received_) {
            RCLCPP_DEBUG(this->get_logger(), "Waiting for pose and velocity data...");
            return;
        }
        
        // Check if state is valid
        if (current_state_.size() != 13) {
            RCLCPP_ERROR(this->get_logger(), "Invalid state size!");
            return;
        }
        
        // Solve MPC
        auto result = mpc_->solve(current_state_);

        if (result.success) {
            publishCommand(result.next_state);
            publishTrajectory(result.state_trajectory);
            
            static int count = 0;
            if (++count % 10 == 0) {  // Log every 10 iterations
                RCLCPP_INFO(this->get_logger(), "MPC solved in %.1f ms", result.solve_time_ms);
                
                // Log current and target positions
                Eigen::Vector3d current_pos = current_state_.segment(0, 3);
                Eigen::Vector3d current_vel = current_state_.segment(3, 3);
                Eigen::Vector3d target_pos = Eigen::Vector3d(0, 0, 1.0);
                double distance = (current_pos - target_pos).norm();
                
                // Log control effort
                if (!result.control_trajectory.empty()) {
                    Eigen::VectorXd first_control = result.control_trajectory[0];
                    Eigen::Vector3d force = first_control.segment(0, 3);
                    double thrust_norm = force.norm();
                    RCLCPP_INFO(this->get_logger(), 
                               "Pos: [%.2f, %.2f, %.2f], Vel: [%.2f, %.2f, %.2f], Dist: %.2f m, Thrust: %.3f N", 
                               current_pos.x(), current_pos.y(), current_pos.z(),
                               current_vel.x(), current_vel.y(), current_vel.z(),
                               distance, thrust_norm);
                }
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "MPC solve failed");
        }
    }
    
    void publishCommand(const Eigen::VectorXd& state) {
        if (state.size() != 13) {
            RCLCPP_ERROR(this->get_logger(), "Invalid state size for publishing!");
            return;
        }
        
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = world_frame_;
        
        // Position
        msg.pose.position.x = state(0);
        msg.pose.position.y = state(1);
        msg.pose.position.z = state(2);
        
        // Velocity
        msg.twist.linear.x = state(3);
        msg.twist.linear.y = state(4);
        msg.twist.linear.z = state(5);
        
        // Quaternion
        Eigen::Quaterniond q(state(6), state(7), state(8), state(9));
        if (q.norm() < 1e-6) q = Eigen::Quaterniond::Identity();
        q.normalize();
        
        msg.pose.orientation.w = q.w();
        msg.pose.orientation.x = q.x();
        msg.pose.orientation.y = q.y();
        msg.pose.orientation.z = q.z();
        
        // Angular velocity
        msg.twist.angular.x = state(10);
        msg.twist.angular.y = state(11);
        msg.twist.angular.z = state(12);
        
        cmd_pub_->publish(msg);
    }
    
    void publishTrajectory(const std::vector<Eigen::VectorXd>& trajectory) {
        nav_msgs::msg::Path path;
        path.header.stamp = this->now();
        path.header.frame_id = world_frame_;
        
        for (const auto& state : trajectory) {
            if (state.size() != 13) continue;
            
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = world_frame_;
            pose.pose.position.x = state(0);
            pose.pose.position.y = state(1);
            pose.pose.position.z = state(2);
            pose.pose.orientation.w = state(6);
            pose.pose.orientation.x = state(7);
            pose.pose.orientation.y = state(8);
            pose.pose.orientation.z = state(9);
            path.poses.push_back(pose);
        }
        
        traj_pub_->publish(path);
    }
    
    // Members
    std::unique_ptr<QuadrotorMPC> mpc_;
    
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<crazyflie_interfaces::msg::LogDataGeneric>::SharedPtr vel_sub_;
    rclcpp::Subscription<crazyflie_interfaces::msg::LogDataGeneric>::SharedPtr acc_sub_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::string drone_name_;
    std::string world_frame_;
    
    Eigen::VectorXd current_state_;
    bool pose_received_;
    bool vel_received_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}