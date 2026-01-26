#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
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
        this->declare_parameter("control_frequency", 50.0);
        
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
        // velocities (3-5, 10-12) already zero
        
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
        
        // Setup TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Initialize state properly
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;  // Identity quaternion
        
        // Set a safe initial position (0,0,0)
        prev_position_ = Eigen::Vector3d::Zero();
        prev_orientation_ = Eigen::Quaterniond::Identity();
        first_iteration_ = true;
        
        // Control timer
        double period = 1.0 / control_freq;
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(period),
            std::bind(&PlannerNode::controlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "CoManDO Planner started for %s", drone_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "Control frequency: %.1f Hz", control_freq);
    }

private:
    bool getCurrentState() {
        try {
            // Try to get transform with timeout
            auto transform = tf_buffer_->lookupTransform(
                world_frame_,
                drone_name_,
                tf2::TimePointZero,
                std::chrono::milliseconds(100));
            
            // Position
            current_state_(0) = transform.transform.translation.x;
            current_state_(1) = transform.transform.translation.y;
            current_state_(2) = transform.transform.translation.z;
            
            // Quaternion
            current_state_(6) = transform.transform.rotation.w;
            current_state_(7) = transform.transform.rotation.x;
            current_state_(8) = transform.transform.rotation.y;
            current_state_(9) = transform.transform.rotation.z;
            
            // Estimate velocities
            rclcpp::Time current_time = transform.header.stamp;
            
            if (!first_iteration_) {
                double dt = (current_time - prev_time_).seconds();
                
                if (dt > 0.001 && dt < 0.1) {
                    // Linear velocity
                    Eigen::Vector3d pos(current_state_(0), current_state_(1), current_state_(2));
                    Eigen::Vector3d vel = (pos - prev_position_) / dt;
                    current_state_.segment(3, 3) = vel;
                    
                    // Angular velocity
                    Eigen::Quaterniond q_now(current_state_(6), current_state_(7), 
                                             current_state_(8), current_state_(9));
                    Eigen::Quaterniond delta_q = q_now * prev_orientation_.inverse();
                    Eigen::AngleAxisd aa(delta_q);
                    Eigen::Vector3d omega = aa.axis() * aa.angle() / dt;
                    current_state_.segment(10, 3) = omega;
                    
                    RCLCPP_DEBUG(this->get_logger(), "Vel estimated: [%.2f, %.2f, %.2f] m/s", 
                                vel.x(), vel.y(), vel.z());
                }
            }
            
            prev_time_ = current_time;
            prev_position_ = Eigen::Vector3d(current_state_(0), current_state_(1), current_state_(2));
            prev_orientation_ = Eigen::Quaterniond(current_state_(6), current_state_(7), 
                                                  current_state_(8), current_state_(9));
            first_iteration_ = false;
            
            RCLCPP_DEBUG(this->get_logger(), "Current state: pos=[%.3f, %.3f, %.3f], q=[%.3f, %.3f, %.3f, %.3f]",
                        current_state_(0), current_state_(1), current_state_(2),
                        current_state_(6), current_state_(7), current_state_(8), current_state_(9));
            
            return true;
            
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "TF error: %s", ex.what());
            return false;
        }
    }
    
    void controlLoop() {
        // Get current state
        if (!getCurrentState()) {
            RCLCPP_DEBUG(this->get_logger(), "Waiting for TF transform...");
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
            // DEBUG: Log commanded state
            RCLCPP_DEBUG(this->get_logger(), "Commanded state: pos=[%.3f, %.3f, %.3f]", 
                        result.next_state(0), result.next_state(1), result.next_state(2));
            
            publishCommand(result.next_state);
            publishTrajectory(result.state_trajectory);
            
            static int count = 0;
            if (++count % 10 == 0) {  // Log every 10 iterations
                RCLCPP_INFO(this->get_logger(), "MPC solved in %.1f ms", result.solve_time_ms);
                
                // Log current and target positions
                Eigen::Vector3d current_pos = current_state_.segment(0, 3);
                Eigen::Vector3d target_pos = Eigen::Vector3d(0, 0, 1.0);
                double distance = (current_pos - target_pos).norm();
                
                // Log control effort
                if (!result.control_trajectory.empty()) {
                    Eigen::VectorXd first_control = result.control_trajectory[0];
                    Eigen::Vector3d force = first_control.segment(0, 3);
                    double thrust_norm = force.norm();
                    RCLCPP_INFO(this->get_logger(), 
                               "Current: [%.2f, %.2f, %.2f], Dist: %.2f m, Thrust: %.3f N", 
                               current_pos.x(), current_pos.y(), current_pos.z(), 
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
        
        // Debug: Verify we're sending FullState (not Control)
        RCLCPP_DEBUG(this->get_logger(), "Publishing FullState cmd: pos=[%.3f, %.3f, %.3f], vel=[%.2f, %.2f, %.2f]",
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z);
        
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
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::string drone_name_;
    std::string world_frame_;
    
    Eigen::VectorXd current_state_;
    Eigen::Vector3d prev_position_;
    Eigen::Quaterniond prev_orientation_;
    rclcpp::Time prev_time_;
    bool first_iteration_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}