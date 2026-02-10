#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
#include <crazyflie_interfaces/msg/log_data_generic.hpp>
#include "quadrotor_mpc.hpp"

#include <chrono>
#include <memory>
#include <filesystem> 
#include <fstream>
#include <iomanip>
#include <deque>
#include <Eigen/Dense>

// Helper function to convert quaternion to euler angles (rxyz convention)
Eigen::Vector3d quaternionToEuler(double w, double x, double y, double z) {
    Eigen::Vector3d euler;
    
    // Roll (x-axis rotation)
    double sinr_cosp = 2.0 * (w * x + y * z);
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    euler(0) = std::atan2(sinr_cosp, cosr_cosp);
    
    // Pitch (y-axis rotation)
    double sinp = 2.0 * (w * y - z * x);
    if (std::abs(sinp) >= 1)
        euler(1) = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        euler(1) = std::asin(sinp);
    
    // Yaw (z-axis rotation)
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    euler(2) = std::atan2(siny_cosp, cosy_cosp);
    
    return euler;
}

using namespace std::chrono_literals;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("comando_planner") {
        this->declare_parameter("drone_name", "cf_1");
        this->declare_parameter("enable_logging", true);
        this->declare_parameter("ocp_type", "hover");
        this->declare_parameter("control_rate", 50);  // Hz
        this->declare_parameter("solver_rate", 10);   // Hz
           
        drone_name_ = this->get_parameter("drone_name").as_string();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();
        std::string ocp_type = this->get_parameter("ocp_type").as_string();
        control_rate_ = this->get_parameter("control_rate").as_int();
        solver_rate_ = this->get_parameter("solver_rate").as_int();
        
        QuadrotorMPC::Config mpc_config;
        mpc_config.ocp_type = ocp_type;
        mpc_.reset(new QuadrotorMPC(mpc_config));
        
        if (logging_enabled_) {
            setupLogging();
        }
        
        // Publishers
        cmd_pub_ = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);
        traj_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/" + drone_name_ + "/planned_trajectory", 10);
        
        // Subscribers with standard QoS (like the example)
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            std::bind(&PlannerNode::poseCallback, this, std::placeholders::_1));
        
        vel_sub_ = this->create_subscription<crazyflie_interfaces::msg::LogDataGeneric>(
            "/" + drone_name_ + "/velocity", 10,
            std::bind(&PlannerNode::velocityCallback, this, std::placeholders::_1));
        
        // Initialize state
        position_.resize(3, 0.0);
        velocity_.resize(3, 0.0);
        attitude_.resize(3, 0.0);
        
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;  // Default quaternion
        
        is_flying_ = false;
        
        // Create timers (similar to example's dual-loop structure)
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / control_rate_),
            std::bind(&PlannerNode::controlLoop, this));
            
        solver_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / solver_rate_),
            std::bind(&PlannerNode::solverLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "CoManDO Planner started for %s", drone_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "OCP Type: %s", ocp_type.c_str());
        RCLCPP_INFO(this->get_logger(), "Control Rate: %d Hz, Solver Rate: %d Hz", 
                    control_rate_, solver_rate_);
    }

private:
    void setupLogging() {
        std::string log_dir = "./logs";
        std::filesystem::create_directories(log_dir);
        
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
        
        std::string flight_folder = log_dir + "/" + drone_name_ + "_flight_" + timestamp.str();
        std::filesystem::create_directories(flight_folder);
        
        std::string base_path = flight_folder + "/";
        commanded_state_log_.open(base_path + "commanded_state.csv");
        actual_state_log_.open(base_path + "actual_state.csv");
        control_log_.open(base_path + "control.csv");
        
        if (commanded_state_log_.is_open()) {
            commanded_state_log_ << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";
        }
        if (actual_state_log_.is_open()) {
            actual_state_log_ << "timestamp,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz\n";
        }
        if (control_log_.is_open()) {
            control_log_ << "timestamp,fx,fy,fz,mx,my,mz,thrust_norm\n";
        }
        
        RCLCPP_INFO(this->get_logger(), "Logging to: %s", flight_folder.c_str());
    }
    
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        // Update position
        position_[0] = msg->pose.position.x;
        position_[1] = msg->pose.position.y;
        position_[2] = msg->pose.position.z;
        
        // Update attitude (convert quaternion to euler angles)
        Eigen::Vector3d euler = quaternionToEuler(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z);
        
        attitude_[0] = euler(0);
        attitude_[1] = euler(1);
        attitude_[2] = euler(2);
        
        // Update current state
        current_state_(0) = position_[0];
        current_state_(1) = position_[1];
        current_state_(2) = position_[2];
        current_state_(6) = msg->pose.orientation.w;
        current_state_(7) = msg->pose.orientation.x;
        current_state_(8) = msg->pose.orientation.y;
        current_state_(9) = msg->pose.orientation.z;
    }
    
    void velocityCallback(const crazyflie_interfaces::msg::LogDataGeneric::SharedPtr msg) {
        if (msg->values.size() >= 3) {
            velocity_[0] = msg->values[0];
            velocity_[1] = msg->values[1];
            velocity_[2] = msg->values[2];
            
            // Update current state
            current_state_(3) = velocity_[0];
            current_state_(4) = velocity_[1];
            current_state_(5) = velocity_[2];
        }
    }
    
    // Solver loop - runs at solver_rate_ Hz (e.g., 10 Hz)
    void solverLoop() {
        // Check if we have valid state data
        if (position_.empty() || velocity_.empty() || attitude_.empty()) {
            static int warning_count = 0;
            if (warning_count++ % 10 == 0) {
                RCLCPP_WARN(this->get_logger(), "Waiting for state data...");
            }
            return;
        }
        
        if (!is_flying_) {
            is_flying_ = true;
            RCLCPP_INFO(this->get_logger(), "State received - starting flight");
        }
        
        // Build full state vector for MPC
        // For now, set angular velocities to zero (can be estimated from attitude changes)
        current_state_(10) = 0.0;  // omega_x
        current_state_(11) = 0.0;  // omega_y
        current_state_(12) = 0.0;  // omega_z
        
        // Solve MPC
        auto result = mpc_->solve(current_state_);
        
        if (result.success) {
            // Store the full solution in the control queue
            control_queue_.clear();
            for (const auto& control : result.control_trajectory) {
                control_queue_.push_back(control);
            }
            
            // Store state trajectory for visualization
            state_trajectory_ = result.state_trajectory;
            
            // Publish trajectory visualization
            publishTrajectory(state_trajectory_);
            
            // Log commanded state (next state in trajectory)
            if (logging_enabled_ && result.state_trajectory.size() > 1) {
                logCommandedState(result.state_trajectory[1]);
            }
            
            static int count = 0;
            if (++count % 10 == 0) {  // Print every 1 second at 10 Hz
                printDiagnostics(result);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "MPC solve failed");
        }
    }
    
    // Control loop - runs at control_rate_ Hz (e.g., 50 Hz)
    void controlLoop() {
        if (!is_flying_) {
            return;
        }
        
        if (position_.empty() || velocity_.empty() || attitude_.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "Empty state message in control loop");
            return;
        }
        
        // Log actual state at control rate
        if (logging_enabled_) {
            logActualState();
        }
        
        // Apply control from queue (if available)
        if (!control_queue_.empty()) {
            Eigen::VectorXd control = control_queue_.front();
            control_queue_.pop_front();
            
            // Publish command based on control
            publishControlCommand(control);
            
            // Log control
            if (logging_enabled_) {
                logControl(control);
            }
        } else {
            // No control available - could publish hover command or use last known good control
            static int warning_count = 0;
            if (warning_count++ % 50 == 0) {
                RCLCPP_DEBUG(this->get_logger(), "Control queue empty, waiting for MPC solution");
            }
        }
    }
    
    void publishControlCommand(const Eigen::VectorXd& control) {
        (void)control;  // Mark as intentionally unused for now
        // For now, we'll publish the next state from the MPC trajectory
        // In the future, you could compute this from the control input
        if (!state_trajectory_.empty() && state_trajectory_.size() > 1) {
            publishCommand(state_trajectory_[1]);
        }
    }
    
    void logActualState() {
        if (actual_state_log_.is_open()) {
            // Use wall time instead of frozen ROS time
            auto now = std::chrono::steady_clock::now();
            auto duration = now.time_since_epoch();
            double timestamp = std::chrono::duration<double>(duration).count();
            
            actual_state_log_ << std::fixed << std::setprecision(6) << timestamp;
            for (int i = 0; i < 13; ++i) {
                actual_state_log_ << "," << current_state_(i);
            }
            actual_state_log_ << "\n";
            actual_state_log_.flush();
        }
    }
    
    void logCommandedState(const Eigen::VectorXd& commanded_state) {
        if (commanded_state_log_.is_open()) {
            // Use wall time instead of frozen ROS time
            auto now = std::chrono::steady_clock::now();
            auto duration = now.time_since_epoch();
            double timestamp = std::chrono::duration<double>(duration).count();
            
            commanded_state_log_ << std::fixed << std::setprecision(6) << timestamp;
            for (int i = 0; i < 13; ++i) {
                commanded_state_log_ << "," << commanded_state(i);
            }
            commanded_state_log_ << "\n";
            commanded_state_log_.flush();
        }
    }
    
    void logControl(const Eigen::VectorXd& control) {
        if (control_log_.is_open() && control.size() >= 6) {
            // Use wall time instead of frozen ROS time
            auto now = std::chrono::steady_clock::now();
            auto duration = now.time_since_epoch();
            double timestamp = std::chrono::duration<double>(duration).count();
            
            Eigen::Vector3d force = control.segment(0, 3);
            Eigen::Vector3d moment = control.segment(3, 3);
            double thrust_norm = force.norm();
            
            control_log_ << std::fixed << std::setprecision(6) << timestamp;
            control_log_ << "," << force(0) << "," << force(1) << "," << force(2);
            control_log_ << "," << moment(0) << "," << moment(1) << "," << moment(2);
            control_log_ << "," << thrust_norm << "\n";
            control_log_.flush();
        }
    }
    
    void printDiagnostics(const QuadrotorMPC::Result& result) {
        Eigen::Vector3d current_pos = current_state_.segment(0, 3);
        Eigen::Vector3d current_vel = current_state_.segment(3, 3);
        Eigen::Vector3d target_pos(0.0, 0.0, 1.0);
        double distance = (current_pos - target_pos).norm();
        
        double thrust_norm = 0.0;
        if (!result.control_trajectory.empty()) {
            Eigen::Vector3d force = result.control_trajectory[0].segment(0, 3);
            thrust_norm = force.norm();
        }
        
        RCLCPP_INFO(this->get_logger(), 
                   "MPC: %.1fms | Pos[%.3f,%.3f,%.3f] Vel[%.3f,%.3f,%.3f] | Dist=%.3fm T=%.3fN",
                   result.solve_time_ms,
                   current_pos.x(), current_pos.y(), current_pos.z(),
                   current_vel.x(), current_vel.y(), current_vel.z(),
                   distance, thrust_norm);
    }
    
    void publishCommand(const Eigen::VectorXd& state) {
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "world";
        
        msg.pose.position.x = state(0);
        msg.pose.position.y = state(1);
        msg.pose.position.z = state(2);
        
        msg.twist.linear.x = state(3);
        msg.twist.linear.y = state(4);
        msg.twist.linear.z = state(5);
        
        msg.pose.orientation.w = state(6);
        msg.pose.orientation.x = state(7);
        msg.pose.orientation.y = state(8);
        msg.pose.orientation.z = state(9);
        
        msg.twist.angular.x = state(10);
        msg.twist.angular.y = state(11);
        msg.twist.angular.z = state(12);
        
        cmd_pub_->publish(msg);
    }
    
    void publishTrajectory(const std::vector<Eigen::VectorXd>& trajectory) {
        nav_msgs::msg::Path path;
        path.header.stamp = this->now();
        path.header.frame_id = "world";
        
        for (const auto& state : trajectory) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = "world";
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
    
    // Member variables
    std::unique_ptr<QuadrotorMPC> mpc_;
    std::string drone_name_;
    bool logging_enabled_;
    int control_rate_;
    int solver_rate_;
    
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cmd_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<crazyflie_interfaces::msg::LogDataGeneric>::SharedPtr vel_sub_;
    
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr solver_timer_;
    
    // State storage (similar to example)
    std::vector<double> position_;
    std::vector<double> velocity_;
    std::vector<double> attitude_;
    
    Eigen::VectorXd current_state_;
    std::vector<Eigen::VectorXd> state_trajectory_;
    std::deque<Eigen::VectorXd> control_queue_;
    
    bool is_flying_;
    
    std::ofstream commanded_state_log_;
    std::ofstream actual_state_log_;
    std::ofstream control_log_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}