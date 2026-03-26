/// @file circular_target_publisher.cpp
/// @brief Publishes a circular trajectory synchronized with the RH solver logic.
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>


#include <cmath>
#include <chrono>


class CircularTargetPublisher : public rclcpp::Node {
public:
    CircularTargetPublisher() : Node("circular_target_publisher") {
        // 1. Parameters - Matched to quad_cf_tracking_rh_circle.cpp
        center_x_ = this->declare_parameter("center_x", 0.0);
        center_y_ = this->declare_parameter("center_y", 0.0);
        center_z_ = this->declare_parameter("center_z", 0.5); // Landing height
        radius_   = this->declare_parameter("radius", 1.0);
        omega_    = this->declare_parameter("omega", 0.1);
        phi0_     = this->declare_parameter("phi0", 0.0);
        publish_hz_ = this->declare_parameter("publish_hz", 100.0);
        frame_id_   = this->declare_parameter("frame_id", "world");

        // 2. Publishers
        odom_pub_  = this->create_publisher<nav_msgs::msg::Odometry>("/target/odom", 10);
        accel_pub_ = this->create_publisher<geometry_msgs::msg::AccelStamped>("/target/accel", 10);

        // 3. Timer - Internal state clock
        start_time_ = this->now();
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / publish_hz_)),
            std::bind(&CircularTargetPublisher::timer_callback, this));
            
        RCLCPP_INFO(this->get_logger(), "Circular Target Started: R=%f, W=%f", radius_, omega_);
    }

private:
    void timer_callback() {
        auto now = this->now();
        // Calculate t_abs relative to node start (matches simulation t_abs)
        double t = (now - start_time_).seconds();
        double ph = omega_ * t + phi0_;

        // --- Position (Analytical matching CircularTarget::pos) ---
        double px = center_x_ + radius_ * std::cos(ph);
        double py = center_y_ + radius_ * std::sin(ph);
        double pz = center_z_;

        // --- Velocity (Analytical matching CircularTarget::vel) ---
        double vx = -radius_ * omega_ * std::sin(ph);
        double vy =  radius_ * omega_ * std::cos(ph);
        double vz = 0.0;

        // --- Acceleration (Analytical matching CircularTarget::dvel_dt) ---
        double ax = -radius_ * omega_ * omega_ * std::cos(ph);
        double ay = -radius_ * omega_ * omega_ * std::sin(ph);
        double az = 0.0;

        // 4. Publish Odometry (for State Estimation)
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = now;
        odom.header.frame_id = frame_id_;
        odom.pose.pose.position.x = px;
        odom.pose.pose.position.y = py;
        odom.pose.pose.position.z = pz;
        odom.twist.twist.linear.x = vx;
        odom.twist.twist.linear.y = vy;
        odom.twist.twist.linear.z = vz;
        odom_pub_->publish(odom);

        // 5. Publish Accel (for Relative Dynamics Snapshot)
        geometry_msgs::msg::AccelStamped accel;
        accel.header.stamp = now;
        accel.accel.linear.x = ax;
        accel.accel.linear.y = ay;
        accel.accel.linear.z = az;
        accel_pub_->publish(accel);

    }

    double center_x_, center_y_, center_z_, radius_, omega_, phi0_, publish_hz_;
    std::string frame_id_;
    rclcpp::Time start_time_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr accel_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CircularTargetPublisher>());
    rclcpp::shutdown();
    return 0;
}