/// @file crazyflie_bridge.cpp
/// @brief CoManDO bridge for Crazyflie (Crazyswarm2).
///
/// Subscribes:  /<drone_name>/pose   (geometry_msgs/PoseStamped)
///              /<drone_name>/odom   (nav_msgs/Odometry)
/// Publishes:   /mpc/state           (nav_msgs/Odometry)      → solver
///              /<drone_name>/cmd_full_state (FullState)       → Crazyflie
///
/// The base class (PlatformBridgeBase) handles:
///   • Subscribing to /mpc/command
///   • Replay timer + transition blend
///   • Publishing /mpc/state

#include "platform/platform_bridge_base.hpp"
#include "utils/logger.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>

class CrazyflieBridge : public PlatformBridgeBase {
public:
    CrazyflieBridge()
        : PlatformBridgeBase("crazyflie_bridge")
    {
        // Logging setup (drone_name_ and logging_enabled_ set by base class)
        if (logging_enabled_) {
            this->declare_parameter("ocp_type",   std::string("hover"));
            this->declare_parameter("mode",       std::string("mpc"));
            this->declare_parameter("solver",     std::string("alipddp"));
            std::string ocp_type   = this->get_parameter("ocp_type").as_string();
            std::string mode       = this->get_parameter("mode").as_string();
            std::string solver     = this->get_parameter("solver").as_string();
            logger_.setup(drone_name_, ocp_type, mode, solver);
        }

        // Must be called here (not in base constructor) — pure virtual rule
        setupPlatformIO();
    }

protected:
    // ── Platform I/O ─────────────────────────────────────────────────────────
    void setupPlatformIO() override
    {
        // Command publisher
        cf_cmd_pub_ = this->create_publisher<crazyflie_interfaces::msg::FullState>(
            "/" + drone_name_ + "/cmd_full_state", 10);

        // State subscriptions
        auto sensor_cb_group = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions sensor_opts;
        sensor_opts.callback_group = sensor_cb_group;

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/" + drone_name_ + "/pose", 10,
            std::bind(&CrazyflieBridge::poseCallback, this, std::placeholders::_1),
            sensor_opts);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/" + drone_name_ + "/odom", 10,
            std::bind(&CrazyflieBridge::odomCallback, this, std::placeholders::_1),
            sensor_opts);

        RCLCPP_INFO(this->get_logger(),
            "[CrazyflieBridge] Subscribed to /%s/pose and /%s/odom",
            drone_name_.c_str(), drone_name_.c_str());
    }

    // ── Convert and send to Crazyflie ─────────────────────────────────────────
    // Crazyflie FullState: pos, vel, quat, angular rate, acceleration
    // We pass acceleration = 0 (Mellinger computes it from attitude PD).
    void sendPlatformCommand(const Eigen::VectorXd& x_cmd,
                             const Eigen::VectorXd& /* u_cmd */) override
    {
        crazyflie_interfaces::msg::FullState msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";

        msg.pose.position.x    = x_cmd(0);
        msg.pose.position.y    = x_cmd(1);
        msg.pose.position.z    = x_cmd(2);
        msg.twist.linear.x     = x_cmd(3);
        msg.twist.linear.y     = x_cmd(4);
        msg.twist.linear.z     = x_cmd(5);
        msg.pose.orientation.w = x_cmd(6);
        msg.pose.orientation.x = x_cmd(7);
        msg.pose.orientation.y = x_cmd(8);
        msg.pose.orientation.z = x_cmd(9);
        msg.twist.angular.x    = x_cmd(10);
        msg.twist.angular.y    = x_cmd(11);
        msg.twist.angular.z    = x_cmd(12);
        msg.acc.x = 0.0;
        msg.acc.y = 0.0;
        msg.acc.z = 0.0;

        cf_cmd_pub_->publish(msg);

        if (logging_enabled_ && logger_.isInitialized())
            logger_.logPublishedCommand(x_cmd, Eigen::Vector3d::Zero());
    }

private:
    // ── Crazyflie state callbacks ─────────────────────────────────────────────

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(state_mutex_);
        state_(0) = msg->pose.position.x;
        state_(1) = msg->pose.position.y;
        state_(2) = msg->pose.position.z;
        state_(6) = msg->pose.orientation.w;
        state_(7) = msg->pose.orientation.x;
        state_(8) = msg->pose.orientation.y;
        state_(9) = msg->pose.orientation.z;
        pose_received_ = true;
        maybePublishState();
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Crazyswarm2 publishes angular rates in deg/s — convert to rad/s
        constexpr double DEG2RAD = M_PI / 180.0;
        std::lock_guard<std::mutex> lk(state_mutex_);
        state_(3)  = msg->twist.twist.linear.x;
        state_(4)  = msg->twist.twist.linear.y;
        state_(5)  = msg->twist.twist.linear.z;
        state_(10) = msg->twist.twist.angular.x * DEG2RAD;
        state_(11) = msg->twist.twist.angular.y * DEG2RAD;
        state_(12) = msg->twist.twist.angular.z * DEG2RAD;
        odom_received_ = true;
        maybePublishState();
    }

    // Publish to /mpc/state only when we have both pose and odom
    void maybePublishState()
    {
        // state_mutex_ already held by caller
        if (pose_received_ && odom_received_)
            publishMpcState(state_);
    }

    // ── Members ───────────────────────────────────────────────────────────────
    rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr cf_cmd_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr   pose_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           odom_sub_;

    // Local state buffer (assembled from pose + odom)
    std::mutex      state_mutex_;
    Eigen::VectorXd state_ = []() {
        Eigen::VectorXd s = Eigen::VectorXd::Zero(13); s(6) = 1.0; return s;
    }();
    bool pose_received_ = false;
    bool odom_received_ = false;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CrazyflieBridge>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}