/// @file px4_bridge.cpp
/// @brief CoManDO bridge for PX4 (uORB-over-DDS).
///
/// Subscribes:  /fmu/out/vehicle_odometry  (VehicleOdometry, NED/FRD)
/// Publishes:   /mpc/state                 (nav_msgs/Odometry, ENU/FLU) → solver
///
/// Control mode (ROS param "control_mode"):
///   "rates"      (default) — converts u0=(fz,Mx,My,Mz) →
///                            VehicleRatesSetpoint (body rates + thrust)
///   "trajectory"           — replays full X trajectory as
///                            TrajectorySetpoint (pos+vel+yaw per step)
///
/// Also handles PX4 offboard heartbeat and arm command.

#ifdef HAS_PX4_MSGS

#include "platform/platform_bridge_base.hpp"
#include "utils/frame_conv.hpp"

#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>

#include <limits>

using namespace std::chrono_literals;

class PX4Bridge : public PlatformBridgeBase {
public:
    PX4Bridge()
        : PlatformBridgeBase("px4_bridge")
    {
        // ── PX4-specific parameters ───────────────────────────────────────────
        this->declare_parameter("control_mode", std::string("rates"));
        this->declare_parameter("mass",         0.027);

        control_mode_ = this->get_parameter("control_mode").as_string();
        mass_         = this->get_parameter("mass").as_double();

        if (control_mode_ != "rates" && control_mode_ != "trajectory") {
            RCLCPP_ERROR(this->get_logger(),
                "[PX4Bridge] Unknown control_mode '%s' — defaulting to 'rates'",
                control_mode_.c_str());
            control_mode_ = "rates";
        }

        RCLCPP_INFO(this->get_logger(),
            "[PX4Bridge] control_mode=%s  mass=%.3fkg",
            control_mode_.c_str(), mass_);

        // Must be called here (not in base constructor) — pure virtual rule
        setupPlatformIO();
    }

protected:
    // ── Platform I/O ─────────────────────────────────────────────────────────
    void setupPlatformIO() override
    {
        // State subscription (NED/FRD → ENU/FLU conversion in callback)
        auto sensor_cb_group = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions sensor_opts;
        sensor_opts.callback_group = sensor_cb_group;

        odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", 10,
            std::bind(&PX4Bridge::odomCallback, this, std::placeholders::_1),
            sensor_opts);

        // Command publishers
        if (control_mode_ == "rates") {
            rates_pub_ = this->create_publisher<px4_msgs::msg::VehicleRatesSetpoint>(
                "/fmu/in/vehicle_rates_setpoint", 10);
        } else {
            traj_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
                "/fmu/in/trajectory_setpoint", 10);
        }

        offboard_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
            "/fmu/in/offboard_control_mode", 10);
        cmd_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>(
            "/fmu/in/vehicle_command", 10);

        // Heartbeat at 10Hz (PX4 requires >2Hz to stay in offboard)
        heartbeat_timer_ = this->create_wall_timer(
            100ms, std::bind(&PX4Bridge::heartbeatCallback, this));

        RCLCPP_INFO(this->get_logger(),
            "[PX4Bridge] Subscribed to /fmu/out/vehicle_odometry");
    }

    // ── Send command to PX4 ───────────────────────────────────────────────────
    void sendPlatformCommand(const Eigen::VectorXd& x_cmd,
                             const Eigen::VectorXd& u_cmd) override
    {
        if (!armed_) {
            px4SetOffboardMode();
            px4Arm();
            armed_ = true;
        }

        if (control_mode_ == "rates") {
            sendRatesSetpoint(u_cmd);
        } else {
            sendTrajectorySetpoint(x_cmd);
        }
    }

private:
    // ── Rates mode: u0 = (fz, Mx, My, Mz) → VehicleRatesSetpoint ────────────
    //
    // PX4 VehicleRatesSetpoint takes:
    //   roll, pitch, yaw  (rad/s in FRD body frame)
    //   thrust_body[3]    (normalised −1..1, body FRD: [0]=roll, [1]=pitch, [2]=−z)
    //
    // Conversion (simplified, assumes small angle Euler integration):
    //   desired_alpha = M / J → desired_omega_dot ≈ M / J
    //   For a single-step approximation:
    //   omega_des = M / J * ocp_dt_  (feedforward from moments)
    //
    // NOTE: this is a first-order approximation.  A full cascaded controller
    // would compute omega_des from the full attitude trajectory.  Suitable for
    // aggressive manoeuvres where the inner-loop rate controller is fast.
    void sendRatesSetpoint(const Eigen::VectorXd& u_cmd)
    {
        if (!rates_pub_) return;

        // Inertia (Crazyflie defaults — override with ROS params if needed)
        static constexpr double Jxx = 1.66e-5;
        static constexpr double Jyy = 1.66e-5;
        static constexpr double Jzz = 2.92e-5;

        const double ocp_dt = this->get_parameter("ocp_dt").as_double();

        // u_cmd = [fz_B (N), Mx (Nm·J_scaled), My, Mz]
        // Unscale moments: M_real = M_scaled / J_SCALE
        static constexpr double J_SCALE = 1.0 / 1.66e-5;
        const double Mx_real = (u_cmd.size() > 1) ? u_cmd(1) / J_SCALE : 0.0;
        const double My_real = (u_cmd.size() > 2) ? u_cmd(2) / J_SCALE : 0.0;
        const double Mz_real = (u_cmd.size() > 3) ? u_cmd(3) / J_SCALE : 0.0;
        const double fz      = (u_cmd.size() > 0) ? u_cmd(0)            : mass_ * 9.81;

        // Desired body rates (FLU) = M / J * dt
        const double wx_flu = Mx_real / Jxx * ocp_dt;
        const double wy_flu = My_real / Jyy * ocp_dt;
        const double wz_flu = Mz_real / Jzz * ocp_dt;

        // Convert FLU → FRD for PX4
        auto w_frd = frame_conv::omega_flu_to_frd({wx_flu, wy_flu, wz_flu});

        // Normalise thrust: PX4 expects −1..1 on body-z (−z = up)
        static constexpr double FMAX = 0.6;
        const float thrust_norm = static_cast<float>(
            std::max(-1.0, std::min(0.0, -fz / FMAX)));

        px4_msgs::msg::VehicleRatesSetpoint sp;
        sp.timestamp = this->now().nanoseconds() / 1000;
        sp.roll      = static_cast<float>(w_frd.x());
        sp.pitch     = static_cast<float>(w_frd.y());
        sp.yaw       = static_cast<float>(w_frd.z());
        sp.thrust_body[0] = 0.0f;
        sp.thrust_body[1] = 0.0f;
        sp.thrust_body[2] = thrust_norm;

        rates_pub_->publish(sp);
    }

    // ── Trajectory mode: x_cmd → TrajectorySetpoint ───────────────────────────
    void sendTrajectorySetpoint(const Eigen::VectorXd& x_cmd)
    {
        if (!traj_pub_) return;

        Eigen::Vector3d pos_ned = frame_conv::enu_to_ned({x_cmd(0), x_cmd(1), x_cmd(2)});
        Eigen::Vector3d vel_ned = frame_conv::enu_to_ned({x_cmd(3), x_cmd(4), x_cmd(5)});
        Eigen::Quaterniond q_enu(x_cmd(6), x_cmd(7), x_cmd(8), x_cmd(9));
        float yaw_ned = frame_conv::yaw_ned_from_enu_quat(q_enu);

        constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

        px4_msgs::msg::TrajectorySetpoint sp;
        sp.timestamp      = this->now().nanoseconds() / 1000;
        sp.position[0]    = static_cast<float>(pos_ned.x());
        sp.position[1]    = static_cast<float>(pos_ned.y());
        sp.position[2]    = static_cast<float>(pos_ned.z());
        sp.velocity[0]    = static_cast<float>(vel_ned.x());
        sp.velocity[1]    = static_cast<float>(vel_ned.y());
        sp.velocity[2]    = static_cast<float>(vel_ned.z());
        sp.acceleration[0] = sp.acceleration[1] = sp.acceleration[2] = NaN;
        sp.jerk[0]  = sp.jerk[1]  = sp.jerk[2]  = NaN;
        sp.yaw      = yaw_ned;
        sp.yawspeed = NaN;

        traj_pub_->publish(sp);
    }

    // ── NED/FRD → ENU/FLU state conversion ───────────────────────────────────
    void odomCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
    {
        Eigen::Vector3d pos_ned(msg->position[0], msg->position[1], msg->position[2]);
        Eigen::Vector3d vel_ned(msg->velocity[0], msg->velocity[1], msg->velocity[2]);
        Eigen::Vector3d omega_frd(msg->angular_velocity[0],
                                   msg->angular_velocity[1],
                                   msg->angular_velocity[2]);

        auto pos_enu   = frame_conv::ned_to_enu(pos_ned);
        auto vel_enu   = frame_conv::ned_to_enu(vel_ned);
        auto q_enu     = frame_conv::quat_ned_to_enu(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
        auto omega_flu = frame_conv::omega_frd_to_flu(omega_frd);

        Eigen::VectorXd state(13);
        state(0)  = pos_enu.x();   state(1)  = pos_enu.y();   state(2)  = pos_enu.z();
        state(3)  = vel_enu.x();   state(4)  = vel_enu.y();   state(5)  = vel_enu.z();
        state(6)  = q_enu.w();     state(7)  = q_enu.x();
        state(8)  = q_enu.y();     state(9)  = q_enu.z();
        state(10) = omega_flu.x(); state(11) = omega_flu.y(); state(12) = omega_flu.z();

        publishMpcState(state);
    }

    // ── PX4 heartbeat (OffboardControlMode) ──────────────────────────────────
    void heartbeatCallback()
    {
        px4_msgs::msg::OffboardControlMode msg;
        msg.timestamp    = this->now().nanoseconds() / 1000;
        msg.position     = (control_mode_ == "trajectory");
        msg.velocity     = (control_mode_ == "trajectory");
        msg.acceleration = false;
        msg.attitude     = false;
        msg.body_rate    = (control_mode_ == "rates");
        offboard_pub_->publish(msg);
    }

    void px4Arm()
    {
        px4_msgs::msg::VehicleCommand msg;
        msg.timestamp        = this->now().nanoseconds() / 1000;
        msg.command          = 400;
        msg.param1           = 1.0f;
        msg.target_system    = msg.source_system    = 1;
        msg.target_component = msg.source_component = 1;
        msg.from_external    = true;
        cmd_pub_->publish(msg);
    }

    void px4SetOffboardMode()
    {
        px4_msgs::msg::VehicleCommand msg;
        msg.timestamp        = this->now().nanoseconds() / 1000;
        msg.command          = 176;
        msg.param1           = 1.0f;
        msg.param2           = 6.0f;
        msg.target_system    = msg.source_system    = 1;
        msg.target_component = msg.source_component = 1;
        msg.from_external    = true;
        cmd_pub_->publish(msg);
    }

    // ── Members ───────────────────────────────────────────────────────────────
    std::string control_mode_ = "rates";
    double      mass_         = 0.027;
    bool        armed_        = false;

    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr       odom_sub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr      rates_pub_;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr        traj_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr       offboard_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr            cmd_pub_;
    rclcpp::TimerBase::SharedPtr                                           heartbeat_timer_;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PX4Bridge>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}

#else

#include <cstdio>
int main() {
    printf("[px4_bridge] Not compiled — px4_msgs not found.\n");
    return 1;
}

#endif // HAS_PX4_MSGS