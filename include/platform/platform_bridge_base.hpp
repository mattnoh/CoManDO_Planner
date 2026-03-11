/// @file platform_bridge_base.hpp
/// @brief Abstract base for all CoManDO platform bridges.
///
/// Every bridge (Crazyflie, PX4, sim, ...) inherits this class and implements
/// two pure-virtual methods:
///
///   sendPlatformCommand(x_cmd, u_cmd)  — convert + send one setpoint
///   setupPlatformIO()                  — subscribe to platform state topics,
///                                        call publishMpcState() in callbacks
///
/// The base class handles everything else:
///   • Subscribes to /mpc/command  (MpcCommand)
///   • Creates the replay timer at ocp_dt Hz on first command
///   • Runs the trajectory replay + transition blend
///   • Publishes platform state as nav_msgs/Odometry on /mpc/state
///   • Logging (optional, enabled via ROS param "enable_logging")
///
/// ── Adding a new platform bridge ──────────────────────────────────────────
///   1. Create src/bridges/<platform>_bridge.cpp
///   2. Include this header, inherit PlatformBridgeBase
///   3. Implement sendPlatformCommand() and setupPlatformIO()
///   4. Add executable in CMakeLists.txt
///   Done — the solver node is untouched.
/// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "comando_planner/msg/mpc_command.hpp"
#include "utils/logger.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <mutex>
#include <vector>

class PlatformBridgeBase : public rclcpp::Node {
public:
    explicit PlatformBridgeBase(const std::string& node_name,
                                const rclcpp::NodeOptions& opts = rclcpp::NodeOptions())
        : Node(node_name, opts)
    {
        // ── Common parameters ─────────────────────────────────────────────────
        this->declare_parameter("drone_name",     std::string("cf_1"));
        this->declare_parameter("ocp_dt",         0.05);
        this->declare_parameter("enable_logging", false);

        drone_name_      = this->get_parameter("drone_name").as_string();
        ocp_dt_          = this->get_parameter("ocp_dt").as_double();
        logging_enabled_ = this->get_parameter("enable_logging").as_bool();

        // ── Initial state ─────────────────────────────────────────────────────
        current_state_ = Eigen::VectorXd::Zero(13);
        current_state_(6) = 1.0;   // upright quaternion

        // ── /mpc/state publisher ──────────────────────────────────────────────
        state_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/mpc/state", 10);

        // ── /mpc/command subscriber ───────────────────────────────────────────
        auto replay_cb_group = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions cmd_opts;
        cmd_opts.callback_group = replay_cb_group;

        cmd_sub_ = this->create_subscription<comando_planner::msg::MpcCommand>(
            "/mpc/command", 10,
            std::bind(&PlatformBridgeBase::onMpcCommand, this, std::placeholders::_1),
            cmd_opts);

        // ── NOTE: setupPlatformIO() is NOT called here.
        // Pure virtuals cannot be called from a base class constructor.
        // Each derived bridge must call setupPlatformIO() at the END of its
        // own constructor, after all members are fully initialized.

        RCLCPP_INFO(this->get_logger(),
            "[%s] Bridge ready. ocp_dt=%.3fs drone=%s logging=%s",
            node_name.c_str(), ocp_dt_, drone_name_.c_str(),
            logging_enabled_ ? "ON" : "OFF");
    }

protected:
    // ── Pure virtuals — every bridge must implement these ────────────────────

    /// Convert (x_cmd, u_cmd) and publish to the platform.
    /// Called by the replay timer at ocp_dt Hz.
    ///   x_cmd: 13-dim [pos, vel, quat(qw-first), omega]  ENU/FLU
    ///   u_cmd:  4-dim [fz_B (N), Mx, My, Mz (Nm, inertia-scaled)]
    virtual void sendPlatformCommand(const Eigen::VectorXd& x_cmd,
                                     const Eigen::VectorXd& u_cmd) = 0;

    /// Subscribe to platform state topics.
    /// Call publishMpcState(state_13d) in every state callback.
    virtual void setupPlatformIO() = 0;

    /// Called once each time a new MpcCommand is fully stored and the replay
    /// timer is about to start walking from replay_start_idx.
    /// Default is a no-op — crazyflie_bridge and px4_bridge are unaffected.
    /// Override in sim_bridge to implement x1_only mode.
    virtual void onNewTrajectory() {}

    // ── Helper: bridges call this in state callbacks ──────────────────────────
    void publishMpcState(const Eigen::VectorXd& state_13d)
    {
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            current_state_ = state_13d;
            state_received_ = true;
        }

        nav_msgs::msg::Odometry msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "world";
        msg.pose.pose.position.x    = state_13d(0);
        msg.pose.pose.position.y    = state_13d(1);
        msg.pose.pose.position.z    = state_13d(2);
        msg.twist.twist.linear.x    = state_13d(3);
        msg.twist.twist.linear.y    = state_13d(4);
        msg.twist.twist.linear.z    = state_13d(5);
        msg.pose.pose.orientation.w = state_13d(6);
        msg.pose.pose.orientation.x = state_13d(7);
        msg.pose.pose.orientation.y = state_13d(8);
        msg.pose.pose.orientation.z = state_13d(9);
        msg.twist.twist.angular.x   = state_13d(10);
        msg.twist.twist.angular.y   = state_13d(11);
        msg.twist.twist.angular.z   = state_13d(12);
        state_pub_->publish(msg);
    }

    // ── Accessible to derived bridges ─────────────────────────────────────────
    std::string    drone_name_;
    bool           logging_enabled_ = false;
    CommandoLogger logger_;

private:
    // ── /mpc/command callback — stores trajectory, (re)creates replay timer ──
    void onMpcCommand(const comando_planner::msg::MpcCommand::SharedPtr msg)
    {
        if (!msg->success) return;

        const int N  = msg->horizon;
        const int nx = msg->nx;
        const int nu = msg->nu;

        // Unpack flattened trajectories
        std::vector<Eigen::VectorXd> new_X(N + 1, Eigen::VectorXd::Zero(nx));
        std::vector<Eigen::VectorXd> new_U(N,     Eigen::VectorXd::Zero(nu));

        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < nx; ++i)
                new_X[k](i) = msg->state_trajectory[k * nx + i];
        }
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < nu; ++i)
                new_U[k](i) = msg->control_trajectory[k * nu + i];
        }

        {
            std::lock_guard<std::mutex> lk(traj_mutex_);

            // Save last commanded point for blending
            if (!mpc_traj_.empty()) {
                const int cur = std::min(replay_idx_, (int)mpc_traj_.size() - 1);
                blend_from_ = mpc_traj_[cur];
            } else if (!new_X.empty()) {
                blend_from_ = new_X[msg->replay_start_idx];
            }
            blend_steps_remaining_ = BLEND_STEPS;

            mpc_traj_   = new_X;
            mpc_ctrl_   = new_U;
            replay_idx_ = msg->replay_start_idx;
        }

        // Notify derived class — sim_bridge uses this for x1_only mode.
        // Called after traj_mutex_ is released to avoid deadlock.
        onNewTrajectory();

        // (Re)create replay timer if ocp_dt changed or first command
        const double new_dt = msg->ocp_dt;
        if (!replay_timer_ || std::abs(new_dt - ocp_dt_) > 1e-6) {
            ocp_dt_ = new_dt;
            const int period_ms = static_cast<int>(std::round(ocp_dt_ * 1000.0));
            auto replay_cb_group = this->create_callback_group(
                rclcpp::CallbackGroupType::MutuallyExclusive);
            replay_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(period_ms),
                std::bind(&PlatformBridgeBase::replayTick, this),
                replay_cb_group);
            RCLCPP_INFO(this->get_logger(),
                "[Bridge] Replay timer set: %d ms/step", period_ms);
        }
    }

    // ── Replay tick — runs at ocp_dt Hz ───────────────────────────────────────
    void replayTick()
    {
        Eigen::VectorXd x_cmd;
        Eigen::VectorXd u_cmd;
        bool have_traj = false;

        {
            std::lock_guard<std::mutex> lk(traj_mutex_);
            if (!mpc_traj_.empty()) {
                const int N   = static_cast<int>(mpc_traj_.size()) - 1;
                const int idx = std::min(replay_idx_, N);
                x_cmd     = mpc_traj_[idx];
                u_cmd     = (idx < (int)mpc_ctrl_.size())
                                ? mpc_ctrl_[idx]
                                : Eigen::VectorXd::Zero(4);
                have_traj = true;
                if (replay_idx_ < N) ++replay_idx_;
                // When replay_idx_ == N we hold: the if-guard above stops incrementing,
                // so x_cmd stays frozen at the last trajectory point until a new command.
            }
        }

        if (!have_traj) {
            // No trajectory yet — hold current state with zero velocity
            Eigen::VectorXd x_hover;
            { std::lock_guard<std::mutex> lk(state_mutex_); x_hover = current_state_; }
            x_hover.segment(3, 3).setZero();
            x_hover.segment(10, 3).setZero();
            sendPlatformCommand(x_hover, Eigen::VectorXd::Zero(4));
            return;
        }

        // ── Transition blend (full 13D state) ─────────────────────────────────
        // Linearly interpolates from the last commanded point to the new
        // trajectory over BLEND_STEPS ticks to hide discontinuities.
        // Quaternion uses lerp + renormalise (valid for small Δq).
        {
            std::lock_guard<std::mutex> lk(traj_mutex_);
            if (blend_steps_remaining_ > 0 && blend_from_.size() == x_cmd.size()) {
                const double alpha = 1.0 -
                    static_cast<double>(blend_steps_remaining_) / BLEND_STEPS;
                const double beta = 1.0 - alpha;

                Eigen::VectorXd blended = x_cmd;

                // pos + vel
                blended.head(6) = beta * blend_from_.head(6) + alpha * x_cmd.head(6);

                // quaternion — shortest path lerp + renormalise
                Eigen::Vector4d q0 = blend_from_.segment(6, 4);
                Eigen::Vector4d q1 = x_cmd.segment(6, 4);
                if (q0.dot(q1) < 0.0) q1 = -q1;
                blended.segment(6, 4) = (beta * q0 + alpha * q1).normalized();

                // angular rate
                blended.segment(10, 3) =
                    beta * blend_from_.segment(10, 3) + alpha * x_cmd.segment(10, 3);

                x_cmd = blended;
                --blend_steps_remaining_;
            }
        }

        sendPlatformCommand(x_cmd, u_cmd);

        // ── Logging ───────────────────────────────────────────────────────────
        if (logging_enabled_ && logger_.isInitialized()) {
            Eigen::VectorXd act;
            { std::lock_guard<std::mutex> lk(state_mutex_); act = current_state_; }
            logger_.logActualState(act);
            logger_.logCommandedState(x_cmd, u_cmd);
        }
    }

    // ── Members ───────────────────────────────────────────────────────────────
    double ocp_dt_ = 0.05;

    rclcpp::Subscription<comando_planner::msg::MpcCommand>::SharedPtr cmd_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr             state_pub_;
    rclcpp::TimerBase::SharedPtr                                      replay_timer_;

    std::mutex                   traj_mutex_;
    std::vector<Eigen::VectorXd> mpc_traj_;
    std::vector<Eigen::VectorXd> mpc_ctrl_;
    int                          replay_idx_ = 0;

    // Transition blend
    static constexpr int BLEND_STEPS = 12;   // ~600ms at 20Hz
    Eigen::VectorXd      blend_from_;
    int                  blend_steps_remaining_ = 0;

    // Current state (for hover fallback)
    std::mutex      state_mutex_;
    Eigen::VectorXd current_state_;
    bool            state_received_ = false;
};