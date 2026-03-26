#include "quadrotor_mpc.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <crazyflie_interfaces/msg/full_state.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

struct CircularTarget {
    Eigen::Vector3d center{0.0, 0.0, 1.5};
    double R = 2.0;
    double omega = 0.4;
    double phi0 = 0.0;

    Eigen::Vector3d pos(double t) const {
        const double ph = omega * t + phi0;
        return center + Eigen::Vector3d(R * std::cos(ph), R * std::sin(ph), 0.0);
    }
    Eigen::Vector3d vel(double t) const {
        const double ph = omega * t + phi0;
        return Eigen::Vector3d(-R * omega * std::sin(ph), R * omega * std::cos(ph), 0.0);
    }
    Eigen::Vector3d accel(double t) const {
        const double ph = omega * t + phi0;
        return Eigen::Vector3d(
            -R * omega * omega * std::cos(ph),
            -R * omega * omega * std::sin(ph),
            0.0);
    }
    Eigen::VectorXd state(double t) const {
        Eigen::VectorXd s(6);
        s.head(3) = pos(t);
        s.tail(3) = vel(t);
        return s;
    }
};

inline void normalizeQuat(Eigen::VectorXd& x13) {
    if (x13.size() < 10) {
        return;
    }
    Eigen::Vector4d q = x13.segment(6, 4);
    const double n = q.norm();
    if (n > 1e-12) {
        x13.segment(6, 4) = q / n;
    } else {
        x13(6) = 1.0;
        x13.segment(7, 3).setZero();
    }
}

inline Eigen::VectorXd interpRelativeState(const std::vector<Eigen::VectorXd>& X, double tau) {
    constexpr int IDX_DT = 13;
    if (X.empty()) {
        Eigen::VectorXd z = Eigen::VectorXd::Zero(13);
        z(6) = 1.0;
        return z;
    }
    if (X.front().size() <= IDX_DT) {
        Eigen::VectorXd x = X.front().head(13);
        normalizeQuat(x);
        return x;
    }
    if (tau <= X.front()(IDX_DT)) {
        Eigen::VectorXd x = X.front().head(13);
        normalizeQuat(x);
        return x;
    }
    if (tau >= X.back()(IDX_DT)) {
        Eigen::VectorXd x = X.back().head(13);
        normalizeQuat(x);
        return x;
    }

    int k1 = 1;
    while (k1 < static_cast<int>(X.size()) && X[k1](IDX_DT) < tau) {
        ++k1;
    }
    const int k0 = std::max(0, k1 - 1);
    k1 = std::min(k1, static_cast<int>(X.size()) - 1);

    const double t0 = X[k0](IDX_DT);
    const double t1 = X[k1](IDX_DT);
    const double dt = std::max(1e-9, t1 - t0);
    const double a = std::clamp((tau - t0) / dt, 0.0, 1.0);

    Eigen::VectorXd x = (1.0 - a) * X[k0].head(13) + a * X[k1].head(13);
    normalizeQuat(x);
    return x;
}

inline Eigen::VectorXd controlAtTau(const std::vector<Eigen::VectorXd>& U,
                                    const std::vector<Eigen::VectorXd>& X,
                                    double tau) {
    constexpr int IDX_DT = 13;
    Eigen::VectorXd u = Eigen::VectorXd::Zero(4);
    if (U.empty()) {
        return u;
    }
    int idx = 0;
    while (idx + 1 < static_cast<int>(X.size()) &&
           X[idx].size() > IDX_DT &&
           X[idx](IDX_DT) < tau) {
        ++idx;
    }
    idx = std::clamp(idx, 0, static_cast<int>(U.size()) - 1);
    if (U[idx].size() >= 4) {
        u = U[idx].head(4);
    }
    return u;
}

inline void publishTarget(
    const rclcpp::Node::SharedPtr& node,
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& odom_pub,
    const rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr& accel_pub,
    const CircularTarget& tgt,
    double t_abs) {
    const auto stamp = node->now();
    const Eigen::Vector3d p = tgt.pos(t_abs);
    const Eigen::Vector3d v = tgt.vel(t_abs);
    const Eigen::Vector3d a = tgt.accel(t_abs);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "world";
    odom.pose.pose.position.x = p.x();
    odom.pose.pose.position.y = p.y();
    odom.pose.pose.position.z = p.z();
    odom.twist.twist.linear.x = v.x();
    odom.twist.twist.linear.y = v.y();
    odom.twist.twist.linear.z = v.z();
    odom_pub->publish(odom);

    geometry_msgs::msg::AccelStamped acc;
    acc.header.stamp = stamp;
    acc.header.frame_id = "world";
    acc.accel.linear.x = a.x();
    acc.accel.linear.y = a.y();
    acc.accel.linear.z = a.z();
    accel_pub->publish(acc);
}

inline void publishDroneState(
    const rclcpp::Node::SharedPtr& node,
    const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr& pose_pub,
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& odom_pub,
    const Eigen::VectorXd& x_abs) {
    const auto stamp = node->now();

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = "world";
    pose.pose.position.x = x_abs(0);
    pose.pose.position.y = x_abs(1);
    pose.pose.position.z = x_abs(2);
    pose.pose.orientation.w = x_abs(6);
    pose.pose.orientation.x = x_abs(7);
    pose.pose.orientation.y = x_abs(8);
    pose.pose.orientation.z = x_abs(9);
    pose_pub->publish(pose);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "world";
    odom.pose.pose = pose.pose;
    odom.twist.twist.linear.x = x_abs(3);
    odom.twist.twist.linear.y = x_abs(4);
    odom.twist.twist.linear.z = x_abs(5);
    odom.twist.twist.angular.x = x_abs(10);
    odom.twist.twist.angular.y = x_abs(11);
    odom.twist.twist.angular.z = x_abs(12);
    odom_pub->publish(odom);
}

inline void publishCommand(
    const rclcpp::Node::SharedPtr& node,
    const rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr& cmd_pub,
    const Eigen::VectorXd& x_cmd,
    const Eigen::VectorXd& u_cmd) {
    crazyflie_interfaces::msg::FullState msg;
    msg.header.stamp = node->now();
    msg.header.frame_id = "world";
    msg.pose.position.x = x_cmd(0);
    msg.pose.position.y = x_cmd(1);
    msg.pose.position.z = x_cmd(2);
    msg.twist.linear.x = x_cmd(3);
    msg.twist.linear.y = x_cmd(4);
    msg.twist.linear.z = x_cmd(5);
    msg.pose.orientation.w = x_cmd(6);
    msg.pose.orientation.x = x_cmd(7);
    msg.pose.orientation.y = x_cmd(8);
    msg.pose.orientation.z = x_cmd(9);
    msg.twist.angular.x = x_cmd(10);
    msg.twist.angular.y = x_cmd(11);
    msg.twist.angular.z = x_cmd(12);
    if (u_cmd.size() >= 1) {
        msg.acc.z = static_cast<float>(u_cmd(0));
    }
    cmd_pub->publish(msg);
}

} // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("test_stateswitch_ocp");

    constexpr int N = 30;
    constexpr int NEX = 7;
    constexpr int NRH = 120;
    constexpr double TH0 = 0.1;
    constexpr double THL = 0.05;
    constexpr double THH = 0.2;
    constexpr double CAP = 0.15;
    constexpr double GS_DEG = 60.0;
    constexpr double SIM_DT = 0.01;
    const double GS_TAN = std::tan(GS_DEG * M_PI / 180.0);

    auto cf_pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("/cf_1/pose", 10);
    auto cf_odom_pub = node->create_publisher<nav_msgs::msg::Odometry>("/cf_1/odom", 10);
    auto tgt_odom_pub = node->create_publisher<nav_msgs::msg::Odometry>("/target/odom", 10);
    auto tgt_acc_pub = node->create_publisher<geometry_msgs::msg::AccelStamped>("/target/accel", 10);
    auto cmd_pub = node->create_publisher<crazyflie_interfaces::msg::FullState>("/cf_1/cmd_full_state", 10);

    Eigen::VectorXd loopback_cmd = Eigen::VectorXd::Zero(13);
    loopback_cmd(6) = 1.0;
    bool cmd_received = false;
    auto cmd_sub = node->create_subscription<crazyflie_interfaces::msg::FullState>(
        "/cf_1/cmd_full_state", 10,
        [&](const crazyflie_interfaces::msg::FullState::SharedPtr msg) {
            loopback_cmd(0) = msg->pose.position.x;
            loopback_cmd(1) = msg->pose.position.y;
            loopback_cmd(2) = msg->pose.position.z;
            loopback_cmd(3) = msg->twist.linear.x;
            loopback_cmd(4) = msg->twist.linear.y;
            loopback_cmd(5) = msg->twist.linear.z;
            loopback_cmd(6) = msg->pose.orientation.w;
            loopback_cmd(7) = msg->pose.orientation.x;
            loopback_cmd(8) = msg->pose.orientation.y;
            loopback_cmd(9) = msg->pose.orientation.z;
            loopback_cmd(10) = msg->twist.angular.x;
            loopback_cmd(11) = msg->twist.angular.y;
            loopback_cmd(12) = msg->twist.angular.z;
            normalizeQuat(loopback_cmd);
            cmd_received = true;
        });
    (void)cmd_sub;

    CircularTarget tgt;
    Eigen::VectorXd x_abs = Eigen::VectorXd::Zero(13);
    x_abs(2) = 5.0;
    x_abs(6) = 1.0;

    QuadrotorMPC::Config cfg;
    cfg.ocp_type = "stateswitch";
    cfg.n_shift = NEX;
    QuadrotorMPC mpc(cfg);

    double t_abs = 0.0;
    int steps = 0;
    int rh = 0;
    bool captured = false;
    bool solve_failed = false;

    std::vector<Eigen::VectorXd> X;
    std::vector<Eigen::VectorXd> U;
    double tau_plan = 0.0;
    double exec_window = 0.0;
    bool have_plan = false;

    double dbg_ms_sum = 0.0;
    double dbg_ms_max = 0.0;
    double dbg_pe_min = 1e9;
    double dbg_cone_max = 0.0;
    double dbg_d_min = 1e9;
    int dbg_nsolves = 0;
    int dbg_rh_d_min = -1;

    while (rclcpp::ok() && rh < NRH && !captured && !solve_failed) {
        rclcpp::spin_some(node);

        publishTarget(node, tgt_odom_pub, tgt_acc_pub, tgt, t_abs);
        publishDroneState(node, cf_pose_pub, cf_odom_pub, x_abs);

        const Eigen::VectorXd ts_now = tgt.state(t_abs);
        const double d0 = (x_abs.head(3) - ts_now.head(3)).norm();
        if (d0 < dbg_d_min) {
            dbg_d_min = d0;
            dbg_rh_d_min = rh;
        }

        if (!have_plan || tau_plan >= exec_window) {
            std::cout << "[RH " << std::setw(3) << rh << "]"
                      << "  t=" << std::setw(7) << std::fixed << std::setprecision(3) << t_abs
                      << "  q=(" << x_abs(0) << "," << x_abs(1) << "," << x_abs(2) << ")"
                      << "  tgt=(" << ts_now(0) << "," << ts_now(1) << "," << ts_now(2) << ")"
                      << "  d=" << d0 << "\n";

            Eigen::VectorXd x_rel = Eigen::VectorXd::Zero(13);
            x_rel.head(3) = x_abs.head(3) - tgt.pos(t_abs);
            x_rel.segment(3, 3) = x_abs.segment(3, 3) - tgt.vel(t_abs);
            x_rel.segment(6, 7) = x_abs.segment(6, 7);

            auto result = mpc.solve(x_rel, tgt.accel(t_abs));
            if (!result.success || result.state_trajectory.size() <= 1) {
                std::cerr << "[test_stateswitch_ocp] solve failed at RH " << rh << "\n";
                solve_failed = true;
                break;
            }

            X = result.state_trajectory;
            U = result.control_trajectory;
            tau_plan = 0.0;
            have_plan = true;

            exec_window = 0.0;
            for (int s = 0; s < NEX && s < N && s < static_cast<int>(U.size()); ++s) {
                double Th = U[s](4);
                Th = std::clamp(Th, THL, THH);
                exec_window += Th;
            }
            if (exec_window <= 1e-9) {
                exec_window = NEX * TH0;
            }

            dbg_ms_sum += result.solve_time_ms;
            dbg_ms_max = std::max(dbg_ms_max, result.solve_time_ms);
            ++dbg_nsolves;

            const double pe = X.back().head(3).norm();
            dbg_pe_min = std::min(dbg_pe_min, pe);

            for (int k = 0; k < N && k < static_cast<int>(X.size()); ++k) {
                const double dx = X[k](0);
                const double dy = X[k](1);
                const double dz = X[k](2);
                const double dxy = std::sqrt(dx * dx + dy * dy);
                const double cv = std::max(0.0, dxy - GS_TAN * dz);
                dbg_cone_max = std::max(dbg_cone_max, cv);
            }

            ++rh;
        }

        if (!have_plan) {
            break;
        }

        const Eigen::VectorXd x_rel_cmd = interpRelativeState(X, tau_plan);
        const Eigen::VectorXd u_cmd = controlAtTau(U, X, tau_plan);

        Eigen::VectorXd x_cmd_abs = Eigen::VectorXd::Zero(13);
        x_cmd_abs.head(3) = x_rel_cmd.head(3) + tgt.pos(t_abs);
        x_cmd_abs.segment(3, 3) = x_rel_cmd.segment(3, 3) + tgt.vel(t_abs);
        x_cmd_abs.segment(6, 7) = x_rel_cmd.segment(6, 7);
        normalizeQuat(x_cmd_abs);

        publishCommand(node, cmd_pub, x_cmd_abs, u_cmd);
        rclcpp::spin_some(node);

        if (cmd_received) {
            x_abs = loopback_cmd;
            cmd_received = false;
        } else {
            x_abs = x_cmd_abs;
        }

        const double d = (x_abs.head(3) - tgt.pos(t_abs)).norm();
        if (d < dbg_d_min) {
            dbg_d_min = d;
            dbg_rh_d_min = std::max(0, rh - 1);
        }
        if (d < CAP) {
            std::cout << "  LANDED at t=" << t_abs << "  d=" << d << "\n";
            captured = true;
        }

        t_abs += SIM_DT;
        tau_plan += SIM_DT;
        ++steps;
    }

    const Eigen::VectorXd ts_fin = tgt.state(t_abs);
    const double d_fin = (x_abs.head(3) - ts_fin.head(3)).norm();

    std::cout << "\n=== Summary ===\n" << std::fixed << std::setprecision(4);
    std::cout << "Result:   " << (captured ? "LANDED" : "FAILED") << "\n";
    std::cout << "Elapsed:  " << t_abs << " s    Steps: " << steps << "\n";
    std::cout << "Drone:    " << x_abs.head(3).transpose() << "\n";
    std::cout << "Target:   " << ts_fin.head(3).transpose() << "\n";
    std::cout << "Distance: " << d_fin << " m\n";
    std::cout << "\n--- Solver ---\n";
    std::cout << "Solves: " << dbg_nsolves
              << "  avg_ms=" << (dbg_nsolves > 0 ? dbg_ms_sum / dbg_nsolves : 0.0)
              << "  max_ms=" << dbg_ms_max << "\n";
    std::cout << "\n--- Glideslope SOC ---\n";
    std::cout << "cone_max ever: " << dbg_cone_max << " m\n";
    std::cout << "\n--- Approach ---\n";
    std::cout << "pos_err min: " << dbg_pe_min << " m\n";
    std::cout << "d_min:       " << dbg_d_min << " m  at RH " << dbg_rh_d_min << "\n";
    std::cout << "d_final:     " << d_fin << " m\n";

    rclcpp::shutdown();
    return (captured && !solve_failed) ? 0 : 1;
}
