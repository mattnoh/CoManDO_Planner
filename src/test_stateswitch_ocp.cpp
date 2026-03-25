#include "quadrotor_mpc.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

struct CircularTarget {
    Eigen::Vector3d center{0.0, 0.0, 1.5};
    double R = 2.0;
    double omega = 0.4;
    double phi0 = 0.0;
    Eigen::Vector3d pos(double t) const {
        double ph = omega * t + phi0;
        return center + Eigen::Vector3d(R * std::cos(ph), R * std::sin(ph), 0.0);
    }
    Eigen::Vector3d vel(double t) const {
        double ph = omega * t + phi0;
        return Eigen::Vector3d(-R * omega * std::sin(ph), R * omega * std::cos(ph), 0.0);
    }
    Eigen::Vector3d accel(double t) const {
        double ph = omega * t + phi0;
        return Eigen::Vector3d(-R * omega * omega * std::cos(ph),
                               -R * omega * omega * std::sin(ph), 0.0);
    }
    Eigen::VectorXd state(double t) const {
        Eigen::VectorXd s(6);
        s.head(3) = pos(t);
        s.tail(3) = vel(t);
        return s;
    }
};

int main() {
    constexpr int N = 30;
    constexpr int NEX = 7;
    constexpr int NRH = 120;
    constexpr double TH0 = 0.1;
    constexpr double THL = 0.05;
    constexpr double THH = 0.2;
    constexpr double CAP = 0.15;
    constexpr double GS_DEG = 60.0;
    const double GS_TAN = std::tan(GS_DEG * M_PI / 180.0);

    CircularTarget tgt;

    Eigen::VectorXd x_abs(13);
    x_abs.setZero();
    x_abs(0) = 0.0;
    x_abs(1) = 0.0;
    x_abs(2) = 5.0;
    x_abs(6) = 1.0;

    QuadrotorMPC::Config cfg;
    cfg.ocp_type = "stateswitch";
    cfg.n_shift = NEX;

    QuadrotorMPC mpc(cfg);

    double t_abs = 0.0;
    int steps = 0;
    bool captured = false;

    double dbg_ms_sum = 0.0;
    double dbg_ms_max = 0.0;
    double dbg_pe_min = 1e9;
    double dbg_cone_max = 0.0;
    double dbg_d_min = 1e9;
    int dbg_nsolves = 0;
    int dbg_rh_d_min = -1;

    for (int rh = 0; rh < NRH && captured == false; ++rh) {
        Eigen::VectorXd ts_now = tgt.state(t_abs);
        double d0 = (x_abs.head(3) - ts_now.head(3)).norm();
        if (d0 < dbg_d_min) {
            dbg_d_min = d0;
            dbg_rh_d_min = rh;
        }

        std::cout << "[RH " << std::setw(3) << rh << "]"
                  << "  t=" << std::setw(7) << std::fixed << std::setprecision(3) << t_abs
                  << "  q=(" << x_abs(0) << "," << x_abs(1) << "," << x_abs(2) << ")"
                  << "  tgt=(" << ts_now(0) << "," << ts_now(1) << "," << ts_now(2) << ")"
                  << "  d=" << d0 << "\n";

        Eigen::VectorXd x_rel(13);
        x_rel.head(3)      = x_abs.head(3)      - tgt.pos(t_abs);
        x_rel.segment(3,3) = x_abs.segment(3,3) - tgt.vel(t_abs);
        x_rel.segment(6,7) = x_abs.segment(6,7);

        auto result = mpc.solve(x_rel, tgt.accel(t_abs));
        if (result.success == false || result.state_trajectory.size() <= 1) {
            std::cerr << "[test_stateswitch_ocp] solve failed at RH " << rh << "\n";
            return 1;
        }

        const auto& X = result.state_trajectory;
        const auto& U = result.control_trajectory;

        dbg_ms_sum += result.solve_time_ms;
        if (result.solve_time_ms > dbg_ms_max) dbg_ms_max = result.solve_time_ms;
        ++dbg_nsolves;

        double pe = X.back().head(3).norm();
        if (pe < dbg_pe_min) dbg_pe_min = pe;

        for (int k = 0; k < N && k < static_cast<int>(X.size()); ++k) {
            double dx = X[k](0);
            double dy = X[k](1);
            double dz = X[k](2);
            double dxy = std::sqrt(dx * dx + dy * dy);
            double cv = std::max(0.0, dxy - GS_TAN * dz);
            if (cv > dbg_cone_max) dbg_cone_max = cv;
        }

        for (int s = 0; s < NEX && s < N && captured == false; ++s) {
            double Th = U[s](4);
            if (Th < THL) Th = THL;
            if (Th > THH) Th = THH;

            t_abs += Th;
            ++steps;

            x_abs.head(3)      = X[s + 1].head(3)      + tgt.pos(t_abs);
            x_abs.segment(3,3) = X[s + 1].segment(3,3) + tgt.vel(t_abs);
            x_abs.segment(6,7) = X[s + 1].segment(6,7);

            double d = (x_abs.head(3) - tgt.pos(t_abs)).norm();
            if (d < dbg_d_min) {
                dbg_d_min = d;
                dbg_rh_d_min = rh;
            }
            if (d < CAP) {
                std::cout << "  LANDED at t=" << t_abs << "  d=" << d << "\n";
                captured = true;
            }
        }
    }

    Eigen::VectorXd ts_fin = tgt.state(t_abs);
    double d_fin = (x_abs.head(3) - ts_fin.head(3)).norm();

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

    return captured ? 0 : 1;
}
