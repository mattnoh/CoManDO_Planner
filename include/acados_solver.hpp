/// @file acados_solver.hpp
/// @brief Acados-based MPC solver for quadrotor 6-DOF OCP.
///        Drop-in alternative to QuadrotorMPC (ALIPDDP) — same Result struct.
///
/// Prerequisites:
///   1. Run scripts/generate_acados_ocp.py to produce acados_generated/
///   2. acados libraries installed at $ACADOS_INSTALL_DIR (default ~/acados)

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <chrono>
#include <string>

// Forward-declare the acados capsule (C struct)
extern "C" {
    struct quadrotor_solver_capsule;
}

class AcadosMPC {
public:
    struct Config {
        std::string ocp_type = "landing";   // "hover" or "landing"

        double mass     = 0.027;            // kg
        int    horizon  = 100;
        double dt       = 0.1;              // seconds per OCP step

        Eigen::VectorXd terminal_state;     // 13-dim reference (target)

        // RTI iterations per solve call — 1 = pure SQP_RTI,
        // >1 = multiple RTI iterations for better convergence (at cost of time)
        int rti_iterations = 1;

        // How many OCP steps to shift the warm-start forward per solve
        int n_shift = 1;

        Config() {
            terminal_state = Eigen::VectorXd::Zero(13);
            terminal_state(2) = 1.0;   // hover at z=1m
            terminal_state(6) = 1.0;   // upright quaternion
        }
    };

    // Same Result layout as QuadrotorMPC for seamless switching
    struct Result {
        bool                          success   = false;
        Eigen::VectorXd               next_state;
        std::vector<Eigen::VectorXd>  state_trajectory;
        std::vector<Eigen::VectorXd>  control_trajectory;
        double                        solve_time_ms = 0.0;
        std::chrono::steady_clock::time_point solve_timestamp;
    };

    explicit AcadosMPC(const Config& config = Config());
    ~AcadosMPC();

    // Non-copyable
    AcadosMPC(const AcadosMPC&) = delete;
    AcadosMPC& operator=(const AcadosMPC&) = delete;

    /// Solve the OCP from current_state.
    /// The warm-start is handled internally by acados (SQP_RTI retains
    /// primal-dual iterate across calls).
    Result solve(const Eigen::VectorXd& current_state);

    /// Update the terminal reference state
    void setTerminalState(const Eigen::VectorXd& terminal);

    /// OCP time step (seconds)
    double getOcpDt() const { return config_.dt; }

private:
    void initSolver();
    void shiftWarmStart();
    void setInitialState(const Eigen::VectorXd& x0);
    void setReference();

    Config                         config_;
    quadrotor_solver_capsule*      capsule_ = nullptr;
    bool                           initialized_ = false;

    // Previous solution for warm-start shift
    std::vector<Eigen::VectorXd>   prev_X_;
    std::vector<Eigen::VectorXd>   prev_U_;
    bool                           has_prev_solution_ = false;
};
