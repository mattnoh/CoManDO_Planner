/// @file acados_solver.hpp
/// @brief Acados SQP_RTI-backed MPC solver.  Drop-in alternative to QuadrotorMPC.
///
/// Prerequisites:
///   1. Run scripts/generate_acados_ocp.py  →  acados_generated/
///   2. acados installed at $ACADOS_INSTALL_DIR (default ~/acados)

#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

extern "C" { struct quadrotor_solver_capsule; }

class AcadosMPC {
public:
    struct Config {
        std::string ocp_type = "landing";

        double mass    = 0.027;
        int    horizon = 100;
        double dt      = 0.1;

        Eigen::VectorXd terminal_state;

        int rti_iterations = 1;
        int n_shift        = 1;

        Config() {
            terminal_state = Eigen::VectorXd::Zero(13);
            terminal_state(2) = 1.0;
            terminal_state(6) = 1.0;
        }
    };

    // Same Result layout as QuadrotorMPC for seamless switching
    struct Result {
        bool                         success   = false;
        Eigen::VectorXd              next_state;
        std::vector<Eigen::VectorXd> state_trajectory;
        std::vector<Eigen::VectorXd> control_trajectory;
        double                       solve_time_ms = 0.0;
        std::chrono::steady_clock::time_point solve_timestamp;
    };

    explicit AcadosMPC(const Config& config = Config());
    ~AcadosMPC();

    AcadosMPC(const AcadosMPC&) = delete;
    AcadosMPC& operator=(const AcadosMPC&) = delete;

    Result solve(const Eigen::VectorXd& current_state);
    void   setTerminalState(const Eigen::VectorXd& terminal);
    double getOcpDt() const { return config_.dt; }

private:
    void initSolver();
    void shiftWarmStart();
    void setInitialState(const Eigen::VectorXd& x0);
    void setReference();

    Config                       config_;
    quadrotor_solver_capsule*    capsule_     = nullptr;
    bool                         initialized_ = false;

    std::vector<Eigen::VectorXd> prev_X_;
    std::vector<Eigen::VectorXd> prev_U_;
    bool                         has_prev_solution_ = false;
};