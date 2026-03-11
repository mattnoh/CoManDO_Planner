/// @file alipddp_solver.hpp
/// @brief ALIPDDP-backed MPC solver for the quadrotor 6-DOF OCP.
#pragma once
#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
class ALIPDDPSolver {
public:
    struct Config {
        std::string     ocp_type  = "hover";
        int             horizon   = 30;
        double          dt        = 0.02;
        double          mass      = 0.027;
        Eigen::Matrix3d inertia   = (Eigen::Matrix3d() <<
            1.66e-5, 0.0,     0.0,
            0.0,     1.66e-5, 0.0,
            0.0,     0.0,     2.92e-5).finished();
        Eigen::VectorXd terminal_state;
        double          max_thrust  = 0.6;
        int             n_shift     = 1;
        int             sim_n_shift = 0;  // 0=hardware timing, 1=match standalone N_SHIFT=1
        Config() {
            terminal_state = Eigen::VectorXd::Zero(13);
            terminal_state(2) = 1.0;
            terminal_state(6) = 1.0;
        }
    };

    struct Result {
        bool                         success       = false;
        Eigen::VectorXd              next_state;
        std::vector<Eigen::VectorXd> state_trajectory;
        std::vector<Eigen::VectorXd> control_trajectory;
        double                       solve_time_ms = 0.0;
        std::chrono::steady_clock::time_point solve_timestamp;
    };

    explicit ALIPDDPSolver(const Config& config = Config());
    ~ALIPDDPSolver() = default;

    Result solve(const Eigen::VectorXd& current_state);
    void   setTerminalState(const Eigen::VectorXd& terminal);
    double getOcpDt() const;

private:
    void setupProblem(const Eigen::VectorXd& current_state);
    void shiftWarmStart();

    Config                                         config_;
    std::shared_ptr<OptimalControlProblem<double>> problem_;
    std::shared_ptr<ALIPDDP<double>>               solver_;
    Param                                          solver_params_;
    std::vector<Eigen::VectorXd>                   prev_X_;
    std::vector<Eigen::VectorXd>                   prev_U_;
    bool                                           has_prev_solution_   = false;
    bool                                           need_problem_rebuild_ = true;
};