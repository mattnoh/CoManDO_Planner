#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <chrono>
#include <string>
#include <any>

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "ocp_registry.hpp"

class QuadrotorMPC {
public:
    struct Config {
        std::string ocp_type = "hover";

        // Planner-provided terminal state for OCPs that use it.
        Eigen::VectorXd terminal_state;

        // Warm-start shift count (PlannerNode sets this from replay settings).
        int n_shift = 1;

        Config() {
            terminal_state = Eigen::VectorXd::Zero(13);
            terminal_state(2) = 1.0;
            terminal_state(6) = 1.0;
        }
    };

    struct Result {
        bool                          success       = false;
        Eigen::VectorXd               next_state;
        std::vector<Eigen::VectorXd>  state_trajectory;
        std::vector<Eigen::VectorXd>  control_trajectory;
        std::vector<Eigen::MatrixXd>  feedback_gains;
        double                        solve_time_ms = 0.0;
        int                           solve_iters   = 0;
        std::chrono::steady_clock::time_point solve_timestamp;
        std::any                      extra_params;
    };

    explicit QuadrotorMPC(const Config& config = Config());
    ~QuadrotorMPC() = default;

    Result solve(const Eigen::VectorXd& current_state,
                 const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero(),
                 const std::any& extra_params = {},
                 double t_abs = 0.0);

    void setTerminalState(const Eigen::VectorXd& terminal);

    double getOcpDt() const;
    double last_solve_ms_ = 0.0;

private:
    void setupProblem(const OCPCreateArgs& args);

    std::vector<Eigen::VectorXd> makeUwarm(int n_shift) const;
    std::vector<Eigen::VectorXd> makeXshifted(int n_shift) const;
    std::vector<Eigen::MatrixXd> makeKshifted(int n_shift) const;

    Config                                         config_;
    std::shared_ptr<OptimalControlProblem<double>> problem_;
    std::shared_ptr<ALIPDDP<double>>               solver_;
    Param                                          solver_params_;

    std::vector<Eigen::VectorXd> prev_X_;
    std::vector<Eigen::VectorXd> prev_U_;
    std::vector<Eigen::MatrixXd> prev_K_;
    Eigen::Vector3d              target_accel_ = Eigen::Vector3d::Zero();
};
