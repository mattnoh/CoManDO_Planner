#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <chrono>
#include <string>

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

class QuadrotorMPC {
public:
    struct Config {
        std::string ocp_type = "hover";

        int              horizon   = 30;
        double           dt        = 0.02;
        double           mass      = 0.027;
        Eigen::Matrix3d  inertia   = (Eigen::Matrix3d() <<
            1.66e-5, 0.0, 0.0,
            0.0, 1.66e-5, 0.0,
            0.0, 0.0, 2.92e-5).finished();

        Eigen::VectorXd  terminal_state;
        double           max_thrust = 0.6;

        // n_shift: how many U steps to shift the warm-start on each solve.
        // Set once by PlannerNode to n_replay_.
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
    };

    explicit QuadrotorMPC(const Config& config = Config());
    ~QuadrotorMPC() = default;

    Result solve(const Eigen::VectorXd& current_state,
                 const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero());

    void   setTerminalState(const Eigen::VectorXd& terminal);
    double getOcpDt() const;
    double last_solve_ms_ = 0.0;

private:
    void setupProblem(const Eigen::VectorXd& current_state,
                      const std::vector<Eigen::VectorXd>& warm_u,
                      const std::vector<Eigen::VectorXd>& warm_x,
                      const std::vector<Eigen::MatrixXd>& warm_k);

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
