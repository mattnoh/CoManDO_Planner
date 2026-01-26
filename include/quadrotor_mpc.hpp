#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <chrono>

// ALIPDDP headers
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

class QuadrotorMPC {
public:
    struct Config {
        int horizon = 50;
        double dt = 0.02;  // 50 Hz
        double mass = 0.027;
        Eigen::Matrix3d inertia = (Eigen::Matrix3d() << 
            1.66e-5, 0.0, 0.0,
            0.0, 1.66e-5, 0.0,
            0.0, 0.0, 2.92e-5).finished();
        
        // Terminal state (13D) - set to whatever you want
        Eigen::VectorXd terminal_state;
        
        // Cost weights
        double attitude_weight = 2.0;
        double thrust_weight = 1e-5;
        double moment_weight = 1e-4;
        
        // Constraint
        double max_thrust = 0.6;
        
        Config() {
            terminal_state = Eigen::VectorXd::Zero(13);
            terminal_state(2) = 1.0;  // z = 1m
            terminal_state(6) = 1.0;  // upright quaternion
        }
    };
    
    struct Result {
        bool success = false;
        Eigen::VectorXd next_state;
        std::vector<Eigen::VectorXd> state_trajectory;
        std::vector<Eigen::VectorXd> control_trajectory;
        double solve_time_ms = 0.0;
    };
    
    QuadrotorMPC(const Config& config = Config());
    ~QuadrotorMPC();
    
    Result solve(const Eigen::VectorXd& current_state);
    
    // Update terminal state if needed
    void setTerminalState(const Eigen::VectorXd& terminal);
    
private:
    void setupProblem(const Eigen::VectorXd& current_state);
    Config config_;
    std::shared_ptr<OptimalControlProblem<double>> problem_;
    std::shared_ptr<ALIPDDP<double>> solver_;
    Param solver_params_;
    
    // For warm start
    std::vector<Eigen::VectorXd> prev_X_;
    std::vector<Eigen::VectorXd> prev_U_;
    bool first_solve_ = true;
    bool problem_initialized_ = false;
};