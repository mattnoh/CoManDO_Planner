#include "quadrotor_mpc.hpp"
#include "ocp_hover.hpp"
#include "ocp_constrained_attitude.hpp"
#include <iostream>

using namespace std;

QuadrotorMPC::QuadrotorMPC(const Config& config) : config_(config) {
    solver_params_.reg1_min = 1e-6;
    solver_params_.reg2_min = 1.0;
    solver_params_.mu_mul = 0.1;
    solver_params_.rho = 20.0;
    solver_params_.rho_mul = 9.0;
    solver_params_.max_iter = 10;
    solver_params_.tolerance = 1e-3;
}

void QuadrotorMPC::setupProblem(const Eigen::VectorXd& current_state) {
    if (config_.ocp_type == "hover") {
        problem_ = HoverOCP::create(current_state);
    }
    else if (config_.ocp_type == "constrained_attitude") {
        problem_ = ConstrainedAttitudeOCP::create(current_state);
    }
    else {
        cerr << "ERROR: Unknown OCP type: " << config_.ocp_type << endl;
        throw runtime_error("Unknown OCP type");
    }
}

QuadrotorMPC::Result QuadrotorMPC::solve(const Eigen::VectorXd& current_state) {
    Result result;
    result.success = false;
    auto start_time = chrono::high_resolution_clock::now();
    
    try {
        setupProblem(current_state);
        
        if (!problem_) {
            cerr << "ERROR: Problem not initialized" << endl;
            return result;
        }
        
        solver_.reset();
        solver_ = std::make_shared<ALIPDDP<double>>(*problem_);
        solver_->init(solver_params_);
        
        solver_->solve();
        
        std::vector<Eigen::VectorXd> X_result = solver_->getResX();
        auto U_result = solver_->getResU();
        
        auto end_time = chrono::high_resolution_clock::now();
        result.solve_time_ms = chrono::duration<double, milli>(end_time - start_time).count();
        
        if (X_result.size() > 1) {
            result.next_state = X_result[1];
            result.state_trajectory = X_result;
            result.control_trajectory = U_result;
            result.success = true;

            int states_to_print = std::min(4, (int)X_result.size());
            for (int i = 0; i < states_to_print; ++i) {
                cout << "X[" << i << "]: " << X_result[i].transpose() << endl;
            }
            
        } else {
            cerr << "ERROR: Empty trajectory from solver" << endl;
        }
        
    } catch (const exception& e) {
        cerr << "ERROR in solve: " << e.what() << endl;
    }
    
    return result;
}

void QuadrotorMPC::setTerminalState(const Eigen::VectorXd& terminal) {
    if (terminal.size() == 13) {
        config_.terminal_state = terminal;
    }
}