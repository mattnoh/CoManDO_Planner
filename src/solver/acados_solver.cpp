/// @file acados_solver.cpp
/// @brief Acados SQP_RTI MPC implementation — wraps generated C solver code.

#include "solver/acados_solver.hpp"
#include "utils/logger.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>

extern "C" {
#include "acados_solver_quadrotor.h"
#include "acados_c/ocp_nlp_interface.h"
}

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
AcadosMPC::AcadosMPC(const Config& config) : config_(config) {
    initSolver();
}

AcadosMPC::~AcadosMPC() {
    if (capsule_) {
        quadrotor_acados_free(capsule_);
        quadrotor_acados_free_capsule(capsule_);
        capsule_ = nullptr;
    }
}

void AcadosMPC::initSolver() {
    capsule_ = quadrotor_acados_create_capsule();
    if (!capsule_)
        throw runtime_error("AcadosMPC: failed to create solver capsule");

    int status = quadrotor_acados_create(capsule_);
    if (status != 0) {
        quadrotor_acados_free_capsule(capsule_);
        capsule_ = nullptr;
        throw runtime_error("AcadosMPC: quadrotor_acados_create() failed, status="
                            + to_string(status));
    }
    initialized_ = true;
    setReference();

    COMANDO_INFO("AcadosMPC", "Solver initialized: N=%d nx=%d nu=%d nh=%d",
                 QUADROTOR_N, QUADROTOR_NX, QUADROTOR_NU, QUADROTOR_NH);
}

// ─────────────────────────────────────────────────────────────────────────────
void AcadosMPC::setInitialState(const Eigen::VectorXd& x0) {
    if (!capsule_ || !initialized_) return;
    ocp_nlp_in*     nlp_in = quadrotor_acados_get_nlp_in(capsule_);
    ocp_nlp_config* cfg    = quadrotor_acados_get_nlp_config(capsule_);
    ocp_nlp_dims*   dims   = quadrotor_acados_get_nlp_dims(capsule_);

    double x0_data[QUADROTOR_NX];
    for (int i = 0; i < QUADROTOR_NX; ++i) x0_data[i] = x0(i);
    ocp_nlp_constraints_model_set(cfg, dims, nlp_in, 0, "lbx", x0_data);
    ocp_nlp_constraints_model_set(cfg, dims, nlp_in, 0, "ubx", x0_data);
}

// ─────────────────────────────────────────────────────────────────────────────
void AcadosMPC::setReference() {
    if (!capsule_ || !initialized_) return;
    const int N = QUADROTOR_N;

    Eigen::VectorXd x_ref = Eigen::VectorXd::Zero(13);
    x_ref(6) = 1.0;
    if (config_.terminal_state.size() == 13) x_ref = config_.terminal_state;

    Eigen::VectorXd u_ref = Eigen::VectorXd::Zero(6);
    u_ref(2) = config_.mass * 9.81;

    double y_ref[QUADROTOR_NY];
    for (int i = 0; i < 13; ++i) y_ref[i]      = x_ref(i);
    for (int i = 0; i < 6;  ++i) y_ref[13 + i] = u_ref(i);

    ocp_nlp_in*     nlp_in = quadrotor_acados_get_nlp_in(capsule_);
    ocp_nlp_config* cfg    = quadrotor_acados_get_nlp_config(capsule_);
    ocp_nlp_dims*   dims   = quadrotor_acados_get_nlp_dims(capsule_);

    for (int k = 0; k < N; ++k)
        ocp_nlp_cost_model_set(cfg, dims, nlp_in, k, "yref", y_ref);

    double y_ref_e[QUADROTOR_NYN];
    for (int i = 0; i < 13; ++i) y_ref_e[i] = x_ref(i);
    ocp_nlp_cost_model_set(cfg, dims, nlp_in, N, "yref", y_ref_e);
}

// ─────────────────────────────────────────────────────────────────────────────
void AcadosMPC::shiftWarmStart() {
    if (!has_prev_solution_ || prev_U_.empty()) return;
    const int N = QUADROTOR_N;
    const int n_shift = max(1, min(config_.n_shift, N - 1));

    ocp_nlp_out*    nlp_out = quadrotor_acados_get_nlp_out(capsule_);
    ocp_nlp_config* cfg     = quadrotor_acados_get_nlp_config(capsule_);
    ocp_nlp_dims*   dims    = quadrotor_acados_get_nlp_dims(capsule_);

    for (int k = 0; k <= N; ++k) {
        int src = min(k + n_shift, N);
        double x_data[QUADROTOR_NX];
        for (int i = 0; i < QUADROTOR_NX; ++i) x_data[i] = prev_X_[src](i);
        ocp_nlp_out_set(cfg, dims, nlp_out, k, "x", x_data);
    }
    for (int k = 0; k < N; ++k) {
        int src = min(k + n_shift, N - 1);
        double u_data[QUADROTOR_NU];
        for (int i = 0; i < QUADROTOR_NU; ++i) u_data[i] = prev_U_[src](i);
        ocp_nlp_out_set(cfg, dims, nlp_out, k, "u", u_data);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
AcadosMPC::Result AcadosMPC::solve(const Eigen::VectorXd& current_state) {
    Result result;
    result.success = false;
    auto t0 = chrono::high_resolution_clock::now();

    if (!capsule_ || !initialized_) {
        COMANDO_ERROR("AcadosMPC", "Solver not initialized");
        return result;
    }

    try {
        const int N = QUADROTOR_N;

        if (has_prev_solution_) shiftWarmStart();
        setInitialState(current_state);

        int status = 0;
        for (int iter = 0; iter < config_.rti_iterations; ++iter)
            status = quadrotor_acados_solve(capsule_);

        result.solve_time_ms = chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
        result.solve_timestamp = chrono::steady_clock::now();

        if (status != 0 && status != 2) {
            COMANDO_ERROR("AcadosMPC", "Solver failed, status=%d", status);
            return result;
        }

        ocp_nlp_out*    nlp_out = quadrotor_acados_get_nlp_out(capsule_);
        ocp_nlp_config* cfg     = quadrotor_acados_get_nlp_config(capsule_);
        ocp_nlp_dims*   dims    = quadrotor_acados_get_nlp_dims(capsule_);

        result.state_trajectory.resize(N + 1);
        result.control_trajectory.resize(N);

        for (int k = 0; k <= N; ++k) {
            result.state_trajectory[k].resize(QUADROTOR_NX);
            double x_data[QUADROTOR_NX];
            ocp_nlp_out_get(cfg, dims, nlp_out, k, "x", x_data);
            for (int i = 0; i < QUADROTOR_NX; ++i)
                result.state_trajectory[k](i) = x_data[i];
        }
        for (int k = 0; k < N; ++k) {
            result.control_trajectory[k].resize(QUADROTOR_NU);
            double u_data[QUADROTOR_NU];
            ocp_nlp_out_get(cfg, dims, nlp_out, k, "u", u_data);
            for (int i = 0; i < QUADROTOR_NU; ++i)
                result.control_trajectory[k](i) = u_data[i];
        }

        result.next_state = result.state_trajectory[1];
        result.success    = true;

        prev_X_ = result.state_trajectory;
        prev_U_ = result.control_trajectory;
        has_prev_solution_ = true;

    } catch (const exception& e) {
        COMANDO_ERROR("AcadosMPC", "Exception: %s", e.what());
        has_prev_solution_ = false;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
void AcadosMPC::setTerminalState(const Eigen::VectorXd& terminal) {
    if (terminal.size() == 13) {
        config_.terminal_state = terminal;
        has_prev_solution_ = false;
        setReference();
    }
}