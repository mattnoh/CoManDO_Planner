/// @file ocp_tracking_bodyrate_tf_imu.hpp
/// @brief Target-frame body-rate tracking OCP with augmented IMU states (19D).
///
/// State (19D physical + DT = 20D):
///   x[0:3]   — p_B^N  : drone position in target frame N
///   x[3:6]   — v_B^N  : drone velocity in target frame N
///   x[6:10]  — q_NB   : quaternion body→target [qw,qx,qy,qz]
///   x[10:13] — Ω_N    : target angular velocity in N (augmented)
///   x[13:16] — â_N    : target linear accel in target frame (augmented)
///   x[16:19] — β_N    : target angular accel in N (augmented)
///   x[19]    — DT     : cumulative time (variable timestep)
///
/// Control (5D): u[0]=T [m/s²], u[1:4]=ω_B [rad/s], u[4]=Theta [s]
/// DroneOdomMode: TargetFrameRelative
#pragma once

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "dynamics/quad_6dof_target_frame.h"
#include "core/ocp_descriptor.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace TrackingBodyrateTfImuOCP {

static constexpr int NX    = 19;
static constexpr int NU    = 4;
static constexpr int NX_SS = 20;
static constexpr int NU_SS = 5;
static constexpr int IDX_P   = 0;
static constexpr int IDX_V   = 3;
static constexpr int IDX_Q   = 6;
static constexpr int IDX_OMN = 10;
static constexpr int IDX_AN  = 13;
static constexpr int IDX_BN  = 16;
static constexpr int IDX_DT  = 19;
static constexpr int IDX_THETA = 4;

static constexpr int HORIZON = 30;
static constexpr double TH_INIT = 0.1;
static constexpr double THL = 0.05;
static constexpr double THH = 0.2;
static constexpr int DEFAULT_N_REPLAY = 7;

static constexpr double DEFAULT_MASS_KG = 0.027;
static constexpr double FMIN = 9.81 * 0.3;
static constexpr double FMAX = 9.81 * 3.0;
static constexpr double OMEGA_MAX = 5.0;
static constexpr double GS_TAN = 1.0;
static constexpr double VZ_LAND_MAX = 0.5;

static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

template<typename Scalar>
class TimeCost : public StageCostBase<Scalar> {
    Scalar eps_;
public:
    explicit TimeCost(double eps = 1e-2) : eps_(eps) {}
    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return u(IDX_THETA) + eps_ * x.segment(IDX_V, 3).squaredNorm();
    }
    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NX_SS);
        g.segment(IDX_V, 3) = Scalar(2) * eps_ * x.segment(IDX_V, 3);
        return g;
    }
    Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NU_SS);
        g(IDX_THETA) = Scalar(1.0);
        return g;
    }
    Matrix<Scalar> qxx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        for (int i = IDX_V; i < IDX_V + 3; ++i) H(i, i) = Scalar(2) * eps_;
        return H;
    }
    Matrix<Scalar> quu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(NU_SS, NU_SS);
    }
    Matrix<Scalar> qxu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(NX_SS, NU_SS);
    }
};

template<typename Scalar>
class TermCost : public TerminalCostBase<Scalar> {
    double wp_, wv_, wvz_, watt_, vz_ref_;
public:
    explicit TermCost(double wp = 500.0, double wv = 50.0,
                      double wvz = 80.0, double watt = 10.0,
                      double vz_ref = -0.25)
        : wp_(wp), wv_(wv), wvz_(wvz), watt_(watt), vz_ref_(vz_ref) {}
    Scalar p(const Vector<Scalar>& x) const override {
        const Scalar ep   = x.template segment<3>(IDX_P).squaredNorm();
        const Scalar evxy = x.template segment<2>(IDX_V).squaredNorm();
        const Scalar evz  = (x(IDX_V + 2) - Scalar(vz_ref_)) * (x(IDX_V + 2) - Scalar(vz_ref_));
        const Scalar eatt = x(IDX_Q + 1) * x(IDX_Q + 1) + x(IDX_Q + 2) * x(IDX_Q + 2);
        return Scalar(wp_) * ep + Scalar(wv_) * evxy + Scalar(wvz_) * evz + Scalar(watt_) * eatt;
    }
    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.segment(IDX_P, 3) = Scalar(2 * wp_) * x.segment(IDX_P, 3);
        g.segment(IDX_V, 2) = Scalar(2 * wv_) * x.segment(IDX_V, 2);
        g(IDX_V + 2) = Scalar(2 * wvz_) * (x(IDX_V + 2) - Scalar(vz_ref_));
        g(IDX_Q + 1) = Scalar(2 * watt_) * x(IDX_Q + 1);
        g(IDX_Q + 2) = Scalar(2 * watt_) * x(IDX_Q + 2);
        return g;
    }
    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H(IDX_P+0,IDX_P+0)=H(IDX_P+1,IDX_P+1)=H(IDX_P+2,IDX_P+2)=Scalar(2*wp_);
        H(IDX_V+0,IDX_V+0)=H(IDX_V+1,IDX_V+1)=Scalar(2*wv_);
        H(IDX_V+2,IDX_V+2)=Scalar(2*wvz_);
        H(IDX_Q+1,IDX_Q+1)=H(IDX_Q+2,IDX_Q+2)=Scalar(2*watt_);
        return H;
    }
};

template<typename Scalar>
class ScalarBoundCon : public StageConstraintBase<Scalar> {
    int idx_;
    Scalar lo_, hi_;
public:
    ScalarBoundCon(int idx, double lo, double hi) : idx_(idx), lo_(lo), hi_(hi) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 2;
    }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(2);
        cn(0) = u(idx_) - hi_;
        cn(1) = lo_ - u(idx_);
        return cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(2, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(2, u.size());
        J(0, idx_) = Scalar(1.0);
        J(1, idx_) = Scalar(-1.0);
        return J;
    }
};

template<typename Scalar>
class GlideslopeTargetCon : public StageConstraintBase<Scalar> {
    Scalar tan_gs_, dz_;
public:
    explicit GlideslopeTargetCon(double tg = GS_TAN, double dz = 0.1)
        : tan_gs_(tg), dz_(dz) {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> cn(3);
        cn(0) = tan_gs_ * (x(IDX_P + 2) + dz_);
        cn(1) = x(IDX_P + 0);
        cn(2) = x(IDX_P + 1);
        return -cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, NX_SS);
        J(0, IDX_P + 2) = -tan_gs_;
        J(1, IDX_P + 0) = Scalar(-1.0);
        J(2, IDX_P + 1) = Scalar(-1.0);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(3, u.size());
    }
};

template<typename Scalar>
class VzTargetCon : public StageConstraintBase<Scalar> {
    Scalar vz_min_;
public:
    explicit VzTargetCon(double vz_min) : vz_min_(vz_min) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return (Vector<Scalar>(1) << Scalar(vz_min_) - x(IDX_V + 2)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, x.size());
        J(0, IDX_V + 2) = Scalar(-1.0);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, u.size());
    }
};

inline Param getSolverParams() {
    Param p;
    p.reg1_min = 1e-2;
    p.reg2_min = 0.5;
    p.mu_mul = 0.1;
    p.rho = 30.0;
    p.rhoT = 5.0;
    p.rho_mul = 20.0;
    p.tolerance = 1e-4;
    p.max_iter = 500;
    p.is_quaternion_in_state = false;
    return p;
}

inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0,
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& /*prev_X*/ = {},
    const std::vector<Eigen::MatrixXd>& /*prev_K*/ = {},
    double th_init = TH_INIT)
{
    auto problem = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    auto dyn = std::make_shared<Quad6DOFTargetFrameBodyRate<double>>();
    dyn->setGravity(GRAVITY);

    auto cost  = std::make_shared<TimeCost<double>>(1e-2);
    auto tcost = std::make_shared<TermCost<double>>(500.0, 50.0, 80.0, 10.0, -0.25);
    auto cT    = std::make_shared<ScalarBoundCon<double>>(0, FMIN, FMAX);
    auto cwx   = std::make_shared<ScalarBoundCon<double>>(1, -OMEGA_MAX, OMEGA_MAX);
    auto cwy   = std::make_shared<ScalarBoundCon<double>>(2, -OMEGA_MAX, OMEGA_MAX);
    auto cwz   = std::make_shared<ScalarBoundCon<double>>(3, -OMEGA_MAX, OMEGA_MAX);
    auto cth   = std::make_shared<ScalarBoundCon<double>>(IDX_THETA, THL, THH);
    auto cvz   = std::make_shared<VzTargetCon<double>>(-VZ_LAND_MAX);
    auto cgs   = std::make_shared<GlideslopeTargetCon<double>>(GS_TAN, 0.1);

    for (int k = 0; k < HORIZON; ++k) {
        problem->setStageDynamics(k, dyn);
        problem->setStageCost(k, cost);
        problem->addStageConstraint(k, cT);
        problem->addStageConstraint(k, cwx);
        problem->addStageConstraint(k, cwy);
        problem->addStageConstraint(k, cwz);
        problem->addStageConstraint(k, cth);
        problem->addStageConstraint(k, cvz);
        problem->addStageConstraint(k, cgs);
    }
    problem->setTerminalCost(tcost);

    Eigen::VectorXd x0ss(NX_SS);
    x0ss.setZero();
    x0ss.segment(0, std::min(static_cast<int>(x0.size()), NX)) =
        x0.head(std::min(static_cast<int>(x0.size()), NX));

    problem->setInitialState(0, x0ss);

    const bool has_prev_u = (!prev_U.empty() && static_cast<int>(prev_U.size()) >= HORIZON);
    const Eigen::Vector4d q0 = x0ss.segment(IDX_Q, 4);
    const double czz = std::max(0.5, Quad6DOFVarTime<double>::calcC(q0)(2, 2));
    Eigen::VectorXd u_hover(NU_SS);
    u_hover.setZero();
    u_hover(0) = std::clamp(9.81 / czz, FMIN, FMAX);
    u_hover(IDX_THETA) = th_init;

    Eigen::VectorXd xsim = x0ss;
    for (int k = 0; k < HORIZON; ++k) {
        Eigen::VectorXd u0 = (has_prev_u && static_cast<int>(prev_U[k].size()) == NU_SS)
                             ? prev_U[k] : u_hover;
        u0(0) = std::clamp(u0(0), FMIN, FMAX);
        for (int i = 1; i <= 3; ++i) u0(i) = std::clamp(u0(i), -OMEGA_MAX, OMEGA_MAX);
        u0(IDX_THETA) = std::clamp(u0(IDX_THETA), THL, THH);
        problem->setInitialControl(k, u0);
        xsim = dyn->f(xsim, u0);
        problem->setInitialState(k + 1, xsim);
    }

    return problem;
}

inline OCPDescriptor descriptor() {
    OCPDescriptor d;
    d.name = "tracking_bodyrate_tf_imu";
    d.dt = TH_INIT;
    d.default_n_replay = DEFAULT_N_REPLAY;
    d.default_mass_kg = DEFAULT_MASS_KG;
    d.warm_start = OCPDescriptor::WarmStart::Shift;
    d.command_mode = OCPDescriptor::CommandMode::CmdBodyRate;
    d.drone_odom_mode = OCPDescriptor::DroneOdomMode::TargetFrameRelative;
    d.skip_altitude_validation = true;
    d.skip_trajectory_validation = true;
    d.variable_dt = true;
    d.state_dim = NX;
    d.control_dim = NU;

    d.state_names = {"px_tgt","py_tgt","pz_tgt","vx_tgt","vy_tgt","vz_tgt","qw","qx","qy","qz",
                     "OmNx","OmNy","OmNz","aNx","aNy","aNz","BNx","BNy","BNz","DT"};
    d.control_names = {"T","omx","omy","omz","Theta"};
    d.log_state_headers = {"px_tgt","py_tgt","pz_tgt","vx_tgt","vy_tgt","vz_tgt",
                           "qw","qx","qy","qz",
                           "OmNx","OmNy","OmNz","aNx","aNy","aNz","BNx","BNy","BNz","DT"};
    d.extract_actual_state_row = [](const Eigen::VectorXd& x,
                                    const TargetSnapshot& /*tgt*/,
                                    const std::string& /*coord*/) -> std::vector<double> {
        std::vector<double> row;
        row.reserve(NX_SS);
        for (int i = 0; i < std::min(static_cast<int>(x.size()), NX_SS); ++i)
            row.push_back(x(i));
        while (static_cast<int>(row.size()) < NX_SS) row.push_back(0.0);
        return row;
    };
    d.make_hover_state = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd h = Eigen::VectorXd::Zero(NX_SS);
        const int n = std::min(static_cast<int>(x.size()), NX);
        h.head(n) = x.head(n);
        return h;
    };

    // Passthrough + inject augmented Ω_N, â_N from target snapshot
    d.transform_state = [](const Eigen::VectorXd& x_drone, const TargetSnapshot& tgt) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(NX);
        const int n = std::min(static_cast<int>(x_drone.size()), 10);
        x.head(n) = x_drone.head(n);  // [p_B^N, v_B^N, q_NB]
        // Rotate target accel into target frame: â_N = R_NW * a_T^W
        const Eigen::Matrix3d R_NW = Quad6DOFVarTime<double>::calcC(tgt.orientation).transpose();
        x.segment(IDX_OMN, 3) = tgt.angular_velocity;
        x.segment(IDX_AN,  3) = R_NW * tgt.acceleration;
        x.segment(IDX_BN,  3) = tgt.angular_acceleration;
        return x;
    };
    d.validate_target = [](const TargetSnapshot& t, double, double) { return t.valid; };
    d.post_process_result = [](SolverResult& r, const TargetSnapshot& t) {
        r.target_snapshot_pos = t.position;
        r.target_snapshot_vel = t.velocity;
        r.target_snapshot_acc = t.acceleration;
        r.target_world_pos_trajectory.clear();
        r.target_world_vel_trajectory.clear();
        r.target_world_pos_trajectory.reserve(r.state_trajectory.size());
        r.target_world_vel_trajectory.reserve(r.state_trajectory.size());
        for (const auto& x : r.state_trajectory) {
            const double t_node = (x.size() > IDX_DT) ? x(IDX_DT) : 0.0;
            const Eigen::Vector3d tgt_p =
                t.position + t.velocity * t_node + 0.5 * t.acceleration * t_node * t_node;
            r.target_world_pos_trajectory.push_back(tgt_p);
            r.target_world_vel_trajectory.push_back(t.velocity + t.acceleration * t_node);
        }
    };
    // Carry augmented states forward from solver-propagated trajectory
    // Removed merge_prev_augmented: sensor values are authoritative for x0.
    // v_B^N (drone vel in target frame N) → world frame via R_WN = calcC(target.orientation)
    d.extract_hover_cmd = [](const Eigen::VectorXd& state,
                              const Eigen::VectorXd& control,
                              const TargetSnapshot& target) -> std::array<float,4> {
        const Eigen::Matrix3d R_WN = Quad6DOFVarTime<double>::calcC(target.orientation);
        const Eigen::Vector3d v_W  = R_WN * state.segment(IDX_V, 3) + target.velocity;
        const double z_cmd = target.position.z() + state(IDX_P + 2);
        return {float(v_W.x()), float(v_W.y()), float(z_cmd),
                float(control.size() > 3 ? control(3) : 0.0)};
    };
    d.getSolverParams = getSolverParams;
    d.create = [](const OCPCreateArgs& a) {
        return create(a.current_state, a.prev_U, a.prev_X, a.prev_K);
    };
    return d;
}

} // namespace TrackingBodyrateTfImuOCP
