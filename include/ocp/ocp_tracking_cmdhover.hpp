/// @file ocp_tracking_cmdhover.hpp
/// @brief Body-frame relative tracking OCP used by cmd_hover mode.
///
/// Dynamics (Quad6DOFBodyFrameRelative) and BodyRelativeVelocityBuffer live in
/// quad_6dof_dynamics_aug.h so they can be shared with other body-relative OCPs.
#pragma once

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "target/target_accel_buffer.hpp"
#include "dynamics/quad_6dof_dynamics_aug.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace TrackingCmdHoverOCP {

static constexpr int NX = 13;
static constexpr int NU = 4;
static constexpr int NX_SS = 14;
static constexpr int NU_SS = 5;
static constexpr int IDX_DT = 13;
static constexpr int IDX_THETA = 4;

static constexpr int IDX_P = 0;
static constexpr int IDX_VD = 3;
static constexpr int IDX_Q = 6;
static constexpr int IDX_OM = 10;

static constexpr int HORIZON = 30;
static constexpr double TH_INIT = 0.1;
static constexpr double THL = 0.05;
static constexpr double THH = 0.2;
static constexpr int DEFAULT_N_REPLAY = 7;

static constexpr double MASS = 0.027;
static constexpr double DEFAULT_MASS_KG = MASS;
static constexpr double IXX = 1.66e-5;
static constexpr double IYY = 1.66e-5;
static constexpr double IZZ = 2.92e-5;
static constexpr double J_SCALE = 1.0 / IXX;
static constexpr double FMIN = 0.08;
static constexpr double FMAX = 0.6;
static constexpr double L_ARM = 0.046;
static constexpr double TAU_MAX = L_ARM * (FMAX / 4.0 - FMIN / 4.0) * J_SCALE;

static constexpr double GS_DEG = 60.0;
static const double GS_TAN = std::tan(GS_DEG * M_PI / 180.0);
static constexpr double VZ_LAND_MAX = 2.5;
static constexpr double TARGET_Z = 1.5;

static const Eigen::Matrix3d J_B = []() {
    Eigen::Matrix3d J;
    J.setZero();
    J(0, 0) = IXX * J_SCALE;
    J(1, 1) = IYY * J_SCALE;
    J(2, 2) = IZZ * J_SCALE;
    return J;
}();
static const Eigen::Matrix3d J_B_INV = J_B.inverse();
static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

// Adapt TargetAccelBuffer → BodyRelativeVelocityBuffer by integrating accelerations.
// BodyRelativeVelocityBuffer and Quad6DOFBodyFrameRelative are defined in
// quad_6dof_dynamics_aug.h (included above) for reuse across body-relative OCPs.
inline BodyRelativeVelocityBuffer makeBodyRelativeVelocityBuffer(
    const target_models::TargetAccelBuffer& acc_buf,
    const Eigen::Vector3d& v0)
{
    BodyRelativeVelocityBuffer out;
    out.t_start = acc_buf.t_start;
    out.dt      = acc_buf.dt;

    if (acc_buf.accels.empty()) {
        out.vels = {v0};
        return out;
    }

    out.vels.resize(acc_buf.accels.size());
    Eigen::Vector3d v = v0;
    for (size_t i = 0; i < acc_buf.accels.size(); ++i) {
        out.vels[i] = v;
        v += acc_buf.accels[i] * out.dt;
    }
    return out;
}

// Quad6DOFBodyFrameRelative<Scalar> is now defined in quad_6dof_dynamics_aug.h.
// All OCP cost and constraint classes follow below.

template<typename Scalar>
class TimeCost : public StageCostBase<Scalar> {
    Scalar eps_, w_vz_;
public:
    explicit TimeCost(double e = 1e-4, double w_vz = 1.0)
        : eps_(e), w_vz_(w_vz) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return u(IDX_THETA)
             + eps_ * (x.segment(IDX_VD, 3).squaredNorm() + x.segment(IDX_OM, 3).squaredNorm())
             + w_vz_ * x(IDX_VD + 2) * x(IDX_VD + 2);
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NX_SS);
        g.segment(IDX_VD, 3) = Scalar(2) * eps_ * x.segment(IDX_VD, 3);
        g.segment(IDX_OM, 3) = Scalar(2) * eps_ * x.segment(IDX_OM, 3);
        g(IDX_VD + 2) += Scalar(2) * w_vz_ * x(IDX_VD + 2);
        return g;
    }

    Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NU_SS);
        g(IDX_THETA) = Scalar(1);
        return g;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        for (int i = IDX_VD; i < IDX_VD + 3; ++i) H(i, i) = Scalar(2) * eps_;
        for (int i = IDX_OM; i < IDX_OM + 3; ++i) H(i, i) = Scalar(2) * eps_;
        H(IDX_VD + 2, IDX_VD + 2) += Scalar(2) * w_vz_;
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
class BodyFrameTermCost : public TerminalCostBase<Scalar> {
    double wp_, wv_, wvz_, watt_, wom_, vz_ref_;

public:
    explicit BodyFrameTermCost(double wp = 500.0, double wv = 30.0,
                               double wvz = 80.0, double watt = 200.0,
                               double wom = 50.0, double vz_ref = -0.1)
        : wp_(wp), wv_(wv), wvz_(wvz), watt_(watt), wom_(wom), vz_ref_(vz_ref) {}

    Scalar p(const Vector<Scalar>& x) const override {
        const Scalar ep = x.template segment<2>(IDX_P).squaredNorm();
        const Scalar ez = x(IDX_P + 2) * x(IDX_P + 2);
        // Terminal lateral speed penalty in body frame (matches standalone baseline).
        const Scalar evxy = x.template segment<2>(IDX_VD).squaredNorm();
        const Scalar evz = (x(IDX_VD + 2) - Scalar(vz_ref_)) * (x(IDX_VD + 2) - Scalar(vz_ref_));
        const Scalar eatt = x(IDX_Q + 1) * x(IDX_Q + 1) + x(IDX_Q + 2) * x(IDX_Q + 2);
        const Scalar eom = x.template segment<3>(IDX_OM).squaredNorm();
        return Scalar(wp_) * (ep + ez)
             + Scalar(wv_) * evxy
             + Scalar(wvz_) * evz
             + Scalar(watt_) * eatt
             + Scalar(wom_) * eom;
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.segment(IDX_P, 2) = Scalar(2 * wp_) * x.segment(IDX_P, 2);
        g(IDX_P + 2) = Scalar(2 * wp_) * x(IDX_P + 2);
        g.segment(IDX_VD, 2) = Scalar(2 * wv_) * x.segment(IDX_VD, 2);
        g(IDX_VD + 2) = Scalar(2 * wvz_) * (x(IDX_VD + 2) - Scalar(vz_ref_));
        g(IDX_Q + 1) = Scalar(2 * watt_) * x(IDX_Q + 1);
        g(IDX_Q + 2) = Scalar(2 * watt_) * x(IDX_Q + 2);
        g.segment(IDX_OM, 3) = Scalar(2 * wom_) * x.segment(IDX_OM, 3);
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H(IDX_P + 0, IDX_P + 0) = H(IDX_P + 1, IDX_P + 1) = H(IDX_P + 2, IDX_P + 2) = Scalar(2 * wp_);
        H(IDX_VD + 0, IDX_VD + 0) = H(IDX_VD + 1, IDX_VD + 1) = Scalar(2 * wv_);
        H(IDX_VD + 2, IDX_VD + 2) = Scalar(2 * wvz_);
        H(IDX_Q + 1, IDX_Q + 1) = H(IDX_Q + 2, IDX_Q + 2) = Scalar(2 * watt_);
        H(IDX_OM + 0, IDX_OM + 0) = H(IDX_OM + 1, IDX_OM + 1) = H(IDX_OM + 2, IDX_OM + 2) = Scalar(2 * wom_);
        return H;
    }
};

template<typename Scalar>
class FminCon : public StageConstraintBase<Scalar> {
public:
    FminCon() { this->constraint_type = ConstraintType::NO; this->dim_c = 1; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return (Vector<Scalar>(1) << Scalar(FMIN) - u(0)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        J(0, 0) = Scalar(-1);
        return J;
    }
};

template<typename Scalar>
class FmaxCon : public StageConstraintBase<Scalar> {
public:
    FmaxCon() { this->constraint_type = ConstraintType::NO; this->dim_c = 1; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return (Vector<Scalar>(1) << u(0) - Scalar(FMAX)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size());
        J(0, 0) = Scalar(1);
        return J;
    }
};

template<typename Scalar>
class MomentCon : public StageConstraintBase<Scalar> {
    Scalar tau_;
public:
    MomentCon() : tau_(static_cast<Scalar>(TAU_MAX)) {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 4;
    }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(4);
        cn << tau_, u(1), u(2), u(3);
        return -cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(4, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(4, u.size());
        J(1, 1) = J(2, 2) = J(3, 3) = Scalar(-1);
        return J;
    }
};

template<typename Scalar>
class ThetaBounds : public StageConstraintBase<Scalar> {
    Scalar lo_, hi_;
public:
    ThetaBounds(double lo, double hi) : lo_(lo), hi_(hi) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 2;
    }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(2);
        cn(0) = u(IDX_THETA) - hi_;
        cn(1) = lo_ - u(IDX_THETA);
        return cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(2, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(2, u.size());
        J(0, IDX_THETA) = Scalar(1);
        J(1, IDX_THETA) = Scalar(-1);
        return J;
    }
};

template<typename Scalar>
class ZAboveCon : public StageConstraintBase<Scalar> {
public:
    ZAboveCon() { this->constraint_type = ConstraintType::NO; this->dim_c = 1; }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return (Vector<Scalar>(1) << x(IDX_P + 2)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, x.size());
        J(0, IDX_P + 2) = Scalar(1);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, u.size());
    }
};

template<typename Scalar>
class VzBodyCon : public StageConstraintBase<Scalar> {
    Scalar vz_min_;
public:
    explicit VzBodyCon(double vz_min) : vz_min_(vz_min) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return (Vector<Scalar>(1) << Scalar(vz_min_) - x(IDX_VD + 2)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, x.size());
        J(0, IDX_VD + 2) = Scalar(-1);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, u.size());
    }
};

template<typename Scalar>
class GlideslopeBodyCon : public StageConstraintBase<Scalar> {
    Scalar tan_gs_;
public:
    explicit GlideslopeBodyCon(double tg = GS_TAN) : tan_gs_(tg) {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 3;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> cn(3);
        cn(0) = -tan_gs_ * x(IDX_P + 2);
        cn(1) = x(IDX_P + 0);
        cn(2) = x(IDX_P + 1);
        return -cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(3, NX_SS);
        J(0, IDX_P + 2) = tan_gs_;
        J(1, IDX_P + 0) = Scalar(-1);
        J(2, IDX_P + 1) = Scalar(-1);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(3, u.size());
    }
};

struct TrackingCmdHoverExtra {
    target_models::TargetAccelBuffer accel_buf;
    Eigen::Vector3d target_velocity0 = Eigen::Vector3d::Zero();
    double t_abs = 0.0;
};

inline Param getSolverParams() {
    Param p;
    p.reg1_min = 1e-2;
    p.reg2_min = 0.5;
    p.mu_mul = 0.1;
    p.rho = 10.0;
    p.rhoT = 1.0;
    p.rho_mul = 10.0;
    p.tolerance = 1e-4;
    p.max_iter = 500;
    p.is_quaternion_in_state = false;
    return p;
}

inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0_body,
    const TrackingCmdHoverExtra& ex,
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& prev_X = {},
    const std::vector<Eigen::MatrixXd>& prev_K = {},
    double th_init = TH_INIT,
    double th_min = THL,
    double th_max = THH)
{
    auto problem = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    auto valid_state = [](const Eigen::VectorXd& x) {
        return x.size() == NX_SS;
    };
    auto valid_control = [](const Eigen::VectorXd& u) {
        return u.size() == NU_SS;
    };
    auto valid_gain = [](const Eigen::MatrixXd& K) {
        return K.rows() == NU_SS && K.cols() == NX_SS;
    };

    BodyRelativeVelocityBuffer vel_buf = makeBodyRelativeVelocityBuffer(ex.accel_buf, ex.target_velocity0);
    auto dyn = std::make_shared<Quad6DOFBodyFrameRelative<double>>(vel_buf, ex.t_abs);
    dyn->setMass(MASS);
    dyn->setGravity(GRAVITY);
    dyn->setJb(J_B);

    auto cost = std::make_shared<TimeCost<double>>(1e-4, 1.0);
    auto tcost = std::make_shared<BodyFrameTermCost<double>>(500.0, 30.0, 80.0, 200.0, 50.0, -0.1);
    auto cfmin = std::make_shared<FminCon<double>>();
    auto cfmax = std::make_shared<FmaxCon<double>>();
    auto cmom = std::make_shared<MomentCon<double>>();
    auto cth = std::make_shared<ThetaBounds<double>>(th_min, th_max);
    auto czab = std::make_shared<ZAboveCon<double>>();
    auto cvz = std::make_shared<VzBodyCon<double>>(-VZ_LAND_MAX);
    auto cgs = std::make_shared<GlideslopeBodyCon<double>>();

    for (int k = 0; k < HORIZON; ++k) {
        problem->setStageDynamics(k, dyn);
        problem->setStageCost(k, cost);
        problem->addStageConstraint(k, cfmin);
        problem->addStageConstraint(k, cfmax);
        problem->addStageConstraint(k, cmom);
        problem->addStageConstraint(k, cth);
        problem->addStageConstraint(k, czab);
        problem->addStageConstraint(k, cvz);
        // problem->addStageConstraint(k, cgs);
    }
    problem->setTerminalCost(tcost);

    Eigen::VectorXd x0ss(NX_SS);
    x0ss.segment(0, NX) = x0_body;
    x0ss(IDX_DT) = 0.0;
    problem->setInitialState(0, x0ss);

    Eigen::VectorXd sim14(NX_SS);
    sim14.segment(0, NX) = x0_body;
    sim14(IDX_DT) = 0.0;

    // K-feedback is intentionally not used: state is body-frame-relative, which
    // rotates between solves. K*(sim14 - prev_X[k]) subtracts across different
    // body frames and produces garbage corrections causing wildly different solves.
    // Use a simple shift warmstart with clamping instead.
    const bool has_prev_u = (!prev_U.empty()
        && static_cast<int>(prev_U.size()) >= HORIZON);

    for (int k = 0; k < HORIZON; ++k) {
        Eigen::VectorXd u0(NU_SS);
        u0.setZero();

        if (has_prev_u && valid_control(prev_U[k])) {
            u0 = prev_U[k];
            u0(0) = std::clamp(u0(0), FMIN, FMAX);
            for (int i = 1; i < 4; ++i) {
                u0(i) = std::clamp(u0(i), -TAU_MAX, TAU_MAX);
            }
            u0(IDX_THETA) = std::clamp(u0(IDX_THETA), th_min, th_max);
        } else {
            const Eigen::Vector4d q_ws = sim14.segment(IDX_Q, 4);
            const Eigen::Matrix3d C_ws = Quad6DOFVarTime<double>::calcC(q_ws);
            const double czz = std::max(0.5, C_ws.transpose()(2, 2));
            u0(0) = std::clamp(MASS * 9.81 / czz, FMIN, FMAX);
            u0(IDX_THETA) = th_init;
        }

        problem->setInitialControl(k, u0);
        Eigen::VectorXd next14 = dyn->f(sim14, u0);
        if (next14(IDX_P + 2) > 0.0) {
            next14(IDX_P + 2) = 0.0;
        }
        problem->setInitialState(k + 1, next14);
        sim14 = next14;
    }

    return problem;
}

} // namespace TrackingCmdHoverOCP
