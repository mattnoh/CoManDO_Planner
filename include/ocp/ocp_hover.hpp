/// @file ocp_hover.hpp
/// @brief OCP formulation for hover / point stabilisation.
///
/// State  x ∈ ℝ¹³: [pos(3), vel(3), quat(4) qw-first, omega(3)]
/// Control u ∈ ℝ⁴:  [fz_B (N), Mx (Nm·scaled), My (Nm·scaled), Mz (Nm·scaled)]

#pragma once

#include "ocp/ocp_base.hpp"

namespace HoverOCP {

// ── Fixed parameters ──────────────────────────────────────────────────────────
const int    HORIZON = 100;
const double DT      = 0.05;     // seconds per OCP step
const double MASS    = 0.027;    // kg

// Scale inertia so diagonal entries become O(1) — improves solver conditioning
const double J_SCALE = 1.0 / 1.66e-5;   // ≈ 60240
const Eigen::Matrix3d INERTIA = (Eigen::Matrix3d() <<
    1.66e-5 * J_SCALE, 0.0,             0.0,
    0.0,             1.66e-5 * J_SCALE, 0.0,
    0.0,             0.0,             2.92e-5 * J_SCALE).finished();

// ── Constraint parameters ─────────────────────────────────────────────────────
const double FMAX = 1.2;    // N — max collective thrust

// ── Solver parameters ─────────────────────────────────────────────────────────
const double SOLVER_REG1_MIN  = 1e-6;
const double SOLVER_REG2_MIN  = 1.0;
const double SOLVER_MU_MUL    = 0.1;
const double SOLVER_RHO       = 20.0;
const double SOLVER_RHO_MUL   = 9.0;
const double SOLVER_TOLERANCE = 1e-3;
const int    SOLVER_MAX_ITER  = 200;

// ── Stage cost weights ────────────────────────────────────────────────────────
const double W_POS_STAGE     = 15.0;
const double W_VEL_STAGE     = 5.0;
const double W_ATT_STAGE     = 2.0;
const double W_ANGRATE_STAGE = 5.0;
const double W_THRUST        = 1e-1;
const double W_MOMENT        = 1e-1;

// ── Terminal cost weights ─────────────────────────────────────────────────────
const double TERM_POS_WEIGHT     = 100.0;
const double TERM_VEL_WEIGHT     = 100.0;
const double TERM_ATT_WEIGHT     = 200.0;
const double TERM_ANGRATE_WEIGHT = 100.0;

// ─────────────────────────────────────────────────────────────────────────────
// Stage cost
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class StageCost : public StageCostBase<Scalar> {
    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
public:
    StageCost(const Eigen::Vector3d& target_pos,
              const Eigen::Vector3d& target_vel = Eigen::Vector3d::Zero())
        : target_pos_(target_pos), target_vel_(target_vel) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Eigen::Vector3d pos   = x.template segment<3>(0);
        Eigen::Vector3d vel   = x.template segment<3>(3);
        Eigen::Vector3d omega = x.template segment<3>(10);
        Scalar q0 = x(6);
        Eigen::Vector3d qv = x.template segment<3>(7);
        Scalar att_err = qv.squaredNorm() + (1.0 - q0) * (1.0 - q0);
        Scalar fz  = u(0);
        Eigen::Vector3d m = u.template segment<3>(1);
        return W_POS_STAGE     * (pos - target_pos_).squaredNorm()
             + W_VEL_STAGE     * (vel - target_vel_).squaredNorm()
             + W_ATT_STAGE     * att_err
             + W_ANGRATE_STAGE * omega.squaredNorm()
             + W_THRUST        * fz * fz
             + W_MOMENT        * m.squaredNorm();
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.template segment<3>(0)  = 2.0 * W_POS_STAGE     * (x.template segment<3>(0).eval() - target_pos_);
        g.template segment<3>(3)  = 2.0 * W_VEL_STAGE     * (x.template segment<3>(3).eval() - target_vel_);
        Scalar q0 = x(6);
        Eigen::Vector3d qv = x.template segment<3>(7);
        g(6)                       = -2.0 * W_ATT_STAGE * (1.0 - q0);
        g.template segment<3>(7)  =  2.0 * W_ATT_STAGE * qv;
        g.template segment<3>(10) =  2.0 * W_ANGRATE_STAGE * x.template segment<3>(10);
        return g;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> g = Vector<Scalar>::Zero(u.size());
        g(0) = 2.0 * W_THRUST * u(0);
        g.template segment<3>(1) = 2.0 * W_MOMENT * u.template segment<3>(1);
        return g;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.template block<3,3>(0,0)   = 2.0 * W_POS_STAGE     * Matrix<Scalar>::Identity(3,3);
        H.template block<3,3>(3,3)   = 2.0 * W_VEL_STAGE     * Matrix<Scalar>::Identity(3,3);
        H(6,6)                        = 2.0 * W_ATT_STAGE;
        H.template block<3,3>(7,7)   = 2.0 * W_ATT_STAGE     * Matrix<Scalar>::Identity(3,3);
        H.template block<3,3>(10,10) = 2.0 * W_ANGRATE_STAGE * Matrix<Scalar>::Identity(3,3);
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(u.size(), u.size());
        H(0,0) = 2.0 * W_THRUST;
        H.template block<3,3>(1,1) = 2.0 * W_MOMENT * Matrix<Scalar>::Identity(3,3);
        return H;
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(x.size(), u.size());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Terminal cost
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class TerminalCost : public TerminalCostBase<Scalar> {
    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
public:
    explicit TerminalCost(const Eigen::VectorXd& target_state)
        : target_pos_(target_state.segment<3>(0)),
          target_vel_(target_state.segment<3>(3)) {}

    Scalar p(const Vector<Scalar>& x) const override {
        Eigen::Vector3d pos   = x.template segment<3>(0);
        Eigen::Vector3d vel   = x.template segment<3>(3);
        Scalar q0 = x(6);
        Eigen::Vector3d qv    = x.template segment<3>(7);
        Eigen::Vector3d omega = x.template segment<3>(10);
        Scalar att_err = qv.squaredNorm() + (1.0 - q0) * (1.0 - q0);
        return TERM_POS_WEIGHT     * (pos - target_pos_).squaredNorm()
             + TERM_VEL_WEIGHT     * (vel - target_vel_).squaredNorm()
             + TERM_ATT_WEIGHT     * att_err
             + TERM_ANGRATE_WEIGHT * omega.squaredNorm();
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.template segment<3>(0)  = 2.0 * TERM_POS_WEIGHT * (x.template segment<3>(0) - target_pos_);
        g.template segment<3>(3)  = 2.0 * TERM_VEL_WEIGHT * (x.template segment<3>(3) - target_vel_);
        Scalar q0 = x(6);
        Eigen::Vector3d qv = x.template segment<3>(7);
        g(6)                       = -2.0 * TERM_ATT_WEIGHT * (1.0 - q0);
        g.template segment<3>(7)  =  2.0 * TERM_ATT_WEIGHT * qv;
        g.template segment<3>(10) =  2.0 * TERM_ANGRATE_WEIGHT * x.template segment<3>(10);
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        (void)x;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H.template block<3,3>(0,0)   = 2.0 * TERM_POS_WEIGHT     * Matrix<Scalar>::Identity(3,3);
        H.template block<3,3>(3,3)   = 2.0 * TERM_VEL_WEIGHT     * Matrix<Scalar>::Identity(3,3);
        H(6,6)                        = 2.0 * TERM_ATT_WEIGHT;
        H.template block<3,3>(7,7)   = 2.0 * TERM_ATT_WEIGHT     * Matrix<Scalar>::Identity(3,3);
        H.template block<3,3>(10,10) = 2.0 * TERM_ANGRATE_WEIGHT * Matrix<Scalar>::Identity(3,3);
        return H;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Max thrust constraint  fz_B ≤ FMAX
// ─────────────────────────────────────────────────────────────────────────────
template <typename Scalar>
class MaxThrustConstraint : public StageConstraintBase<Scalar> {
    Scalar fmax_;
public:
    explicit MaxThrustConstraint(Scalar fmax = static_cast<Scalar>(FMAX)) : fmax_(fmax) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; Vector<Scalar> c_n(1); c_n(0) = u(0) - fmax_; return c_n;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u; return Matrix<Scalar>::Zero(1, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size()); J(0,0) = 1.0; return J;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& current_state,
    const Eigen::VectorXd& terminal_state)
{
    auto prob = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    auto dyn = std::make_shared<Quad6DOF<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(Eigen::Vector3d(0.0, 0.0, -9.81));
    dyn->setJb(INERTIA);
    dyn->setDt(DT);
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageDynamics(i, dyn);

    Eigen::Vector3d target_pos = terminal_state.segment<3>(0);
    Eigen::Vector3d target_vel = terminal_state.segment<3>(3);
    auto stage_cost = std::make_shared<StageCost<double>>(target_pos, target_vel);
    for (int i = 0; i < HORIZON; ++i)
        prob->setStageCost(i, stage_cost);

    prob->setTerminalCost(std::make_shared<TerminalCost<double>>(terminal_state));

    auto mt = std::make_shared<MaxThrustConstraint<double>>(FMAX);
    for (int i = 0; i < HORIZON; ++i)
        prob->addStageConstraint(i, mt);

    prob->setInitialState(0, current_state);

    Eigen::VectorXd u0(4);
    u0 << MASS * 9.81, 0.0, 0.0, 0.0;
    for (int i = 0; i < HORIZON; ++i)
        prob->setInitialControl(i, u0);

    return prob;
}

} // namespace HoverOCP