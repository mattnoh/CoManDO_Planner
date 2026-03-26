/// @file ocp_tracking_circle.hpp
/// @brief OCP formulation for tracking a circular trajectory (absolute frame).
///
/// This is a port of quad_cf_tracking_rh_circle.cpp from ALIPDDP-main.
/// Key differences from original:
/// - Uses absolute frame (same as original)
/// - Rebuilds OCP each iteration (original reused cterm with setTabs)
/// - Receives live t_abs and CircularTarget from ROS via planner_node
///
/// Reference: ALIPDDP-main/problem_examples/quad_cf_tracking_rh_circle.cpp
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"

namespace TrackingCircleOCP {

// ── Dimensions ────────────────────────────────────────────────────────────────
static constexpr int NX = 13;
static constexpr int NU = 4;
static constexpr int NX_SS = 14;
static constexpr int NU_SS = 5;
static constexpr int IDX_DT = 13;
static constexpr int IDX_THETA = 4;

// ── Horizon / timing ─────────────────────────────────────────────────────────
static constexpr int HORIZON = 30;
static constexpr double TH_INIT = 0.1;
static constexpr double THL = 0.05;
static constexpr double THH = 0.2;
static constexpr int DEFAULT_N_REPLAY = 7;

// ── Vehicle parameters ───────────────────────────────────────────────────────
static constexpr double MASS = 0.027;
static constexpr double IXX = 1.66e-5;
static constexpr double IYY = 1.66e-5;
static constexpr double IZZ = 2.92e-5;
static constexpr double J_SCALE = 1.0 / IXX;
static constexpr double FMIN = 0.08;
static constexpr double FMAX = 0.6;
static constexpr double L_ARM = 0.046;
static const double TAU_MAX = L_ARM * (FMAX/4.0 - FMIN/4.0) * J_SCALE;

static const Eigen::Matrix3d J_B = [](){
    Eigen::Matrix3d J; J.setZero();
    J(0,0)=IXX*J_SCALE; J(1,1)=IYY*J_SCALE; J(2,2)=IZZ*J_SCALE;
    return J;
}();
static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

// ── Solver parameters ─────────────────────────────────────────────────────────
static constexpr double SOLVER_REG1_MIN = 1e-2;
static constexpr double SOLVER_REG2_MIN = 0.5;
static constexpr double SOLVER_MU_MUL = 0.1;
static constexpr double SOLVER_RHO = 1.0;
static constexpr double SOLVER_RHOT = 10.0;
static constexpr double SOLVER_RHO_MUL = 10.0;
static constexpr double SOLVER_TOLERANCE = 1e-4;
static constexpr int SOLVER_MAX_ITER = 500;

inline Param getSolverParams() {
    Param p;
    p.reg1_min = SOLVER_REG1_MIN;
    p.reg2_min = SOLVER_REG2_MIN;
    p.mu_mul = SOLVER_MU_MUL;
    p.rho = SOLVER_RHO;
    p.rhoT = SOLVER_RHOT;
    p.rho_mul = SOLVER_RHO_MUL;
    p.tolerance = SOLVER_TOLERANCE;
    p.max_iter = SOLVER_MAX_ITER;
    p.is_quaternion_in_state = false;
    return p;
}

// ── Helper: Rotation matrix from quaternion ───────────────────────────────────
static Eigen::Matrix3d calcC(const Eigen::Vector4d& q) {
    double qw=q(0), qx=q(1), qy=q(2), qz=q(3);
    Eigen::Matrix3d C;
    C << 1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy),
         2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx),
         2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy);
    return C;
}

// ── Helper: Omega matrix for quaternion derivative ────────────────────────────
static Eigen::Matrix4d calcOmega(const Eigen::Vector3d& w) {
    Eigen::Matrix4d Om;
    Om << 0, -w(0), -w(1), -w(2),
          w(0), 0, w(2), -w(1),
          w(1), -w(2), 0, w(0),
          w(2), w(1), -w(0), 0;
    return Om;
}

// ── Helper: Continuous state derivative (physical 13-dim) ─────────────────────
static Eigen::VectorXd xdot(const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    Eigen::Vector3d v = x.segment(3, 3);
    Eigen::Vector3d w = x.segment(10, 3);
    Eigen::Vector4d q = x.segment(6, 4);
    Eigen::VectorXd d(NX);
    d.segment(0, 3) = v;
    d.segment(3, 3) = calcC(q).col(2) * u(0) / MASS + GRAVITY;
    d.segment(6, 4) = 0.5 * calcOmega(w) * q;
    d.segment(10, 3) = J_B.inverse() * (u.segment(1, 3) - w.cross(J_B * w));
    return d;
}

// ── Helper: RK4 integration with quaternion normalization ─────────────────────
static Eigen::VectorXd rk4(const Eigen::VectorXd& x, const Eigen::VectorXd& u, double Th) {
    auto k1 = xdot(x, u);
    auto k2 = xdot(x + 0.5*Th*k1, u);
    auto k3 = xdot(x + 0.5*Th*k2, u);
    auto k4 = xdot(x + Th*k3, u);
    Eigen::VectorXd xn = x + (Th/6.0) * (k1 + 2*k2 + 2*k3 + k4);
    xn.segment(6, 4).normalize();
    return xn;
}

// ── Helper: Tilt quaternion toward a direction (from reference) ───────────────
static Eigen::Vector4d tiltQuat(const Eigen::Vector3d& dir) {
    Eigen::Vector3d ez(0, 0, 1);
    Eigen::Vector3d ax = ez.cross(dir);
    double sa = ax.norm(), ca = ez.dot(dir);
    if (sa < 1e-8) {
        return (ca > 0) ? Eigen::Vector4d(1, 0, 0, 0) : Eigen::Vector4d(0, 1, 0, 0);
    }
    ax /= sa;
    double h = std::atan2(sa, ca) / 2.0;
    Eigen::Vector4d q;
    q(0) = std::cos(h);
    q.segment(1, 3) = std::sin(h) * ax;
    return q;
}

// ── Stage cost ───────────────────────────────────────────────────────────────
template<typename Scalar>
class TimeCost: public StageCostBase<Scalar> {
    Scalar eps_;
public:
    explicit TimeCost(double e=1e-4): eps_(static_cast<Scalar>(e)) {}
    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        return u(IDX_THETA) + eps_*(x.segment(3,3).squaredNorm() + x.segment(10,3).squaredNorm());
    }
    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NX_SS);
        g.segment(3,3) = Scalar(2)*eps_*x.segment(3,3);
        g.segment(10,3) = Scalar(2)*eps_*x.segment(10,3);
        return g;
    }
    Vector<Scalar> qu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NU_SS); g(IDX_THETA) = Scalar(1); return g;
    }
    Matrix<Scalar> qxx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        for(int i=3; i<6; ++i) H(i,i) = Scalar(2)*eps_;
        for(int i=10; i<13; ++i) H(i,i) = Scalar(2)*eps_;
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
class ZeroTermCost: public TerminalCostBase<Scalar> {
public:
    Scalar p(const Vector<Scalar>&) const override { return Scalar(0); }
    Vector<Scalar> px(const Vector<Scalar>& x) const override { return Vector<Scalar>::Zero(x.size()); }
    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override { return Matrix<Scalar>::Zero(x.size(), x.size()); }
};

// ── Stage constraints ────────────────────────────────────────────────────────
template<typename Scalar>
class FminCon: public StageConstraintBase<Scalar> {
public:
    FminCon() { this->constraint_type = ConstraintType::NO; this->dim_c = 1; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return (Vector<Scalar>(1) << Scalar(FMIN) - u(0)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size()); J(0,0) = Scalar(-1); return J;
    }
};

template<typename Scalar>
class FmaxCon: public StageConstraintBase<Scalar> {
public:
    FmaxCon() { this->constraint_type = ConstraintType::NO; this->dim_c = 1; }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return (Vector<Scalar>(1) << u(0) - Scalar(FMAX)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(1, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, u.size()); J(0,0) = Scalar(1); return J;
    }
};

template<typename Scalar>
class MomentCon: public StageConstraintBase<Scalar> {
    Scalar tau_;
public:
    MomentCon(): tau_(static_cast<Scalar>(TAU_MAX)) {
        this->constraint_type = ConstraintType::SOC; this->dim_c = 4;
    }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(4); cn << tau_, u(1), u(2), u(3); return -cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(4, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(4, u.size());
        J(1,1) = J(2,2) = J(3,3) = Scalar(-1); return J;
    }
};

template<typename Scalar>
class ThetaBounds: public StageConstraintBase<Scalar> {
    Scalar lo_, hi_;
public:
    ThetaBounds(double lo, double hi): lo_(static_cast<Scalar>(lo)), hi_(static_cast<Scalar>(hi)) {
        this->constraint_type = ConstraintType::NO; this->dim_c = 2;
    }
    Vector<Scalar> c(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Vector<Scalar> cn(2); cn(0) = u(IDX_THETA) - hi_; cn(1) = lo_ - u(IDX_THETA); return cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(2, x.size());
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(2, u.size());
        J(0,IDX_THETA) = Scalar(1); J(1,IDX_THETA) = Scalar(-1); return J;
    }
};

// ── Terminal constraint for intercepting the moving circle ───────────────────
struct CircularTarget {
    Eigen::Vector3d center;
    double R, omega, phi0;

    Eigen::Vector3d pos(double t) const {
        double ph = omega*t + phi0;
        return center + Eigen::Vector3d(R*std::cos(ph), R*std::sin(ph), 0.0);
    }
    Eigen::Vector3d vel(double t) const {
        double ph = omega*t + phi0;
        return Eigen::Vector3d(-R*omega*std::sin(ph), R*omega*std::cos(ph), 0.0);
    }
    Eigen::VectorXd state(double t) const {
        Eigen::VectorXd s(6); s.head(3) = pos(t); s.tail(3) = vel(t); return s;
    }
    Eigen::Vector3d dpos_dt(double t) const { return vel(t); }
    Eigen::Vector3d dvel_dt(double t) const {
        double ph = omega*t + phi0;
        return Eigen::Vector3d(-R*omega*omega*std::cos(ph), -R*omega*omega*std::sin(ph), 0.0);
    }
};

template<typename Scalar>
class CircularInterceptCon: public TerminalConstraintBase<Scalar> {
    CircularTarget tgt_;
    double t_abs_;
public:
    CircularInterceptCon(const CircularTarget& tgt, double t_abs): tgt_(tgt), t_abs_(t_abs) {
        this->constraint_type = ConstraintType::EQ; this->dim_cT = 6;
    }
    void setTabs(double t) { t_abs_ = t; }

    Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        double tau = t_abs_ + static_cast<double>(x(IDX_DT));
        Eigen::Vector3d p_ref = tgt_.pos(tau), v_ref = tgt_.vel(tau);
        Vector<Scalar> c(6);
        for(int i=0; i<3; ++i) { c(i) = x(i) - Scalar(p_ref(i)); c(3+i) = x(3+i) - Scalar(v_ref(i)); }
        return c;
    }
    Matrix<Scalar> cTx(const Vector<Scalar>& x) const override {
        double tau = t_abs_ + static_cast<double>(x(IDX_DT));
        Eigen::Vector3d dp = tgt_.dpos_dt(tau), dv = tgt_.dvel_dt(tau);
        Matrix<Scalar> J = Matrix<Scalar>::Zero(6, NX_SS);
        J.block(0,0,3,3) = Matrix<Scalar>::Identity(3,3);
        J.block(3,3,3,3) = Matrix<Scalar>::Identity(3,3);
        for(int i=0; i<3; ++i) { J(i,IDX_DT) = Scalar(-dp(i)); J(3+i,IDX_DT) = Scalar(-dv(i)); }
        return J;
    }
};

// ── Factory function to create the OCP instance ─────────────────────────────
/// @brief Creates the OCP for tracking a circular target.
/// @param x0 Initial state (13-dim: pos, vel, quat, omega)
/// @param tgt CircularTarget parameters (center, R, omega, phi0)
/// @param t_abs Absolute time at solve start (advanced each RH iteration)
/// @param th_init Initial timestep guess
/// @param th_min Minimum timestep
/// @param th_max Maximum timestep
/// @param cterm_out Optional output for terminal constraint (for setTabs updates)
/// @param prev_U Warm-start controls from previous solve (NU_SS=5 dim each)
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0,
    const CircularTarget& tgt,
    double t_abs,
    double th_init = TH_INIT,
    double th_min = THL,
    double th_max = THH,
    std::shared_ptr<CircularInterceptCon<double>>* cterm_out = nullptr,
    const std::vector<Eigen::VectorXd>& prev_U = {}
) {
    auto problem = std::make_shared<OptimalControlProblem<double>>(HORIZON);
    auto dyn = std::make_shared<Quad6DOFVarTime<double>>();
    dyn->setMass(MASS);
    dyn->setGravity(GRAVITY);
    dyn->setJb(J_B);

    auto cost = std::make_shared<TimeCost<double>>(1e-4);
    auto tcost = std::make_shared<ZeroTermCost<double>>();
    auto cfmin = std::make_shared<FminCon<double>>();
    auto cfmax = std::make_shared<FmaxCon<double>>();
    auto cmom = std::make_shared<MomentCon<double>>();
    auto cth = std::make_shared<ThetaBounds<double>>(th_min, th_max);
    auto cterm = std::make_shared<CircularInterceptCon<double>>(tgt, t_abs);
    if (cterm_out) *cterm_out = cterm;

    for(int k=0; k<HORIZON; ++k) {
        problem->setStageDynamics(k, dyn);
        problem->setStageCost(k, cost);
        problem->addStageConstraint(k, cfmin);
        problem->addStageConstraint(k, cfmax);
        problem->addStageConstraint(k, cmom);
        problem->addStageConstraint(k, cth);
    }
    problem->setTerminalCost(tcost);
    problem->addTerminalConstraint(cterm);

    // Initial state
    Eigen::VectorXd x0ss(NX_SS);
    x0ss.segment(0, NX) = x0;
    x0ss(IDX_DT) = 0.0;
    problem->setInitialState(0, x0ss);

    // ── Cold-start initialization with pre-tilt (from reference) ──────────────
    const double T_g = HORIZON * th_init;
    const Eigen::Vector3d p0 = x0.segment(0, 3);
    const Eigen::Vector3d v0q = x0.segment(3, 3);
    const Eigen::Vector3d pt = tgt.pos(t_abs + T_g);
    const Eigen::Vector3d a_req = 2.0 * (pt - p0 - v0q * T_g) / (T_g * T_g);
    const Eigen::Vector3d fw = MASS * (a_req - GRAVITY);
    double fz_g = std::max(FMIN, std::min(FMAX, fw.norm()));
    const Eigen::Vector4d qt = tiltQuat(fw.normalized());

    Eigen::VectorXd u_g(NU);
    u_g.setZero();
    u_g(0) = fz_g;

    Eigen::VectorXd u0(NU_SS);
    u0.setZero();
    u0.segment(0, NU) = u_g;
    u0(IDX_THETA) = th_init;

    // Simulate from tilted quaternion (actual RK4 forward sim)
    Eigen::VectorXd sim = x0;
    sim.segment(6, 4) = qt;  // Apply pre-tilt

    for(int k=0; k<HORIZON; ++k) {
        problem->setInitialControl(k, u0);
        sim = rk4(sim, u_g, th_init);  // BUG 5 FIX: actual forward simulation
        Eigen::VectorXd xk(NX_SS);
        xk.segment(0, NX) = sim;
        xk(IDX_DT) = (k+1) * th_init;
        problem->setInitialState(k+1, xk);
    }

    // ── BUG 1 FIX: Apply warm-start controls AFTER cold-start rollout ─────────
    if (!prev_U.empty()) {
        for (int k = 0; k < HORIZON && k < static_cast<int>(prev_U.size()); ++k) {
            problem->setInitialControl(k, prev_U[k]);
        }
    }

    return problem;
}

} // namespace TrackingCircleOCP
