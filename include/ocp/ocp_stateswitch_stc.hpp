/// @file ocp_stateswitch_stc.hpp
/// @brief Clean CT-cSTC state-switch OCP for relative-frame landing on a moving target.
///
/// This file is the planner-side port of
/// `ALIPDDP-main/problem_examples/landing/quad_cf_tracking_rh_landing_stateswitch_stc.cpp`.
/// It is the paper's formulation: 15-D augmented state (13 physical + t + y
/// accumulator), terminal-AL equality y_N=0, single-mode stage cost with
/// sigma_cap/sigma_desc-blended follow-stage-land reference, and no
/// `phase_latch` / explicit `r_follow` plumbing in the descriptor extras.
///
/// The hardware-flown variant (with `phase_latch`, `CTCSLandingEnvelopeCon`,
/// dual-mode cost) is preserved verbatim in
/// `ocp_stateswitch_stc_hard.hpp` under namespace `StateswitchStcHardOCP`
/// and registry name `stateswitch_stc_hard`.

#pragma once

#include "optimal_control_problem.h"
#include "alipddp/alipddp.h"
#include "planner_core/types.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

namespace StateswitchStcOCP {

// ── Dimensions ────────────────────────────────────────────────────────────────
static constexpr int NX     = 13;
static constexpr int NU     = 4;
static constexpr int NX_SS  = 15;   // NX + IDX_DT + IDX_CTCS_Y
static constexpr int NU_SS  = 5;    // NU + IDX_THETA
static constexpr int IDX_DT      = 13;
static constexpr int IDX_CTCS_Y  = 14;
static constexpr int IDX_THETA   = 4;

// ── Horizon & replay ──────────────────────────────────────────────────────────
static constexpr int    HORIZON           = 40;
static constexpr double TH_INIT           = 0.07;
static constexpr double THL               = 0.05;
static constexpr double THH               = 0.12;
static constexpr int    DEFAULT_N_REPLAY  = 6;

// ── Vehicle / inertia ─────────────────────────────────────────────────────────
static constexpr double MASS    = 0.027;
static constexpr double IXX     = 1.66e-5;
static constexpr double IYY     = 1.66e-5;
static constexpr double IZZ     = 2.92e-5;
static constexpr double J_SCALE = 1.0 / IXX;
static constexpr double FMIN    = 0.08;
static constexpr double FMAX    = 0.6;
static constexpr double L_ARM   = 0.046;
static const     double TAU_MAX = L_ARM * (FMAX / 4.0 - FMIN / 4.0) * J_SCALE;
static constexpr double HOVER_THRUST = MASS * 9.81;

static const Eigen::Matrix3d J_B = [](){
    Eigen::Matrix3d J;
    J.setZero();
    J(0,0) = IXX * J_SCALE;
    J(1,1) = IYY * J_SCALE;
    J(2,2) = IZZ * J_SCALE;
    return J;
}();
static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

// ── Approach geometry & cost weights ──────────────────────────────────────────
static constexpr double GS_HALF_ANGLE_DEG = 25.0;
static const     double GS_TAN            = std::tan(GS_HALF_ANGLE_DEG * M_PI / 180.0);
static constexpr double GS_APEX_OFFSET    = 0.10;       // m, apex sits below target
static constexpr double VZ_LAND_MAX       = 0.5;        // m/s descent cap

static constexpr double TERM_W_POS_XY = 1800.0;
static constexpr double TERM_W_POS_Z  = 700.0;
static constexpr double TERM_W_VEL_XY = 900.0;
static constexpr double TERM_W_VEL_Z  = 600.0;
static constexpr double TERM_W_TILT   = 200.0;
static constexpr double TERM_W_OM     = 50.0;
static constexpr double STAGE_W_TIME  = 0.04;

// Control regularizers — Tikhonov on (thrust deviation from hover, moments).
static constexpr double STAGE_REG_THRUST = 0.25;
static constexpr double STAGE_REG_MOMENT = 1.0;

// Follow-point geometry (behind / above the target).
static constexpr double D_BEHIND  = 0.28;
static constexpr double DZ_ABOVE  = 1.6;
static constexpr double EPS_XY    = 0.35;   // unused in clean form but kept for parity
static constexpr double EPS_Z     = 0.35;
static constexpr double EPS_V     = 0.60;
static constexpr double W_SIGMA   = 15.0;

// Position / velocity reference weights for the blended stage cost.
static constexpr double W3_POS_XY = 180.0;
static constexpr double W3_POS_Z  = 220.0;
static constexpr double W3_VEL    = 120.0;
// Keep a mild velocity penalty active before sigma_desc engages, to damp
// pre-trigger chasing/back-and-forth around the moving follow reference.
static constexpr double VEL_GATE_MIN = 0.30;

// Path bounds (degenerate-trigger CT-cSTC consequences integrated into y).
static constexpr double THETA_PHASE0_DEG = 35.0;
static constexpr double OMEGA_PHASE0_MAX = 2.5;
static constexpr double SPD_PHASE0_MAX   = 3.5;

// Trigger-gated tight bounds (STC consequences).
static constexpr double THETA_STC_DEG     = 5.0;
static constexpr double ALPHA_THETA_STC   = 1.0;
static constexpr double OMEGA_STC_MAX     = 0.6;
static constexpr double SPD_STC_MAX       = 0.8;
static constexpr double T_MIN_AFT         = 0.21;
static constexpr double T_MAX_AFT         = 0.40;
static constexpr double T_MIN_WIDE        = 0.12;   // OR regime lower bound
static constexpr double T_MAX_WIDE        = 0.52;   // OR regime upper bound

static constexpr double CTCS_BETA             = 2e-2;
static constexpr double W_GLIDESLOPE          = 70.0;
static constexpr double V_THRUST_TRIG         = 1.0;
static constexpr double THETA_THRUST_TRIG_DEG = 15.0;

// Trigger geometry.
static const     double ALT_TRIG        = 0.45;
static const     double CAP_RADIUS      = 0.45;
static constexpr double CAP_COST_RADIUS = 0.50;
static constexpr double K_CAP           = 20.0;
static constexpr double Z_STAGE         = DZ_ABOVE;
static constexpr double CAP_DESC_RADIUS = 0.10;
static constexpr double K_DESC          = 40.0;

// ── sigma helpers (shared by stage cost & STC) ────────────────────────────────
// sigma_cap: lateral capture sigmoid around the target origin.
template<typename Scalar>
static Scalar sigmaLandVal(const Scalar& rxy2) {
    const Scalar g = Scalar(CAP_COST_RADIUS * CAP_COST_RADIUS) - rxy2;
    return Scalar(0.5) * (Scalar(1) + std::tanh(Scalar(K_CAP) * g));
}
template<typename Scalar>
static void sigmaLandGrad(const Scalar& rxy2, const Scalar& x0, const Scalar& x1,
                          Scalar& sig, Scalar& gx0, Scalar& gx1) {
    const Scalar g       = Scalar(CAP_COST_RADIUS * CAP_COST_RADIUS) - rxy2;
    const Scalar th      = std::tanh(Scalar(K_CAP) * g);
    sig     = Scalar(0.5) * (Scalar(1) + th);
    const Scalar dsig_dg = Scalar(0.5) * Scalar(K_CAP) * (Scalar(1) - th * th);
    gx0 = dsig_dg * Scalar(-2) * x0;
    gx1 = dsig_dg * Scalar(-2) * x1;
}

// sigma_desc: descent gate (pure lateral, fires inside CAP_DESC_RADIUS).
template<typename Scalar>
static Scalar sigmaDescVal(const Scalar& rxy2, const Scalar& /*z*/) {
    const Scalar g = Scalar(CAP_DESC_RADIUS * CAP_DESC_RADIUS) - rxy2;
    return Scalar(0.5) * (Scalar(1) + std::tanh(Scalar(K_DESC) * g));
}
template<typename Scalar>
static void sigmaDescGrad(const Scalar& rxy2, const Scalar& x0, const Scalar& x1,
                          const Scalar& /*z*/, Scalar& sig,
                          Scalar& gx0, Scalar& gx1, Scalar& gz) {
    const Scalar g    = Scalar(CAP_DESC_RADIUS * CAP_DESC_RADIUS) - rxy2;
    const Scalar th   = std::tanh(Scalar(K_DESC) * g);
    sig = Scalar(0.5) * (Scalar(1) + th);
    const Scalar dsig = Scalar(0.5) * Scalar(K_DESC) * (Scalar(1) - th * th);
    gx0 = dsig * Scalar(-2) * x0;
    gx1 = dsig * Scalar(-2) * x1;
    gz  = Scalar(0);
}

// ── Forward declaration of the CT-cSTC integrand ──────────────────────────────
template<typename Scalar>
static Scalar evalCtcsIntegrand(const Vector<Scalar>& x,
                                const Vector<Scalar>& u);

// ── Target predictor (planner-side: jerk/snap available, but only predictAccel
//    is consumed inside the dynamics RK4) ──────────────────────────────────────
struct TargetPredictor {
    Eigen::Vector3d position     = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity     = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d jerk         = Eigen::Vector3d::Zero();
    Eigen::Vector3d snap         = Eigen::Vector3d::Zero();

    Eigen::Vector3d predictPos(double dt) const {
        const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
        return position + velocity * dt + 0.5 * acceleration * dt2 +
               (1.0 / 6.0) * jerk * dt3 + (1.0 / 24.0) * snap * dt4;
    }
    Eigen::Vector3d predictVel(double dt) const {
        const double dt2 = dt * dt, dt3 = dt2 * dt;
        return velocity + acceleration * dt + 0.5 * jerk * dt2 +
               (1.0 / 6.0) * snap * dt3;
    }
    Eigen::Vector3d predictAccel(double dt) const {
        const double dt2 = dt * dt;
        return acceleration + jerk * dt + 0.5 * snap * dt2;
    }
};

// ── Augmented relative dynamics with y-accumulator (15-D state) ───────────────
template<typename Scalar>
class Quad6DOFVarTimeRelativePred : public Quad6DOFVarTimeRelative<Scalar> {
    std::shared_ptr<TargetPredictor> pred_;

    double ctcsRate(const Eigen::VectorXd& x_phys,
                    const Eigen::VectorXd& u_full) const {
        Eigen::VectorXd x_aug(NX_SS);
        x_aug.setZero();
        x_aug.segment(0, this->NX_PHYS) = x_phys;
        return evalCtcsIntegrand<double>(x_aug, u_full);
    }

public:
    explicit Quad6DOFVarTimeRelativePred(std::shared_ptr<TargetPredictor> pred)
        : Quad6DOFVarTimeRelative<Scalar>(), pred_(std::move(pred)) {
        this->dim_x = NX_SS;
    }

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd     = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud     = u.segment(0, this->NU_PHYS).template cast<double>();
        auto u_full = u.template cast<double>();
        const double Th   = static_cast<double>(u(IDX_THETA));
        const double tabs = static_cast<double>(x(IDX_DT));

        const Eigen::Vector3d a1 = pred_->predictAccel(tabs);
        auto k1 = this->xdot_impl(xd, ud, a1);
        const double h1 = ctcsRate(xd, u_full);

        const Eigen::Vector3d a2 = pred_->predictAccel(tabs + 0.5 * Th);
        Eigen::VectorXd x2 = xd + 0.5 * Th * k1;
        auto k2 = this->xdot_impl(x2, ud, a2);
        const double h2 = ctcsRate(x2, u_full);

        const Eigen::Vector3d a3 = pred_->predictAccel(tabs + 0.5 * Th);
        Eigen::VectorXd x3 = xd + 0.5 * Th * k2;
        auto k3 = this->xdot_impl(x3, ud, a3);
        const double h3 = ctcsRate(x3, u_full);

        const Eigen::Vector3d a4 = pred_->predictAccel(tabs + Th);
        Eigen::VectorXd x4 = xd + Th * k3;
        auto k4 = this->xdot_impl(x4, ud, a4);
        const double h4 = ctcsRate(x4, u_full);

        Eigen::VectorXd xn = xd + (Th / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        xn.segment(6, 4).normalize();
        const double yn = static_cast<double>(x(IDX_CTCS_Y)) +
                          (Th / 6.0) * (h1 + 2.0 * h2 + 2.0 * h3 + h4);

        const_cast<Quad6DOFVarTimeRelativePred*>(this)
            ->setTargetAccel(pred_->predictAccel(tabs + 0.5 * Th));

        Vector<Scalar> res(NX_SS);
        res.setZero();
        res.segment(0, this->NX_PHYS) = xn.template cast<Scalar>();
        res(IDX_DT)     = x(IDX_DT) + u(IDX_THETA);
        res(IDX_CTCS_Y) = Scalar(yn);
        return res;
    }

    Matrix<Scalar> fx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        const double Th   = static_cast<double>(u(IDX_THETA));
        const double tabs = static_cast<double>(x(IDX_DT));
        const Eigen::Vector3d amid = pred_->predictAccel(tabs + 0.5 * Th);
        auto j = this->rk4_jac_impl(xd, ud, Th, amid);

        Matrix<Scalar> Fx = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        Fx.block(0, 0, this->NX_PHYS, this->NX_PHYS) = j.dxdx.template cast<Scalar>();
        Fx(IDX_DT, IDX_DT)           = Scalar(1.0);
        Fx(IDX_CTCS_Y, IDX_CTCS_Y)   = Scalar(1.0);

        // Central-difference the y-accumulator row w.r.t. all physical/time states.
        const double eps = 1e-6;
        for (int i = 0; i < NX_SS; ++i) {
            if (i == IDX_CTCS_Y) continue;
            Vector<Scalar> xp = x, xm = x;
            xp(i) += Scalar(eps);
            xm(i) -= Scalar(eps);
            const Scalar yp = f(xp, u)(IDX_CTCS_Y);
            const Scalar ym = f(xm, u)(IDX_CTCS_Y);
            Fx(IDX_CTCS_Y, i) = (yp - ym) / Scalar(2.0 * eps);
        }
        const_cast<Quad6DOFVarTimeRelativePred*>(this)->setTargetAccel(amid);
        return Fx;
    }

    Matrix<Scalar> fu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        const double Th   = static_cast<double>(u(IDX_THETA));
        const double tabs = static_cast<double>(x(IDX_DT));
        const Eigen::Vector3d amid = pred_->predictAccel(tabs + 0.5 * Th);
        auto j = this->rk4_jac_impl(xd, ud, Th, amid);

        Matrix<Scalar> Fu = Matrix<Scalar>::Zero(NX_SS, NU_SS);
        Fu.block(0, 0, this->NX_PHYS, this->NU_PHYS) = j.dxdu.template cast<Scalar>();
        Fu.block(0, IDX_THETA, this->NX_PHYS, 1)     = j.dxdT.template cast<Scalar>();
        Fu(IDX_DT, IDX_THETA) = Scalar(1.0);

        const double eps = 1e-6;
        for (int i = 0; i < NU_SS; ++i) {
            Vector<Scalar> up = u, um = u;
            up(i) += Scalar(eps);
            um(i) -= Scalar(eps);
            const Scalar yp = f(x, up)(IDX_CTCS_Y);
            const Scalar ym = f(x, um)(IDX_CTCS_Y);
            Fu(IDX_CTCS_Y, i) = (yp - ym) / Scalar(2.0 * eps);
        }
        const_cast<Quad6DOFVarTimeRelativePred*>(this)->setTargetAccel(amid);
        return Fu;
    }

    // 15-dim propagate for warm-start rollouts (tracks IDX_DT and y).
    Eigen::VectorXd propagate15(const Eigen::VectorXd& x15,
                                const Eigen::VectorXd& u_phys,
                                double Th) const {
        const double tabs = x15(IDX_DT);
        auto xd = x15.segment(0, this->NX_PHYS);
        Eigen::VectorXd u_full(NU_SS);
        u_full.setZero();
        u_full.segment(0, NU) = u_phys;
        u_full(IDX_THETA) = Th;

        const Eigen::Vector3d a1 = pred_->predictAccel(tabs);
        auto k1 = this->xdot_impl(xd, u_phys, a1);
        const double h1 = ctcsRate(xd, u_full);
        const Eigen::Vector3d a2 = pred_->predictAccel(tabs + 0.5 * Th);
        Eigen::VectorXd x2 = xd + 0.5 * Th * k1;
        auto k2 = this->xdot_impl(x2, u_phys, a2);
        const double h2 = ctcsRate(x2, u_full);
        const Eigen::Vector3d a3 = pred_->predictAccel(tabs + 0.5 * Th);
        Eigen::VectorXd x3 = xd + 0.5 * Th * k2;
        auto k3 = this->xdot_impl(x3, u_phys, a3);
        const double h3 = ctcsRate(x3, u_full);
        const Eigen::Vector3d a4 = pred_->predictAccel(tabs + Th);
        Eigen::VectorXd x4 = xd + Th * k3;
        auto k4 = this->xdot_impl(x4, u_phys, a4);
        const double h4 = ctcsRate(x4, u_full);

        Eigen::VectorXd xn15(NX_SS);
        xn15.setZero();
        xn15.segment(0, this->NX_PHYS) = xd + (Th / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        xn15.segment(6, 4).normalize();
        xn15(IDX_DT)     = x15(IDX_DT) + Th;
        xn15(IDX_CTCS_Y) = x15(IDX_CTCS_Y) + (Th / 6.0) * (h1 + 2.0 * h2 + 2.0 * h3 + h4);
        return xn15;
    }
};

// ── Follow-point helper (behind & above the moving target) ────────────────────
inline Eigen::Vector3d computeFollowRef(const Eigen::Vector3d& target_velocity) {
    Eigen::Vector3d vhat(1.0, 0.0, 0.0);
    const double speed = target_velocity.head<2>().norm();
    if (speed > 1e-6) {
        vhat = Eigen::Vector3d(target_velocity.x() / speed,
                               target_velocity.y() / speed, 0.0);
    }
    return -D_BEHIND * vhat + Eigen::Vector3d(0.0, 0.0, DZ_ABOVE);
}

// ── Descriptor extras passed via OCPCreateArgs ────────────────────────────────
// Clean form: just the predictor and the cost-side r_follow reference. No
// `phase_latch`, no latch-management state — the STC handles all phase logic.
struct StateswitchStcExtra {
    std::shared_ptr<TargetPredictor> predictor;
    Eigen::Vector3d r_follow = Eigen::Vector3d(-D_BEHIND, 0.0, DZ_ABOVE);
    bool target_valid        = false;
};

// ── CT-cSTC integrand (single source of truth for trigger × consequence) ─────
template<typename Scalar>
static Scalar posPart(const Scalar& v) {
    return (v > Scalar(0)) ? v : Scalar(0);
}

template<typename Scalar>
struct CtcsIntegrandTerms {
    Scalar a_altitude_state  = Scalar(0);
    Scalar a_thrust          = Scalar(0);
    Scalar loose_tilt        = Scalar(0);
    Scalar loose_omega       = Scalar(0);
    Scalar speed             = Scalar(0);
    Scalar tilt              = Scalar(0);
    Scalar omega             = Scalar(0);
    Scalar glideslope        = Scalar(0);
    Scalar thrust_hi         = Scalar(0);
    Scalar thrust_lo         = Scalar(0);
    Scalar thrust_wide_hi    = Scalar(0);
    Scalar thrust_wide_lo    = Scalar(0);
    Scalar path_value         = Scalar(0);
    Scalar staging_value      = Scalar(0);
    Scalar landing_value      = Scalar(0);
    Scalar thrust_tight_value = Scalar(0);
    Scalar thrust_wide_value  = Scalar(0);
    Scalar stc_value          = Scalar(0);
    Scalar value              = Scalar(0);
};

template<typename Scalar>
static CtcsIntegrandTerms<Scalar> evalCtcsIntegrandTerms(const Vector<Scalar>& x,
                                                         const Vector<Scalar>& u) {
    CtcsIntegrandTerms<Scalar> out;

    // Triggers — pure state, normalized to [0,1] for numerical conditioning.
    const Scalar trig_alt = posPart(Scalar(ALT_TRIG) - x(2)) / Scalar(ALT_TRIG);
    const Scalar rxy2     = x(0) * x(0) + x(1) * x(1);
    const Scalar cap2     = Scalar(CAP_RADIUS * CAP_RADIUS);
    const Scalar trig_cap = posPart(cap2 - rxy2) / cap2;
    out.a_altitude_state  = trig_alt * trig_cap;

    const Scalar v2     = x.template segment<3>(3).squaredNorm();
    const Scalar qperp2 = x(7) * x(7) + x(8) * x(8);
    const Scalar om2    = x.template segment<3>(10).squaredNorm();
    const Scalar rxy    = std::sqrt(rxy2 + Scalar(1e-12));
    const Scalar v_spd  = std::sqrt(v2 + Scalar(1e-12));

    // H_path: always-on tilt and omega bounds (degenerate-trigger CT-cSTC).
    const Scalar loose_tilt_lim =
        Scalar(std::pow(std::sin(0.5 * THETA_PHASE0_DEG * M_PI / 180.0), 2));
    out.loose_tilt  = posPart(qperp2 - loose_tilt_lim);
    out.loose_omega = posPart(om2 - Scalar(OMEGA_PHASE0_MAX * OMEGA_PHASE0_MAX));

    // H_stc consequences.
    const Scalar tilt_lim = Scalar(
        std::pow(std::sin(0.5 * ALPHA_THETA_STC * THETA_STC_DEG * M_PI / 180.0), 2));
    out.speed      = posPart(v2     - Scalar(SPD_STC_MAX   * SPD_STC_MAX));
    out.tilt       = posPart(qperp2 - tilt_lim);
    out.omega      = posPart(om2    - Scalar(OMEGA_STC_MAX * OMEGA_STC_MAX));
    const Scalar gs_val = rxy - Scalar(GS_TAN) * (x(2) + Scalar(GS_APEX_OFFSET));
    out.glideslope = posPart(gs_val);

    const Scalar tilt_trig_thresh =
        Scalar(std::pow(std::sin(0.5 * THETA_THRUST_TRIG_DEG * M_PI / 180.0), 2));
    const Scalar trig_slow    = posPart(Scalar(V_THRUST_TRIG) - v_spd);
    const Scalar trig_upright = posPart(tilt_trig_thresh - qperp2);
    out.a_thrust  = out.a_altitude_state * trig_slow * trig_upright;
    out.thrust_hi = posPart(u(0) - Scalar(T_MAX_AFT));
    out.thrust_lo = posPart(Scalar(T_MIN_AFT) - u(0));

    const Scalar trig_fast        = posPart(v_spd  - Scalar(V_THRUST_TRIG));
    const Scalar trig_tilted      = posPart(qperp2 - tilt_trig_thresh);
    const Scalar wide_trigger_sum = trig_fast * trig_fast + trig_tilted * trig_tilted;
    out.thrust_wide_hi = posPart(u(0) - Scalar(T_MAX_WIDE));
    out.thrust_wide_lo = posPart(Scalar(T_MIN_WIDE) - u(0));

    const Scalar path_sum  = out.loose_tilt  * out.loose_tilt  +
                             out.loose_omega * out.loose_omega;
    const Scalar thrust_tight_sum = out.thrust_hi * out.thrust_hi +
                                    out.thrust_lo * out.thrust_lo;
    const Scalar landing_sum = out.speed * out.speed + out.tilt * out.tilt +
                               out.omega * out.omega +
                               Scalar(W_GLIDESLOPE) * out.glideslope * out.glideslope +
                               thrust_tight_sum;
    const Scalar thrust_wide_sum = out.thrust_wide_hi * out.thrust_wide_hi +
                                   out.thrust_wide_lo * out.thrust_wide_lo;

    out.path_value = path_sum;

    {
        const Scalar trig_not_cap      = Scalar(1) - sigmaLandVal(rxy2);
        const Scalar stage_height_viol = posPart(Scalar(Z_STAGE) - x(2));
        out.staging_value = trig_not_cap * trig_not_cap *
                            stage_height_viol * stage_height_viol;
    }
    out.landing_value = out.a_altitude_state * out.a_altitude_state * landing_sum;
    out.thrust_tight_value = Scalar(0); // folded into landing_value
    out.thrust_wide_value  = out.a_altitude_state * out.a_altitude_state *
                             wide_trigger_sum * thrust_wide_sum;

    out.stc_value = out.staging_value + out.landing_value +
                    out.thrust_tight_value + out.thrust_wide_value;
    out.value     = out.path_value + out.stc_value;
    return out;
}

template<typename Scalar>
static Scalar evalCtcsIntegrand(const Vector<Scalar>& x,
                                const Vector<Scalar>& u) {
    return evalCtcsIntegrandTerms(x, u).value;
}

// ── Single-mode stage cost (follow → stage → land blended by sigma_cap/desc) ─
template<typename Scalar>
class CTcSTCStageCost : public StageCostBase<Scalar> {
    Scalar eps_;
    Scalar w_pos_xy_, w_pos_z_, w_vel_;
    Scalar rf_x_, rf_y_, rf_z_;

public:
    CTcSTCStageCost(double eps, double w_pos_xy, double w_pos_z, double w_vel,
                    const Eigen::Vector3d& r_follow)
        : eps_(Scalar(eps)),
          w_pos_xy_(Scalar(w_pos_xy)), w_pos_z_(Scalar(w_pos_z)), w_vel_(Scalar(w_vel)),
          rf_x_(Scalar(r_follow(0))), rf_y_(Scalar(r_follow(1))), rf_z_(Scalar(r_follow(2))) {}

    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const Scalar rxy2 = x(0) * x(0) + x(1) * x(1);
        const Scalar sc   = sigmaLandVal(rxy2);
        const Scalar sd   = sigmaDescVal(rxy2, x(2));
        const Scalar vel_gate = Scalar(VEL_GATE_MIN) +
                                (Scalar(1) - Scalar(VEL_GATE_MIN)) * sd;
        const Scalar oms  = Scalar(1) - sd;
        const Scalar oms_p = oms * oms * oms * oms;

        const Scalar rref_x = (Scalar(1) - sc) * rf_x_;
        const Scalar rref_y = (Scalar(1) - sc) * rf_y_;
        const Scalar rref_z = (Scalar(1) - sc) * rf_z_ + sc * oms_p * Scalar(Z_STAGE);

        const Scalar drx = x(0) - rref_x;
        const Scalar dry = x(1) - rref_y;
        const Scalar drz = x(2) - rref_z;
        const Scalar v2  = x.template segment<3>(3).squaredNorm();

        const Scalar pos_cost = w_pos_xy_ * (drx * drx + dry * dry) + w_pos_z_ * drz * drz;
        const Scalar vel_cost = vel_gate * w_vel_ * v2;
        const Scalar reg      = eps_ * (v2 + x.template segment<3>(10).squaredNorm());
        const Scalar du0      = u(0) - Scalar(HOVER_THRUST);
        const Scalar u_reg    = Scalar(STAGE_REG_THRUST) * du0 * du0 +
                                Scalar(STAGE_REG_MOMENT) * u.template segment<3>(1).squaredNorm();
        return Scalar(STAGE_W_TIME) * u(IDX_THETA) + reg + pos_cost + vel_cost + u_reg;
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        const Scalar rxy2 = x(0) * x(0) + x(1) * x(1);
        Scalar sc, sc_gx0, sc_gx1;
        sigmaLandGrad(rxy2, x(0), x(1), sc, sc_gx0, sc_gx1);
        Scalar sd, sd_gx0, sd_gx1, sd_gz;
        sigmaDescGrad(rxy2, x(0), x(1), x(2), sd, sd_gx0, sd_gx1, sd_gz);

        const Scalar Zs    = Scalar(Z_STAGE);
        const Scalar vel_mix = Scalar(1) - Scalar(VEL_GATE_MIN);
        const Scalar vel_gate = Scalar(VEL_GATE_MIN) + vel_mix * sd;
        const Scalar oms   = Scalar(1) - sd;
        const Scalar oms2  = oms * oms;
        const Scalar oms3  = oms2 * oms;
        const Scalar oms_p = oms2 * oms2;
        const Scalar rref_x = (Scalar(1) - sc) * rf_x_;
        const Scalar rref_y = (Scalar(1) - sc) * rf_y_;
        const Scalar rref_z = (Scalar(1) - sc) * rf_z_ + sc * oms_p * Zs;

        const Scalar drx = x(0) - rref_x;
        const Scalar dry = x(1) - rref_y;
        const Scalar drz = x(2) - rref_z;
        const Scalar v2  = x.template segment<3>(3).squaredNorm();

        const Scalar ddrx_dx0 = Scalar(1) + sc_gx0 * rf_x_;
        const Scalar ddrx_dx1 = sc_gx1 * rf_x_;
        const Scalar ddry_dx0 = sc_gx0 * rf_y_;
        const Scalar ddry_dx1 = Scalar(1) + sc_gx1 * rf_y_;
        const Scalar ddrz_dx0 = sc_gx0 * (rf_z_ - oms_p * Zs) +
                                Scalar(4) * sc * oms3 * sd_gx0 * Zs;
        const Scalar ddrz_dx1 = sc_gx1 * (rf_z_ - oms_p * Zs) +
                                Scalar(4) * sc * oms3 * sd_gx1 * Zs;
        const Scalar ddrz_dz  = Scalar(1) + Scalar(4) * sc * oms3 * sd_gz * Zs;

        Vector<Scalar> g = Vector<Scalar>::Zero(NX_SS);
        g(0) = Scalar(2) * w_pos_xy_ * (drx * ddrx_dx0 + dry * ddry_dx0) +
               Scalar(2) * w_pos_z_ * drz * ddrz_dx0;
        g(1) = Scalar(2) * w_pos_xy_ * (drx * ddrx_dx1 + dry * ddry_dx1) +
               Scalar(2) * w_pos_z_ * drz * ddrz_dx1;
        g(2) = Scalar(2) * w_pos_z_ * drz * ddrz_dz;
        g(0) += vel_mix * sd_gx0 * w_vel_ * v2;
        g(1) += vel_mix * sd_gx1 * w_vel_ * v2;
        g(2) += vel_mix * sd_gz  * w_vel_ * v2;
        g.template segment<3>(3) =
            (Scalar(2) * eps_ + Scalar(2) * vel_gate * w_vel_) *
            x.template segment<3>(3);
        g.template segment<3>(10) = Scalar(2) * eps_ * x.template segment<3>(10);
        return g;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x;
        Vector<Scalar> g = Vector<Scalar>::Zero(NU_SS);
        g(0) = Scalar(2) * Scalar(STAGE_REG_THRUST) * (u(0) - Scalar(HOVER_THRUST));
        g.template segment<3>(1) = Scalar(2) * Scalar(STAGE_REG_MOMENT) *
                                   u.template segment<3>(1);
        g(IDX_THETA) = Scalar(STAGE_W_TIME);
        return g;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)u;
        const Scalar rxy2 = x(0) * x(0) + x(1) * x(1);
        Scalar sc, sc_gx0, sc_gx1;
        sigmaLandGrad(rxy2, x(0), x(1), sc, sc_gx0, sc_gx1);
        Scalar sd, sd_gx0, sd_gx1, sd_gz;
        sigmaDescGrad(rxy2, x(0), x(1), x(2), sd, sd_gx0, sd_gx1, sd_gz);
        const Scalar Zs   = Scalar(Z_STAGE);
        const Scalar vel_mix = Scalar(1) - Scalar(VEL_GATE_MIN);
        const Scalar oms  = Scalar(1) - sd;
        const Scalar oms3 = oms * oms * oms;
        const Scalar oms_p = oms3 * oms;

        const Scalar ddrx_dx0 = Scalar(1) + sc_gx0 * rf_x_;
        const Scalar ddrx_dx1 = sc_gx1 * rf_x_;
        const Scalar ddry_dx0 = sc_gx0 * rf_y_;
        const Scalar ddry_dx1 = Scalar(1) + sc_gx1 * rf_y_;
        const Scalar ddrz_dx0 = sc_gx0 * (rf_z_ - oms_p * Zs) +
                                Scalar(4) * sc * oms3 * sd_gx0 * Zs;
        const Scalar ddrz_dx1 = sc_gx1 * (rf_z_ - oms_p * Zs) +
                                Scalar(4) * sc * oms3 * sd_gx1 * Zs;
        const Scalar ddrz_dz  = Scalar(1) + Scalar(4) * sc * oms3 * sd_gz * Zs;

        Matrix<Scalar> H = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        H(0, 0) = Scalar(2) * w_pos_xy_ * (ddrx_dx0 * ddrx_dx0 + ddry_dx0 * ddry_dx0) +
                  Scalar(2) * w_pos_z_ * ddrz_dx0 * ddrz_dx0;
        H(1, 1) = Scalar(2) * w_pos_xy_ * (ddrx_dx1 * ddrx_dx1 + ddry_dx1 * ddry_dx1) +
                  Scalar(2) * w_pos_z_ * ddrz_dx1 * ddrz_dx1;
        H(2, 2) = Scalar(2) * w_pos_z_ * ddrz_dz * ddrz_dz;
        H(0, 1) = H(1, 0) = Scalar(2) * w_pos_xy_ *
                            (ddrx_dx0 * ddrx_dx1 + ddry_dx0 * ddry_dx1) +
                            Scalar(2) * w_pos_z_ * ddrz_dx0 * ddrz_dx1;
        H(0, 2) = H(2, 0) = Scalar(2) * w_pos_z_ * ddrz_dx0 * ddrz_dz;
        H(1, 2) = H(2, 1) = Scalar(2) * w_pos_z_ * ddrz_dx1 * ddrz_dz;

        for (int i = 3; i < 6;  ++i) {
            const Scalar vel_gate = Scalar(VEL_GATE_MIN) + vel_mix * sd;
            H(i, i) = Scalar(2) * eps_ + Scalar(2) * vel_gate * w_vel_;
        }
        for (int i = 10; i < 13; ++i) H(i, i) = Scalar(2) * eps_;
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NU_SS, NU_SS);
        H(0, 0) = Scalar(2) * Scalar(STAGE_REG_THRUST);
        H(1, 1) = H(2, 2) = H(3, 3) = Scalar(2) * Scalar(STAGE_REG_MOMENT);
        return H;
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        (void)x; (void)u;
        return Matrix<Scalar>::Zero(NX_SS, NU_SS);
    }
};

// ── Terminal cost ─────────────────────────────────────────────────────────────
template<typename Scalar>
class RelTermCost : public TerminalCostBase<Scalar> {
    double wp_xy_, wp_z_, wv_xy_, wv_z_, watt_, wom_, vz_ref_;

public:
    explicit RelTermCost(double wp_xy = 500.0, double wp_z = 500.0,
                         double wv_xy = 50.0,  double wv_z = 100.0,
                         double watt  = 200.0, double wom = 50.0,
                         double vz_ref = -0.1)
        : wp_xy_(wp_xy), wp_z_(wp_z), wv_xy_(wv_xy), wv_z_(wv_z),
          watt_(watt), wom_(wom), vz_ref_(vz_ref) {}

    static void bodyZHorizontal(const Vector<Scalar>& x, Scalar& bx, Scalar& by) {
        const Scalar qw = x(6), qx = x(7), qy = x(8), qz = x(9);
        bx = Scalar(2) * (qx * qz + qw * qy);
        by = Scalar(2) * (qy * qz - qw * qx);
    }
    static Matrix<Scalar> bodyZHorizontalJacobian(const Vector<Scalar>& x) {
        const Scalar qw = x(6), qx = x(7), qy = x(8), qz = x(9);
        Matrix<Scalar> J = Matrix<Scalar>::Zero(2, 4);
        J(0, 0) = Scalar(2) * qy;
        J(0, 1) = Scalar(2) * qz;
        J(0, 2) = Scalar(2) * qw;
        J(0, 3) = Scalar(2) * qx;
        J(1, 0) = Scalar(-2) * qx;
        J(1, 1) = Scalar(-2) * qw;
        J(1, 2) = Scalar(2) * qz;
        J(1, 3) = Scalar(2) * qy;
        return J;
    }

    Scalar p(const Vector<Scalar>& x) const override {
        const Scalar ep   = x.template segment<2>(0).squaredNorm();
        const Scalar ez   = x(2) * x(2);
        const Scalar evxy = x.template segment<2>(3).squaredNorm();
        const Scalar evz  = (x(5) - Scalar(vz_ref_)) * (x(5) - Scalar(vz_ref_));
        Scalar bx, by; bodyZHorizontal(x, bx, by);
        const Scalar eom  = x.template segment<3>(10).squaredNorm();
        return Scalar(0.5) *
               (Scalar(wp_xy_) * ep + Scalar(wp_z_) * ez +
                Scalar(wv_xy_) * evxy + Scalar(wv_z_) * evz +
                Scalar(watt_) * (bx * bx + by * by) +
                Scalar(wom_) * eom);
    }
    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.segment(0, 2) = Scalar(wp_xy_) * x.segment(0, 2);
        g(2)            = Scalar(wp_z_) * x(2);
        g.segment(3, 2) = Scalar(wv_xy_) * x.segment(3, 2);
        g(5)            = Scalar(wv_z_) * (x(5) - Scalar(vz_ref_));
        Scalar bx, by; bodyZHorizontal(x, bx, by);
        Matrix<Scalar> Jtilt = bodyZHorizontalJacobian(x);
        Vector<Scalar> etilt(2); etilt << bx, by;
        g.segment(6, 4)  += Scalar(watt_) * Jtilt.transpose() * etilt;
        g.segment(10, 3) = Scalar(wom_) * x.segment(10, 3);
        return g;
    }
    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H(0, 0) = H(1, 1) = Scalar(wp_xy_);
        H(2, 2)           = Scalar(wp_z_);
        H(3, 3) = H(4, 4) = Scalar(wv_xy_);
        H(5, 5)           = Scalar(wv_z_);
        Matrix<Scalar> Jtilt = bodyZHorizontalJacobian(x);
        H.block(6, 6, 4, 4) += Scalar(watt_) * Jtilt.transpose() * Jtilt;
        H(10, 10) = H(11, 11) = H(12, 12) = Scalar(wom_);
        return H;
    }
};

// ── Stage constraints ─────────────────────────────────────────────────────────
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
    ThetaBounds(double lo, double hi)
        : lo_(static_cast<Scalar>(lo)), hi_(static_cast<Scalar>(hi)) {
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
class ZFloorCon : public StageConstraintBase<Scalar> {
public:
    ZFloorCon() { this->constraint_type = ConstraintType::NO; this->dim_c = 1; }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        return (Vector<Scalar>(1) << -x(2)).finished();
    }
    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, x.size());
        J(0, 2) = Scalar(-1);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>& u) const override {
        return Matrix<Scalar>::Zero(1, u.size());
    }
};

template<typename Scalar>
class GeneralSpeedCon : public StageConstraintBase<Scalar> {
public:
    GeneralSpeedCon() {
        this->constraint_type = ConstraintType::SOC;
        this->dim_c = 4;
    }
    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>&) const override {
        Vector<Scalar> cn(4);
        cn << Scalar(SPD_PHASE0_MAX), x(3), x(4), x(5);
        return -cn;
    }
    Matrix<Scalar> cx(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(4, NX_SS);
        J(1, 3) = Scalar(-1);
        J(2, 4) = Scalar(-1);
        J(3, 5) = Scalar(-1);
        return J;
    }
    Matrix<Scalar> cu(const Vector<Scalar>&, const Vector<Scalar>&) const override {
        return Matrix<Scalar>::Zero(4, NU_SS);
    }
};

// ── Terminal AL equality: y_N = 0 ─────────────────────────────────────────────
template<typename Scalar>
class CtcsAccumulatorTerminalCon : public TerminalConstraintBase<Scalar> {
public:
    CtcsAccumulatorTerminalCon() {
        this->constraint_type = ConstraintType::EQ;
        this->dim_cT = 1;
    }
    Vector<Scalar> cT(const Vector<Scalar>& x) const override {
        return (Vector<Scalar>(1) << x(IDX_CTCS_Y)).finished();
    }
    Matrix<Scalar> cTx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(1, x.size());
        J(0, IDX_CTCS_Y) = Scalar(1);
        return J;
    }
};

// ── Solver params (match the cpp's clean version) ─────────────────────────────
inline Param getSolverParams() {
    Param p;
    p.reg1_min = 1e-3;
    p.reg2_min = 0.1;
    p.mu_mul   = 0.1;
    p.rho      = 5.0;
    p.rhoT     = 5.0;
    p.rho_mul  = 10.0;
    p.tolerance = 1e-6;
    p.max_iter  = 800;
    p.is_quaternion_in_state = false;
    p.use_ddp_terms = false;
    return p;
}

// ── OCP construction ──────────────────────────────────────────────────────────
inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0_rel,
    const Eigen::VectorXd& /*terminal_state*/,
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& prev_X = {},
    const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero(),
    const std::vector<Eigen::MatrixXd>& prev_K = {},
    std::shared_ptr<TargetPredictor> predictor = nullptr,
    const Eigen::Vector3d& r_follow = Eigen::Vector3d(-D_BEHIND, 0.0, DZ_ABOVE)) {

    auto problem = std::make_shared<OptimalControlProblem<double>>(HORIZON);

    if (!predictor) {
        predictor = std::make_shared<TargetPredictor>();
        predictor->acceleration = target_accel;
    }

    auto dyn = std::make_shared<Quad6DOFVarTimeRelativePred<double>>(predictor);
    dyn->setMass(MASS);
    dyn->setGravity(GRAVITY);
    dyn->setJb(J_B);
    dyn->setTargetAccel(predictor->predictAccel(0.0));

    auto tcost = std::make_shared<RelTermCost<double>>(
        TERM_W_POS_XY, TERM_W_POS_Z,
        TERM_W_VEL_XY, TERM_W_VEL_Z,
        TERM_W_TILT, TERM_W_OM, -0.2);
    auto cfmin  = std::make_shared<FminCon<double>>();
    auto cfmax  = std::make_shared<FmaxCon<double>>();
    auto cmom   = std::make_shared<MomentCon<double>>();
    auto cth    = std::make_shared<ThetaBounds<double>>(THL, THH);
    auto czfl   = std::make_shared<ZFloorCon<double>>();
    auto cspeed = std::make_shared<GeneralSpeedCon<double>>();
    auto stage_cost = std::make_shared<CTcSTCStageCost<double>>(
        8.0, W3_POS_XY, W3_POS_Z, W3_VEL, r_follow);

    for (int k = 0; k < HORIZON; ++k) {
        problem->setStageDynamics(k, dyn);
        problem->setStageCost(k, stage_cost);
        problem->addStageConstraint(k, cfmin);
        problem->addStageConstraint(k, cfmax);
        problem->addStageConstraint(k, cmom);
        problem->addStageConstraint(k, cth);
        problem->addStageConstraint(k, czfl);
        problem->addStageConstraint(k, cspeed);
    }
    problem->setTerminalCost(tcost);
    problem->addTerminalConstraint(std::make_shared<CtcsAccumulatorTerminalCon<double>>());

    // 15-D initial state: (physical, IDX_DT=0, IDX_CTCS_Y=0).
    Eigen::VectorXd x0ss(NX_SS);
    x0ss.setZero();
    x0ss.segment(0, NX) = x0_rel;
    x0ss(IDX_DT)        = 0.0;
    x0ss(IDX_CTCS_Y)    = 0.0;
    problem->setInitialState(0, x0ss);

    Eigen::VectorXd sim15(NX_SS);
    sim15.setZero();
    sim15.segment(0, NX) = x0_rel;

    const bool use_feedback =
        (!prev_U.empty() && !prev_X.empty() && !prev_K.empty() &&
         (int)prev_U.size() >= HORIZON &&
         (int)prev_X.size() > HORIZON &&
         (int)prev_K.size() >= HORIZON);

    for (int k = 0; k < HORIZON; ++k) {
        Eigen::VectorXd u0(NU_SS);
        u0.setZero();

        if (use_feedback) {
            Eigen::VectorXd px = Eigen::VectorXd::Zero(NX_SS);
            const int n = std::min<int>(prev_X[k].size(), NX_SS);
            px.head(n) = prev_X[k].head(n);
            Eigen::VectorXd dx = sim15 - px;
            const Eigen::VectorXd u_shift = prev_U[k];
            if (prev_K[k].cols() == dx.size()) {
                u0 = prev_U[k] + prev_K[k] * dx;
            } else {
                u0 = prev_U[k] + prev_K[k] * dx.head(prev_K[k].cols());
            }
            const bool wild =
                !u0.allFinite() ||
                std::abs(u0(1)) > 2.0 * TAU_MAX ||
                std::abs(u0(2)) > 2.0 * TAU_MAX ||
                std::abs(u0(3)) > 2.0 * TAU_MAX;
            if (wild) u0 = u_shift;
        } else if (!prev_U.empty() && k < (int)prev_U.size()) {
            u0 = prev_U[k];
        } else {
            // Cold warm-start: gentle aim at the follow point.
            const Eigen::Vector3d p0 = sim15.head(3);
            const Eigen::Vector3d v0 = sim15.segment(3, 3);
            const Eigen::Vector3d p_aim = r_follow;
            double T_g = (HORIZON - k) * TH_INIT;
            if (T_g < TH_INIT) T_g = TH_INIT;
            Eigen::Vector3d a_req = 2.0 * (p_aim - p0 - v0 * T_g) / (T_g * T_g);
            a_req.z() = std::max(a_req.z(), (-VZ_LAND_MAX - v0.z()) / T_g);
            const Eigen::Vector3d a_tgt_ws = predictor->predictAccel(sim15(IDX_DT));
            const Eigen::Vector3d fw = MASS * (a_req - a_tgt_ws - GRAVITY);
            u0(0)          = std::max(HOVER_THRUST, std::min((double)FMAX, fw.norm()));
            u0(IDX_THETA)  = TH_INIT;
        }
        u0(0) = std::max((double)FMIN, std::min((double)FMAX, (double)u0(0)));
        u0(1) = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(1)));
        u0(2) = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(2)));
        u0(3) = std::max(-TAU_MAX, std::min(TAU_MAX, (double)u0(3)));
        u0(IDX_THETA) = std::max(THL, std::min(THH, (double)u0(IDX_THETA)));
        problem->setInitialControl(k, u0);

        Eigen::VectorXd sim15_next =
            dyn->propagate15(sim15, u0.segment(0, NU), u0(IDX_THETA));
        sim15_next(2) = std::max(sim15_next(2), 0.0);
        problem->setInitialState(k + 1, sim15_next);
        sim15 = sim15_next;
    }
    return problem;
}

// ── Descriptor ────────────────────────────────────────────────────────────────
inline OCPDescriptor descriptor() {
    OCPDescriptor d;
    d.name = "stateswitch_stc";
    d.dt = TH_INIT;
    d.default_n_replay = DEFAULT_N_REPLAY;
    d.default_mass_kg = MASS;
    d.warm_start = OCPDescriptor::WarmStart::Shift;
    d.command_mode = OCPDescriptor::CommandMode::CmdFullState;
    d.drone_odom_mode = OCPDescriptor::DroneOdomMode::Absolute;
    d.variable_dt = true;
    d.disarm_on_landing_finish = true;
    d.state_dim = NX;
    d.control_dim = NU;
    d.skip_altitude_validation = true;
    d.skip_trajectory_validation = true;
    d.log_state_headers = OCPLoggerDefaults::getStateHeaders13D();
    d.state_names = OCPLoggerDefaults::getStateNames13D();
    d.control_names = OCPLoggerDefaults::getControlNames4D();
    d.extract_actual_state_row = OCPLoggerDefaults::getActualStateRow13D;
    d.make_hover_state = OCPLoggerDefaults::makeHoverState13D;
    // Require a fresh target snapshot before solving. Without this gate, early
    // solves can run in absolute frame (t.valid=false), which biases the first
    // commands toward descending in world z instead of forward tracking.
    d.validate_target = [](const TargetSnapshot& t, double, double) {
        return t.valid;
    };
    d.transform_state = [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
        if (t.valid) {
            Eigen::VectorXd xr = x;
            xr.segment(0, 3) -= t.position;
            xr.segment(3, 3) -= t.velocity;
            return xr;
        }
        return x;
    };
    d.post_process_result = [](SolverResult& r, const TargetSnapshot& t) {
        r.is_relative_plan = t.valid;
        if (t.valid) {
            r.target_snapshot_pos = t.position;
            r.target_snapshot_vel = t.velocity;
            r.target_snapshot_acc = t.acceleration;
        } else {
            r.target_snapshot_pos.setZero();
            r.target_snapshot_vel.setZero();
            r.target_snapshot_acc.setZero();
        }
    };
    d.prepare_extra = [](const PlannerConfig&, double, const TargetSnapshot& tgt_snapshot) {
        StateswitchStcExtra ex;
        ex.predictor = std::make_shared<TargetPredictor>();
        ex.predictor->position     = tgt_snapshot.position;
        ex.predictor->velocity     = tgt_snapshot.velocity;
        ex.predictor->acceleration = tgt_snapshot.acceleration;
        ex.r_follow     = computeFollowRef(tgt_snapshot.velocity);
        ex.target_valid = tgt_snapshot.valid;
        return std::any(ex);
    };
    d.reconstruct_world_state = [](const Eigen::VectorXd& x, const TargetSnapshot& t) {
        Eigen::VectorXd xw = x;
        if (t.valid && xw.size() >= 6) {
            xw.segment(0, 3) += t.position;
            xw.segment(3, 3) += t.velocity;
        }
        return xw;
    };
    d.getSolverParams = getSolverParams;
    d.create = [](const OCPCreateArgs& a) {
        StateswitchStcExtra ex;
        if (a.extra.has_value()) {
            try {
                ex = std::any_cast<StateswitchStcExtra>(a.extra);
            } catch (const std::bad_any_cast&) {}
        }
        return create(a.current_state, a.terminal_state,
                      a.prev_U, a.prev_X, a.target_accel, a.prev_K,
                      ex.predictor, ex.r_follow);
    };
    return d;
}

}  // namespace StateswitchStcOCP
