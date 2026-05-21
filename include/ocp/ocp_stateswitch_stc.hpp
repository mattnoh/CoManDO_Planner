/// @file ocp_stateswitch_stc.hpp
/// @brief CT-cSTC state-switch OCP for relative-frame landing on a moving target.
///
/// --- Hardware flight 1 (2026-05-21, gogogo_stateswitch_stc_mpc_alipddp_20260521_111539) ---
/// Problem observed:
///   After 5 accepted solves, solver entered a cascade of 30+ consecutive rejections
///   (constraint_error up to 28,836) from which it never recovered. Root cause:
///   CTCSLandingEnvelopeCon constraints became infeasible the moment the drone descended
///   within ALT_TRIG=0.8 m of the target with realistic hardware approach speed.
///   The speed constraint trig*(||v_rel||^2 - SPD_STC_TIGHT^2) <= CTCS_BETA required
///   ||v_rel|| <= 0.81 m/s when triggered, but the drone's relative speed was ~1.1 m/s
///   at solve 2 (recovery_live handoff, control_jump_norm=12.19). The tight bound plus
///   tiny CTCS_BETA slack made every subsequent recovery solve infeasible.
///
/// Fix applied (2026-05-21):
///   SPD_STC_TIGHT : 0.8  -> 1.0  m/s  (allows ||v_rel|| up to 1.07 m/s when fully triggered)
///   CTCS_BETA     : 2e-2 -> 0.15      (7.5x more slack for hardware tracking transients)
///   ALT_TRIG      : 0.8  -> 0.45 m    (tight bounds only activate in final 45 cm of descent)
/// ---------------------------------------------------------------------------------------------
///
/// --- Hardware flight 2 (2026-05-21, gogogo_stateswtich_stc_mpc_alipddp_20260521_123115) ---
/// Problem observed:
///   Solve 0 rejected 3 times (warm-start trajectory diverged to 40+ m in absolute mode)
///   before accepting with constraint_error=0.883. At the solve 0→1 handoff, control_jump
///   norm=47.74: solve 1 computed mx=41.5, my=22.7 rad/s² at node 0 vs solve 0's
///   mx=-0.33, my=-0.28. The drone's trajectory switched from gradual attitude evolution
///   to nearly upright in 0.12 s (nodes 0→2 of solve 1), causing a violent attitude
///   transient. Flight ended after only 3 accepted solves.
///
///   Root cause: W_PHASE0_TAU=1e-3 is negligible — applying mx=41.5 rad/s² cost only
///   1e-3*41.5²=1.7 per stage, while the W_PHASE0_TILT=3000 tilt penalty at 14° tilt
///   cost 130. The optimizer rationally chose instant attitude correction, but in hardware
///   this produces a discontinuous trajectory at every plan handoff after a tracking error.
///
/// Fix applied (2026-05-21):
///   W_PHASE0_TAU  : 1e-3 -> 0.3    (300x; at mx=41.5 cost becomes 517 vs tilt cost ~43,
///                                    forcing gradual attitude correction)
///   W_PHASE0_TILT : 3000 -> 1000   (3x reduction; reduces per-step urgency; hard tilt
///                                    constraint at 25° + terminal TERM_W_TILT=200 remain)
/// ---------------------------------------------------------------------------------------------
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

static constexpr int NX = 13;
static constexpr int NU = 4;
static constexpr int NX_SS = 14;
static constexpr int NU_SS = 5;
static constexpr int IDX_DT = 13;
static constexpr int IDX_THETA = 4;

static constexpr int HORIZON = 40;
static constexpr double TH_INIT = 0.07;
static constexpr double THL = 0.05;
static constexpr double THH = 0.12;
static constexpr int DEFAULT_N_REPLAY = 8;

static constexpr double MASS = 0.027;
static constexpr double IXX = 1.66e-5;
static constexpr double IYY = 1.66e-5;
static constexpr double IZZ = 2.92e-5;
static constexpr double J_SCALE = 1.0 / IXX;
static constexpr double FMIN = 0.08;
static constexpr double FMAX = 0.6;
static constexpr double L_ARM = 0.046;
static const double TAU_MAX = L_ARM * (FMAX / 4.0 - FMIN / 4.0) * J_SCALE;
static constexpr double HOVER_THRUST = MASS * 9.81;

static const Eigen::Matrix3d J_B = []() {
    Eigen::Matrix3d J;
    J.setZero();
    J(0, 0) = IXX * J_SCALE;
    J(1, 1) = IYY * J_SCALE;
    J(2, 2) = IZZ * J_SCALE;
    return J;
}();
static const Eigen::Vector3d GRAVITY(0.0, 0.0, -9.81);

static constexpr double GS_DEG = 35.0;
static const double GS_TAN = std::tan(GS_DEG * M_PI / 180.0);
static constexpr double GS_APEX_OFFSET = 0.1;
static constexpr double VZ_LAND_MAX = 0.5;

static constexpr double TERM_W_POS_XY = 500.0;
static constexpr double TERM_W_POS_Z = 1200.0;
static constexpr double TERM_W_VEL_XY = 900.0;
static constexpr double TERM_W_VEL_Z = 600.0;
static constexpr double TERM_W_TILT = 200.0;
static constexpr double TERM_W_OM = 50.0;
static constexpr double STAGE_W_TIME = 0.04;

static constexpr double D_BEHIND = 0.2;
static constexpr double DZ_ABOVE = 1.0;
static constexpr double EPS_XY = 0.35;
static constexpr double EPS_Z = 0.35;
static constexpr double EPS_V = 0.60;
static constexpr double CAPTURE_LATCH_THRESHOLD = 0.95;
static constexpr double CAPTURE_LATCH_DWELL_SEC = 1.0;
static constexpr double CAPTURE_LATCH_REARM_Z = DZ_ABOVE + 0.5;
static constexpr double W_FOLLOW_XY = 30.0;
static constexpr double W_FOLLOW_Z = 250.0;
static constexpr double W_SIGMA = 15.0;
static constexpr double W_LAND_STAGE_XY = 80.0;
static constexpr double W_LAND_STAGE_Z = 140.0;
static constexpr double W_LAND_STAGE_V = 10.0;
static constexpr double W_PHASE0_HOVER_T = 100000.0;
static constexpr double W_PHASE0_TAU = 0.3;           // was 1e-3; prevents large moment spikes at plan handoffs
static constexpr double W_PHASE0_TILT = 1000.0;       // was 3000; hard 25° tilt con + TERM_W_TILT=200 still apply
static constexpr double W_PHASE0_OMEGA = 50.0;
static constexpr double W_PHASE0_VZ = 200.0;

static constexpr double THETA_PHASE0_DEG = 25.0;
static constexpr double OMEGA_PHASE0_MAX = 2.5;
static constexpr double THETA_STC_DEG = 10.0;
static constexpr double ALPHA_THETA_STC = 1.0;
static constexpr double OMEGA_STC_TIGHT = 0.6;
static constexpr double GS_STC_TIGHT_TAN = 0.466;
static constexpr double SPD_STC_TIGHT = 1.0;    // was 0.8; loosened for hardware tracking error
static constexpr double T_MIN_AFT = 0.21;
static constexpr double T_MAX_AFT = 0.40;
static constexpr double CTCS_BETA = 0.15;        // was 2e-2; loosened for hardware tracking error
static constexpr double V_THRUST_TRIG = 1.0;
static constexpr double THETA_THRUST_TRIG_DEG = 15.0;
static constexpr double ALT_TRIG = 0.45;         // was 0.8; activate tight bounds only in final descent
static constexpr double ALT_TRIG_FULL = 0.0;

static constexpr double W_STC_TILT = 0.0;
static constexpr double W_STC_OM = 0.0;
static constexpr double W_STC_GS = 0.0;
static constexpr double W_STC_SPD = 0.0;
static constexpr double W_STC_THRUST = 0.0;

struct TargetPredictor {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d jerk = Eigen::Vector3d::Zero();
    Eigen::Vector3d snap = Eigen::Vector3d::Zero();

    Eigen::Vector3d predictPos(double dt) const {
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt3 * dt;
        return position + velocity * dt + 0.5 * acceleration * dt2 +
               (1.0 / 6.0) * jerk * dt3 + (1.0 / 24.0) * snap * dt4;
    }

    Eigen::Vector3d predictVel(double dt) const {
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        return velocity + acceleration * dt + 0.5 * jerk * dt2 +
               (1.0 / 6.0) * snap * dt3;
    }

    Eigen::Vector3d predictAccel(double dt) const {
        const double dt2 = dt * dt;
        return acceleration + jerk * dt + 0.5 * snap * dt2;
    }
};

template<typename Scalar>
class Quad6DOFVarTimeRelativePred : public Quad6DOFVarTimeRelative<Scalar> {
    std::shared_ptr<TargetPredictor> predictor_;

public:
    explicit Quad6DOFVarTimeRelativePred(std::shared_ptr<TargetPredictor> predictor)
        : Quad6DOFVarTimeRelative<Scalar>(), predictor_(std::move(predictor)) {}

    Vector<Scalar> f(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        auto xd = x.segment(0, this->NX_PHYS).template cast<double>();
        auto ud = u.segment(0, this->NU_PHYS).template cast<double>();
        const double Th = static_cast<double>(u(IDX_THETA));
        const double tabs = static_cast<double>(x(IDX_DT));

        const Eigen::Vector3d a1 = predictor_->predictAccel(tabs);
        auto k1 = this->xdot_impl(xd, ud, a1);
        const Eigen::Vector3d a2 = predictor_->predictAccel(tabs + 0.5 * Th);
        auto k2 = this->xdot_impl(xd + 0.5 * Th * k1, ud, a2);
        const Eigen::Vector3d a3 = predictor_->predictAccel(tabs + 0.5 * Th);
        auto k3 = this->xdot_impl(xd + 0.5 * Th * k2, ud, a3);
        const Eigen::Vector3d a4 = predictor_->predictAccel(tabs + Th);
        auto k4 = this->xdot_impl(xd + Th * k3, ud, a4);

        Eigen::VectorXd xn = xd + (Th / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        xn.segment(6, 4).normalize();

        const_cast<Quad6DOFVarTimeRelativePred*>(this)->setTargetAccel(
            predictor_->predictAccel(tabs + 0.5 * Th));

        Vector<Scalar> res(NX_SS);
        res.segment(0, this->NX_PHYS) = xn.template cast<Scalar>();
        res(IDX_DT) = x(IDX_DT) + u(IDX_THETA);
        return res;
    }

    Eigen::VectorXd propagate(const Eigen::VectorXd&,
                              const Eigen::VectorXd&,
                              double) const {
        assert(false && "Quad6DOFVarTimeRelativePred: use propagate14()");
        return Eigen::VectorXd();
    }

    Eigen::VectorXd propagate14(const Eigen::VectorXd& x14,
                                const Eigen::VectorXd& u_phys,
                                double Th) const {
        const double tabs = x14(IDX_DT);
        const auto xd = x14.segment(0, this->NX_PHYS);

        const Eigen::Vector3d a1 = predictor_->predictAccel(tabs);
        auto k1 = this->xdot_impl(xd, u_phys, a1);
        const Eigen::Vector3d a2 = predictor_->predictAccel(tabs + 0.5 * Th);
        auto k2 = this->xdot_impl(xd + 0.5 * Th * k1, u_phys, a2);
        const Eigen::Vector3d a3 = predictor_->predictAccel(tabs + 0.5 * Th);
        auto k3 = this->xdot_impl(xd + 0.5 * Th * k2, u_phys, a3);
        const Eigen::Vector3d a4 = predictor_->predictAccel(tabs + Th);
        auto k4 = this->xdot_impl(xd + Th * k3, u_phys, a4);

        Eigen::VectorXd xn14(NX_SS);
        xn14.segment(0, this->NX_PHYS) = xd + (Th / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        xn14.segment(6, 4).normalize();
        xn14(IDX_DT) = x14(IDX_DT) + Th;
        return xn14;
    }
};

struct StateswitchStcExtra {
    std::shared_ptr<TargetPredictor> predictor;
    Eigen::Vector3d r_follow = Eigen::Vector3d(-D_BEHIND, 0.0, DZ_ABOVE);
    double phase_latch = 0.0;
    bool target_valid = false;
};

inline Eigen::Vector3d computeFollowRef(const Eigen::Vector3d& target_velocity) {
    Eigen::Vector3d vhat(1.0, 0.0, 0.0);
    const double speed = target_velocity.head<2>().norm();
    if (speed > 1e-6) {
        vhat = Eigen::Vector3d(target_velocity.x() / speed, target_velocity.y() / speed, 0.0);
    }
    return -D_BEHIND * vhat + Eigen::Vector3d(0.0, 0.0, DZ_ABOVE);
}

template<typename Scalar>
static Scalar captureScalarRef(const Vector<Scalar>& x,
                               double rf_x,
                               double rf_y,
                               double rf_z,
                               Vector<Scalar>* dsigma_dx = nullptr) {
    const Scalar ex = x(0) - Scalar(rf_x);
    const Scalar ey = x(1) - Scalar(rf_y);
    const Scalar ez = x(2) - Scalar(rf_z);
    const Scalar rxy = std::sqrt(ex * ex + ey * ey + Scalar(1e-8));
    const Scalar vnorm = std::sqrt(x(3) * x(3) + x(4) * x(4) + x(5) * x(5) + Scalar(1e-8));
    Scalar e_xy = rxy - Scalar(EPS_XY);
    if (e_xy < Scalar(0)) e_xy = Scalar(0);
    const Scalar e_z_abs = std::abs(ez);
    Scalar e_z = e_z_abs - Scalar(EPS_Z);
    if (e_z < Scalar(0)) e_z = Scalar(0);
    Scalar e_v = vnorm - Scalar(EPS_V);
    if (e_v < Scalar(0)) e_v = Scalar(0);

    const Scalar denom = Scalar(1) + Scalar(W_SIGMA) * (e_xy * e_xy + e_z * e_z + e_v * e_v);
    const Scalar sigma = Scalar(1) / denom;

    if (dsigma_dx) {
        *dsigma_dx = Vector<Scalar>::Zero(x.size());
        const Scalar coeff = -Scalar(W_SIGMA) * sigma * sigma;
        if (e_xy > Scalar(0)) {
            const Scalar c = coeff * Scalar(2) * e_xy / rxy;
            (*dsigma_dx)(0) = c * ex;
            (*dsigma_dx)(1) = c * ey;
        }
        if (e_z > Scalar(0)) {
            (*dsigma_dx)(2) = coeff * Scalar(2) * e_z *
                              ((ez >= Scalar(0)) ? Scalar(1) : Scalar(-1));
        }
        if (e_v > Scalar(0)) {
            const Scalar c = coeff * Scalar(2) * e_v / vnorm;
            (*dsigma_dx)(3) = c * x(3);
            (*dsigma_dx)(4) = c * x(4);
            (*dsigma_dx)(5) = c * x(5);
        }
    }
    return sigma;
}

template<typename Scalar>
static Scalar triggerScalar(const Vector<Scalar>& x,
                            double rf_x,
                            double rf_y,
                            double rf_z,
                            double phase_latch,
                            Vector<Scalar>* dtrig_dx = nullptr) {
    if (phase_latch >= 0.8) {
        if (dtrig_dx) *dtrig_dx = Vector<Scalar>::Zero(x.size());
        return Scalar(1);
    }

    const Scalar gs_v = Scalar(GS_STC_TIGHT_TAN) *
                            std::sqrt(x(0) * x(0) + x(1) * x(1) + Scalar(1e-12)) -
                        (x(2) + Scalar(GS_APEX_OFFSET));
    if (gs_v > Scalar(0)) {
        if (dtrig_dx) *dtrig_dx = Vector<Scalar>::Zero(x.size());
        return Scalar(0);
    }

    const Scalar s = Scalar(ALT_TRIG) - x(2);
    const Scalar den = Scalar(ALT_TRIG - ALT_TRIG_FULL);
    Scalar alt_gate = Scalar(0);
    Scalar dalt_dz = Scalar(0);
    if (s > Scalar(0)) {
        if (s >= den) {
            alt_gate = Scalar(1);
        } else {
            alt_gate = (s * s) / (den * den);
            dalt_dz = Scalar(-2) * s / (den * den);
        }
    }

    Vector<Scalar> dsigma_dx;
    const Scalar sigma = captureScalarRef(x, rf_x, rf_y, rf_z, dtrig_dx ? &dsigma_dx : nullptr);
    if (dtrig_dx) {
        *dtrig_dx = alt_gate * dsigma_dx;
        (*dtrig_dx)(2) += sigma * dalt_dz;
    }
    return alt_gate * sigma;
}

template<typename Scalar>
static Scalar triggerAltCapture(const Vector<Scalar>& x,
                                double rf_x,
                                double rf_y,
                                double rf_z,
                                double phase_latch,
                                Vector<Scalar>* dtrig_dx = nullptr) {
    if (phase_latch >= 0.8) {
        if (dtrig_dx) *dtrig_dx = Vector<Scalar>::Zero(x.size());
        return Scalar(1);
    }

    const Scalar s = Scalar(ALT_TRIG) - x(2);
    const Scalar den = Scalar(ALT_TRIG - ALT_TRIG_FULL);
    Scalar alt_gate = Scalar(0);
    Scalar dalt_dz = Scalar(0);
    if (s > Scalar(0)) {
        if (s >= den) {
            alt_gate = Scalar(1);
        } else {
            alt_gate = (s * s) / (den * den);
            dalt_dz = Scalar(-2) * s / (den * den);
        }
    }

    Vector<Scalar> dsigma_dx;
    const Scalar sigma = captureScalarRef(x, rf_x, rf_y, rf_z, dtrig_dx ? &dsigma_dx : nullptr);
    if (dtrig_dx) {
        *dtrig_dx = alt_gate * dsigma_dx;
        (*dtrig_dx)(2) += sigma * dalt_dz;
    }
    return alt_gate * sigma;
}

template<typename Scalar>
static Scalar triggerSlowUpright(const Vector<Scalar>& x,
                                 Vector<Scalar>* dsig_dx = nullptr) {
    static const double tilt_trig =
        std::pow(std::sin(0.5 * THETA_THRUST_TRIG_DEG * M_PI / 180.0), 2);
    const Scalar vt2 = Scalar(V_THRUST_TRIG * V_THRUST_TRIG);
    const Scalar v2 = x(3) * x(3) + x(4) * x(4) + x(5) * x(5);
    const Scalar raw_slow = (vt2 - v2) / vt2;
    const Scalar sigma_slow =
        (raw_slow < Scalar(0)) ? Scalar(0) : (raw_slow > Scalar(1)) ? Scalar(1) : raw_slow;
    const Scalar tilt2 = x(7) * x(7) + x(8) * x(8);
    const Scalar raw_up = (Scalar(tilt_trig) - tilt2) / Scalar(tilt_trig);
    const Scalar sigma_up =
        (raw_up < Scalar(0)) ? Scalar(0) : (raw_up > Scalar(1)) ? Scalar(1) : raw_up;

    if (dsig_dx) {
        *dsig_dx = Vector<Scalar>::Zero(x.size());
        if (raw_slow > Scalar(0) && raw_slow < Scalar(1)) {
            (*dsig_dx)(3) = sigma_up * Scalar(-2) * x(3) / vt2;
            (*dsig_dx)(4) = sigma_up * Scalar(-2) * x(4) / vt2;
            (*dsig_dx)(5) = sigma_up * Scalar(-2) * x(5) / vt2;
        }
        if (raw_up > Scalar(0) && raw_up < Scalar(1)) {
            (*dsig_dx)(7) += sigma_slow * Scalar(-2) * x(7) / Scalar(tilt_trig);
            (*dsig_dx)(8) += sigma_slow * Scalar(-2) * x(8) / Scalar(tilt_trig);
        }
    }
    return sigma_slow * sigma_up;
}

template<typename Scalar>
class RelTermCost : public TerminalCostBase<Scalar> {
    double wp_xy_;
    double wp_z_;
    double wv_xy_;
    double wv_z_;
    double watt_;
    double wom_;
    double vz_ref_;

public:
    RelTermCost(double wp_xy,
                double wp_z,
                double wv_xy,
                double wv_z,
                double watt,
                double wom,
                double vz_ref)
        : wp_xy_(wp_xy),
          wp_z_(wp_z),
          wv_xy_(wv_xy),
          wv_z_(wv_z),
          watt_(watt),
          wom_(wom),
          vz_ref_(vz_ref) {}

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
        const Scalar ep = x.template segment<2>(0).squaredNorm();
        const Scalar ez = x(2) * x(2);
        const Scalar evxy = x.template segment<2>(3).squaredNorm();
        const Scalar evz = (x(5) - Scalar(vz_ref_)) * (x(5) - Scalar(vz_ref_));
        Scalar bx, by;
        bodyZHorizontal(x, bx, by);
        const Scalar eom = x.template segment<3>(10).squaredNorm();
        return Scalar(0.5) *
               (Scalar(wp_xy_) * ep + Scalar(wp_z_) * ez + Scalar(wv_xy_) * evxy +
                Scalar(wv_z_) * evz + Scalar(watt_) * (bx * bx + by * by) +
                Scalar(wom_) * eom);
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        g.segment(0, 2) = Scalar(wp_xy_) * x.segment(0, 2);
        g(2) = Scalar(wp_z_) * x(2);
        g.segment(3, 2) = Scalar(wv_xy_) * x.segment(3, 2);
        g(5) = Scalar(wv_z_) * (x(5) - Scalar(vz_ref_));
        Scalar bx, by;
        bodyZHorizontal(x, bx, by);
        Matrix<Scalar> Jtilt = bodyZHorizontalJacobian(x);
        Vector<Scalar> etilt(2);
        etilt << bx, by;
        g.segment(6, 4) += Scalar(watt_) * Jtilt.transpose() * etilt;
        g.segment(10, 3) = Scalar(wom_) * x.segment(10, 3);
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        H(0, 0) = H(1, 1) = Scalar(wp_xy_);
        H(2, 2) = Scalar(wp_z_);
        H(3, 3) = H(4, 4) = Scalar(wv_xy_);
        H(5, 5) = Scalar(wv_z_);
        Matrix<Scalar> Jtilt = bodyZHorizontalJacobian(x);
        H.block(6, 6, 4, 4) += Scalar(watt_) * Jtilt.transpose() * Jtilt;
        H(10, 10) = H(11, 11) = H(12, 12) = Scalar(wom_);
        return H;
    }
};

template<typename Scalar>
class CaptureAwareTermCost : public TerminalCostBase<Scalar> {
    double wp_xy_;
    double wp_z_;
    double wv_xy_;
    double wv_z_;
    double watt_;
    double wom_;
    double vz_ref_;
    double phase_latch_;
    Eigen::Vector3d r_follow_;

public:
    CaptureAwareTermCost(double wp_xy,
                         double wp_z,
                         double wv_xy,
                         double wv_z,
                         double watt,
                         double wom,
                         double vz_ref,
                         const Eigen::Vector3d& r_follow,
                         double phase_latch)
        : wp_xy_(wp_xy),
          wp_z_(wp_z),
          wv_xy_(wv_xy),
          wv_z_(wv_z),
          watt_(watt),
          wom_(wom),
          vz_ref_(vz_ref),
          phase_latch_(phase_latch),
          r_follow_(r_follow) {}

    Scalar committedPhase() const {
        if (phase_latch_ <= 0.0) return Scalar(0);
        if (phase_latch_ >= 1.0) return Scalar(1);
        return Scalar(phase_latch_);
    }

    Scalar p(const Vector<Scalar>& x) const override {
        const Scalar phase = committedPhase();
        Vector<Scalar> rref(3);
        rref << Scalar(r_follow_.x()), Scalar(r_follow_.y()), Scalar(r_follow_.z());
        rref *= (Scalar(1) - phase);
        const Vector<Scalar> ep = x.template segment<3>(0) - rref;
        const Scalar evxy = x.template segment<2>(3).squaredNorm();
        const Scalar evz = x(5) - phase * Scalar(vz_ref_);
        Scalar bx, by;
        RelTermCost<Scalar>::bodyZHorizontal(x, bx, by);
        const Scalar eom = x.template segment<3>(10).squaredNorm();
        return Scalar(0.5) *
               (Scalar(wp_xy_) * ep.template segment<2>(0).squaredNorm() +
                Scalar(wp_z_) * ep(2) * ep(2) + Scalar(wv_xy_) * evxy +
                Scalar(wv_z_) * evz * evz + Scalar(watt_) * (bx * bx + by * by) +
                Scalar(wom_) * eom);
    }

    Vector<Scalar> px(const Vector<Scalar>& x) const override {
        const Scalar phase = committedPhase();
        Vector<Scalar> rf = Vector<Scalar>::Zero(3);
        rf << Scalar(r_follow_.x()), Scalar(r_follow_.y()), Scalar(r_follow_.z());
        const Vector<Scalar> ep = x.template segment<3>(0) - (Scalar(1) - phase) * rf;

        Matrix<Scalar> Jp = Matrix<Scalar>::Zero(3, x.size());
        Jp(0, 0) = Scalar(1);
        Jp(1, 1) = Scalar(1);
        Jp(2, 2) = Scalar(1);

        Vector<Scalar> g = Vector<Scalar>::Zero(x.size());
        Vector<Scalar> wpe(3);
        wpe << Scalar(wp_xy_) * ep(0), Scalar(wp_xy_) * ep(1), Scalar(wp_z_) * ep(2);
        g += Jp.transpose() * wpe;
        g(3) += Scalar(wv_xy_) * x(3);
        g(4) += Scalar(wv_xy_) * x(4);

        const Scalar evz = x(5) - phase * Scalar(vz_ref_);
        Vector<Scalar> Jvz = Vector<Scalar>::Zero(x.size());
        Jvz(5) = Scalar(1);
        g += Scalar(wv_z_) * evz * Jvz;

        Scalar bx, by;
        RelTermCost<Scalar>::bodyZHorizontal(x, bx, by);
        Matrix<Scalar> Jtilt = RelTermCost<Scalar>::bodyZHorizontalJacobian(x);
        Vector<Scalar> etilt(2);
        etilt << bx, by;
        g.segment(6, 4) += Scalar(watt_) * Jtilt.transpose() * etilt;
        g.segment(10, 3) += Scalar(wom_) * x.segment(10, 3);
        return g;
    }

    Matrix<Scalar> pxx(const Vector<Scalar>& x) const override {
        Vector<Scalar> rf = Vector<Scalar>::Zero(3);
        rf << Scalar(r_follow_.x()), Scalar(r_follow_.y()), Scalar(r_follow_.z());

        Matrix<Scalar> H = Matrix<Scalar>::Zero(x.size(), x.size());
        Matrix<Scalar> Jp = Matrix<Scalar>::Zero(3, x.size());
        Jp(0, 0) = Scalar(1);
        Jp(1, 1) = Scalar(1);
        Jp(2, 2) = Scalar(1);

        Matrix<Scalar> Wp = Matrix<Scalar>::Zero(3, 3);
        Wp(0, 0) = Scalar(wp_xy_);
        Wp(1, 1) = Scalar(wp_xy_);
        Wp(2, 2) = Scalar(wp_z_);
        H += Jp.transpose() * Wp * Jp;
        H(3, 3) += Scalar(wv_xy_);
        H(4, 4) += Scalar(wv_xy_);

        Vector<Scalar> Jvz = Vector<Scalar>::Zero(x.size());
        Jvz(5) = Scalar(1);
        H += Scalar(wv_z_) * (Jvz * Jvz.transpose());

        Matrix<Scalar> Jtilt = RelTermCost<Scalar>::bodyZHorizontalJacobian(x);
        H.block(6, 6, 4, 4) += Scalar(watt_) * Jtilt.transpose() * Jtilt;
        H(10, 10) += Scalar(wom_);
        H(11, 11) += Scalar(wom_);
        H(12, 12) += Scalar(wom_);
        return H;
    }
};

template<typename Scalar>
class CTcSTCStageCost : public StageCostBase<Scalar> {
    Scalar eps_;
    Scalar w_follow_xy_;
    Scalar w_follow_z_;
    double rf_x_;
    double rf_y_;
    double rf_z_;
    double phase_latch_;

public:
    CTcSTCStageCost(double eps,
                    double w_follow_xy,
                    double w_follow_z,
                    const Eigen::Vector3d& r_follow,
                    double phase_latch)
        : eps_(Scalar(eps)),
          w_follow_xy_(Scalar(w_follow_xy)),
          w_follow_z_(Scalar(w_follow_z)),
          rf_x_(r_follow.x()),
          rf_y_(r_follow.y()),
          rf_z_(r_follow.z()),
          phase_latch_(phase_latch) {}

private:
    Scalar committedPhase() const {
        if (phase_latch_ <= 0.0) return Scalar(0);
        if (phase_latch_ >= 1.0) return Scalar(1);
        return Scalar(phase_latch_);
    }

    void evalSigma(const Vector<Scalar>& x,
                   Scalar& sigma,
                   Scalar& e_xy,
                   Scalar& e_v,
                   Scalar& rxy,
                   Scalar& vnorm) const {
        const Scalar ex = x(0) - Scalar(rf_x_);
        const Scalar ey = x(1) - Scalar(rf_y_);
        const Scalar ez = x(2) - Scalar(rf_z_);
        rxy = std::sqrt(ex * ex + ey * ey + Scalar(1e-8));
        vnorm = std::sqrt(x(3) * x(3) + x(4) * x(4) + x(5) * x(5) + Scalar(1e-8));
        e_xy = rxy - Scalar(EPS_XY);
        if (e_xy < Scalar(0)) e_xy = Scalar(0);
        Scalar e_z = std::abs(ez) - Scalar(EPS_Z);
        if (e_z < Scalar(0)) e_z = Scalar(0);
        e_v = vnorm - Scalar(EPS_V);
        if (e_v < Scalar(0)) e_v = Scalar(0);
        sigma = Scalar(1) /
                (Scalar(1) + Scalar(W_SIGMA) * (e_xy * e_xy + e_z * e_z + e_v * e_v));
    }

    void evalTrig(const Vector<Scalar>& x, Scalar& trig, Scalar& s_alt) const {
        s_alt = Scalar(ALT_TRIG) - x(2);
        if (s_alt < Scalar(0)) s_alt = Scalar(0);
        const Scalar den = Scalar(ALT_TRIG - ALT_TRIG_FULL);
        if (s_alt >= den) {
            s_alt = Scalar(0);
            trig = Scalar(1);
            return;
        }
        trig = (s_alt * s_alt) / (den * den);
    }

    struct Pen {
        Scalar g_tilt, g_om, g_v, g_gs, g_fmin, g_fmax;
        Scalar pen_tilt, pen_om, pen_v, pen_gs, pen_fmin, pen_fmax;
        Scalar pen_sum;
        Scalar qperp_norm, om_norm;
    };

    Pen evalPen(const Vector<Scalar>& x, const Vector<Scalar>& u) const {
        Pen p;
        static const Scalar thr_tilt =
            Scalar(std::sin(0.5 * ALPHA_THETA_STC * THETA_STC_DEG * M_PI / 180.0));
        p.qperp_norm = std::sqrt(x(7) * x(7) + x(8) * x(8) + Scalar(1e-8));
        p.g_tilt = p.qperp_norm - thr_tilt;
        const Scalar vt = (p.g_tilt > Scalar(0)) ? p.g_tilt : Scalar(0);
        p.pen_tilt = vt * vt;

        p.om_norm = std::sqrt(x(10) * x(10) + x(11) * x(11) + x(12) * x(12) + Scalar(1e-8));
        p.g_om = p.om_norm - Scalar(OMEGA_STC_TIGHT);
        const Scalar vo = (p.g_om > Scalar(0)) ? p.g_om : Scalar(0);
        p.pen_om = vo * vo;

        const Scalar v2 = x(3) * x(3) + x(4) * x(4) + x(5) * x(5);
        p.g_v = v2 - Scalar(SPD_STC_TIGHT * SPD_STC_TIGHT);
        const Scalar vv = (p.g_v > Scalar(0)) ? p.g_v : Scalar(0);
        p.pen_v = vv * vv;

        const Scalar rxy = std::sqrt(x(0) * x(0) + x(1) * x(1) + Scalar(1e-12));
        p.g_gs = Scalar(GS_STC_TIGHT_TAN) * rxy - (x(2) + Scalar(GS_APEX_OFFSET));
        const Scalar vg = (p.g_gs > Scalar(0)) ? p.g_gs : Scalar(0);
        p.pen_gs = vg * vg;

        p.g_fmin = Scalar(T_MIN_AFT) - u(0);
        const Scalar vfn = (p.g_fmin > Scalar(0)) ? p.g_fmin : Scalar(0);
        p.pen_fmin = vfn * vfn;
        p.g_fmax = u(0) - Scalar(T_MAX_AFT);
        const Scalar vfx = (p.g_fmax > Scalar(0)) ? p.g_fmax : Scalar(0);
        p.pen_fmax = vfx * vfx;

        p.pen_sum = Scalar(W_STC_TILT) * p.pen_tilt + Scalar(W_STC_OM) * p.pen_om +
                    Scalar(W_STC_SPD) * p.pen_v + Scalar(W_STC_GS) * p.pen_gs +
                    Scalar(W_STC_THRUST) * (p.pen_fmin + p.pen_fmax);
        return p;
    }

public:
    Scalar q(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const Scalar phase = committedPhase();
        const Scalar phase0 = Scalar(1) - phase;
        Scalar trig, s_alt;
        evalTrig(x, trig, s_alt);
        auto pen = evalPen(x, u);

        const Scalar dx = x(0) - Scalar(rf_x_);
        const Scalar dy = x(1) - Scalar(rf_y_);
        const Scalar dz = x(2) - Scalar(rf_z_);
        const Scalar F = w_follow_xy_ * (dx * dx + dy * dy) + w_follow_z_ * dz * dz;
        const Scalar L = Scalar(W_LAND_STAGE_XY) * (x(0) * x(0) + x(1) * x(1)) +
                         Scalar(W_LAND_STAGE_Z) * x(2) * x(2) +
                         Scalar(W_LAND_STAGE_V) * x.segment(3, 3).squaredNorm();
        const Scalar reg = eps_ * (x.segment(3, 3).squaredNorm() +
                                   x.segment(10, 3).squaredNorm());
        const Scalar ef = u(0) - Scalar(HOVER_THRUST);
        const Scalar ctrl = Scalar(W_PHASE0_HOVER_T) * ef * ef +
                            Scalar(W_PHASE0_TAU) * u.segment(1, 3).squaredNorm();
        const Scalar hover_state = Scalar(W_PHASE0_TILT) * (x(7) * x(7) + x(8) * x(8)) +
                                   Scalar(W_PHASE0_OMEGA) * x.segment(10, 3).squaredNorm() +
                                   Scalar(W_PHASE0_VZ) * x(5) * x(5);

        return Scalar(STAGE_W_TIME) * u(IDX_THETA) + reg + phase0 * F + phase * L +
               phase0 * ctrl + phase0 * hover_state + trig * pen.pen_sum;
    }

    Vector<Scalar> qx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NX_SS);
        g.segment(3, 3) = Scalar(2) * eps_ * x.segment(3, 3);
        g.segment(10, 3) = Scalar(2) * eps_ * x.segment(10, 3);

        const Scalar phase = committedPhase();
        const Scalar phase0 = Scalar(1) - phase;
        const Scalar dx = x(0) - Scalar(rf_x_);
        const Scalar dy = x(1) - Scalar(rf_y_);
        const Scalar dz = x(2) - Scalar(rf_z_);
        g(0) += Scalar(2) * phase0 * w_follow_xy_ * dx;
        g(1) += Scalar(2) * phase0 * w_follow_xy_ * dy;
        g(2) += Scalar(2) * phase0 * w_follow_z_ * dz;

        g(0) += phase * Scalar(2) * Scalar(W_LAND_STAGE_XY) * x(0);
        g(1) += phase * Scalar(2) * Scalar(W_LAND_STAGE_XY) * x(1);
        g(2) += phase * Scalar(2) * Scalar(W_LAND_STAGE_Z) * x(2);
        g.segment(3, 3) += phase * Scalar(2) * Scalar(W_LAND_STAGE_V) * x.segment(3, 3);

        g(5) += phase0 * Scalar(2) * Scalar(W_PHASE0_VZ) * x(5);
        g(7) += phase0 * Scalar(2) * Scalar(W_PHASE0_TILT) * x(7);
        g(8) += phase0 * Scalar(2) * Scalar(W_PHASE0_TILT) * x(8);
        g.segment(10, 3) += phase0 * Scalar(2) * Scalar(W_PHASE0_OMEGA) *
                             x.segment(10, 3);

        Scalar trig, s_alt;
        evalTrig(x, trig, s_alt);
        auto pen = evalPen(x, u);
        if (s_alt > Scalar(0)) {
            const Scalar den = Scalar(ALT_TRIG - ALT_TRIG_FULL);
            g(2) += Scalar(-2) * s_alt / (den * den) * pen.pen_sum;
        }
        if (pen.g_tilt > Scalar(0)) {
            const Scalar c = trig * Scalar(W_STC_TILT) * Scalar(2) * pen.g_tilt /
                             pen.qperp_norm;
            g(7) += c * x(7);
            g(8) += c * x(8);
        }
        if (pen.g_om > Scalar(0)) {
            const Scalar c = trig * Scalar(W_STC_OM) * Scalar(2) * pen.g_om / pen.om_norm;
            g(10) += c * x(10);
            g(11) += c * x(11);
            g(12) += c * x(12);
        }
        if (pen.g_v > Scalar(0)) {
            const Scalar c = trig * Scalar(W_STC_SPD) * Scalar(4) * pen.g_v;
            g(3) += c * x(3);
            g(4) += c * x(4);
            g(5) += c * x(5);
        }
        const Scalar rxy_pen = std::sqrt(x(0) * x(0) + x(1) * x(1) + Scalar(1e-12));
        if (pen.g_gs > Scalar(0)) {
            const Scalar c = trig * Scalar(W_STC_GS) * Scalar(2) * pen.g_gs;
            g(0) += c * Scalar(GS_STC_TIGHT_TAN) * x(0) / rxy_pen;
            g(1) += c * Scalar(GS_STC_TIGHT_TAN) * x(1) / rxy_pen;
            g(2) -= c;
        }
        return g;
    }

    Vector<Scalar> qu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Vector<Scalar> g = Vector<Scalar>::Zero(NU_SS);
        g(IDX_THETA) = Scalar(STAGE_W_TIME);
        const Scalar phase0 = Scalar(1) - committedPhase();
        g(0) += phase0 * Scalar(2) * Scalar(W_PHASE0_HOVER_T) *
                (u(0) - Scalar(HOVER_THRUST));
        g.segment(1, 3) += phase0 * Scalar(2) * Scalar(W_PHASE0_TAU) * u.segment(1, 3);

        Scalar trig, s_alt;
        evalTrig(x, trig, s_alt);
        auto pen = evalPen(x, u);
        if (pen.g_fmin > Scalar(0)) {
            g(0) -= trig * Scalar(W_STC_THRUST) * Scalar(2) * pen.g_fmin;
        }
        if (pen.g_fmax > Scalar(0)) {
            g(0) += trig * Scalar(W_STC_THRUST) * Scalar(2) * pen.g_fmax;
        }
        return g;
    }

    Matrix<Scalar> qxx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NX_SS, NX_SS);
        for (int i = 3; i < 6; ++i) H(i, i) = Scalar(2) * eps_;
        for (int i = 10; i < 13; ++i) H(i, i) = Scalar(2) * eps_;

        const Scalar phase = committedPhase();
        const Scalar phase0 = Scalar(1) - phase;
        H(0, 0) += Scalar(2) * phase0 * w_follow_xy_;
        H(1, 1) += Scalar(2) * phase0 * w_follow_xy_;
        H(2, 2) += Scalar(2) * phase0 * w_follow_z_;
        H(0, 0) += phase * Scalar(2) * Scalar(W_LAND_STAGE_XY);
        H(1, 1) += phase * Scalar(2) * Scalar(W_LAND_STAGE_XY);
        H(2, 2) += phase * Scalar(2) * Scalar(W_LAND_STAGE_Z);
        H(3, 3) += phase * Scalar(2) * Scalar(W_LAND_STAGE_V);
        H(4, 4) += phase * Scalar(2) * Scalar(W_LAND_STAGE_V);
        H(5, 5) += phase * Scalar(2) * Scalar(W_LAND_STAGE_V);
        H(5, 5) += phase0 * Scalar(2) * Scalar(W_PHASE0_VZ);
        H(7, 7) += phase0 * Scalar(2) * Scalar(W_PHASE0_TILT);
        H(8, 8) += phase0 * Scalar(2) * Scalar(W_PHASE0_TILT);
        H(10, 10) += phase0 * Scalar(2) * Scalar(W_PHASE0_OMEGA);
        H(11, 11) += phase0 * Scalar(2) * Scalar(W_PHASE0_OMEGA);
        H(12, 12) += phase0 * Scalar(2) * Scalar(W_PHASE0_OMEGA);

        Scalar trig, s_alt;
        evalTrig(x, trig, s_alt);
        auto pen = evalPen(x, u);
        if (pen.pen_sum > Scalar(0)) {
            const Scalar den = Scalar(ALT_TRIG - ALT_TRIG_FULL);
            H(2, 2) += pen.pen_sum * Scalar(2) / (den * den);
        }
        if (s_alt > Scalar(0)) {
            const Scalar den = Scalar(ALT_TRIG - ALT_TRIG_FULL);
            const Scalar dtdz = Scalar(-2) * s_alt / (den * den);
            H(2, 2) += pen.pen_sum * dtdz * dtdz;
        }
        if (pen.g_tilt > Scalar(0)) {
            const Scalar c = trig * Scalar(W_STC_TILT) * Scalar(2) /
                             (pen.qperp_norm * pen.qperp_norm);
            H(7, 7) += c * x(7) * x(7);
            H(7, 8) += c * x(7) * x(8);
            H(8, 7) += c * x(8) * x(7);
            H(8, 8) += c * x(8) * x(8);
        }
        if (pen.g_om > Scalar(0)) {
            const Scalar c = trig * Scalar(W_STC_OM) * Scalar(2) /
                             (pen.om_norm * pen.om_norm);
            for (int i = 10; i < 13; ++i) {
                for (int j = 10; j < 13; ++j) {
                    H(i, j) += c * x(i) * x(j);
                }
            }
        }
        if (pen.g_v > Scalar(0)) {
            const Scalar c = trig * Scalar(W_STC_SPD) * Scalar(8);
            for (int i = 3; i < 6; ++i) {
                for (int j = 3; j < 6; ++j) {
                    H(i, j) += c * x(i) * x(j);
                }
            }
        }
        const Scalar rxy_pen = std::sqrt(x(0) * x(0) + x(1) * x(1) + Scalar(1e-12));
        if (pen.g_gs > Scalar(0)) {
            const Scalar c = trig * Scalar(W_STC_GS) * Scalar(2);
            const Scalar dgdx = Scalar(GS_STC_TIGHT_TAN) * x(0) / rxy_pen;
            const Scalar dgdy = Scalar(GS_STC_TIGHT_TAN) * x(1) / rxy_pen;
            H(0, 0) += c * dgdx * dgdx;
            H(0, 1) += c * dgdx * dgdy;
            H(0, 2) -= c * dgdx;
            H(1, 0) += c * dgdy * dgdx;
            H(1, 1) += c * dgdy * dgdy;
            H(1, 2) -= c * dgdy;
            H(2, 0) -= c * dgdx;
            H(2, 1) -= c * dgdy;
            H(2, 2) += c;
        }
        return H;
    }

    Matrix<Scalar> quu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> H = Matrix<Scalar>::Zero(NU_SS, NU_SS);
        const Scalar phase0 = Scalar(1) - committedPhase();
        H(0, 0) += phase0 * Scalar(2) * Scalar(W_PHASE0_HOVER_T);
        H(1, 1) += phase0 * Scalar(2) * Scalar(W_PHASE0_TAU);
        H(2, 2) += phase0 * Scalar(2) * Scalar(W_PHASE0_TAU);
        H(3, 3) += phase0 * Scalar(2) * Scalar(W_PHASE0_TAU);

        Scalar trig, s_alt;
        evalTrig(x, trig, s_alt);
        auto pen = evalPen(x, u);
        if (pen.g_fmin > Scalar(0)) H(0, 0) += trig * Scalar(W_STC_THRUST) * Scalar(2);
        if (pen.g_fmax > Scalar(0)) H(0, 0) += trig * Scalar(W_STC_THRUST) * Scalar(2);
        return H;
    }

    Matrix<Scalar> qxu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(NX_SS, NU_SS);
        Scalar trig, s_alt;
        evalTrig(x, trig, s_alt);
        auto pen = evalPen(x, u);
        if (s_alt > Scalar(0) && Scalar(W_STC_THRUST) != Scalar(0)) {
            const Scalar den = Scalar(ALT_TRIG - ALT_TRIG_FULL);
            const Scalar dtdz = Scalar(-2) * s_alt / (den * den);
            Scalar dpen_du0 = Scalar(0);
            if (pen.g_fmin > Scalar(0)) {
                dpen_du0 -= Scalar(W_STC_THRUST) * Scalar(2) * pen.g_fmin;
            }
            if (pen.g_fmax > Scalar(0)) {
                dpen_du0 += Scalar(W_STC_THRUST) * Scalar(2) * pen.g_fmax;
            }
            J(2, 0) = dtdz * dpen_du0;
        }
        return J;
    }
};

template<typename Scalar>
class CTCSLandingEnvelopeCon : public StageConstraintBase<Scalar> {
    Eigen::Vector3d r_follow_;
    double phase_latch_;

public:
    explicit CTCSLandingEnvelopeCon(const Eigen::Vector3d& r_follow, double phase_latch)
        : r_follow_(r_follow), phase_latch_(phase_latch) {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 8;
    }

    Vector<Scalar> c(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        const Scalar trig =
            triggerScalar(x, r_follow_.x(), r_follow_.y(), r_follow_.z(), phase_latch_);
        const Scalar trig_ac =
            triggerAltCapture(x, r_follow_.x(), r_follow_.y(), r_follow_.z(), phase_latch_);
        const Scalar sigma_su = triggerSlowUpright(x);
        static const double tilt_loose_sq =
            std::pow(std::sin(0.5 * THETA_PHASE0_DEG * M_PI / 180.0), 2);
        static const double tilt_tight_sq =
            std::pow(std::sin(0.5 * ALPHA_THETA_STC * THETA_STC_DEG * M_PI / 180.0), 2);
        const Scalar rxy = std::sqrt(x(0) * x(0) + x(1) * x(1) + Scalar(1e-12));

        Vector<Scalar> cn(8);
        cn(0) = x(7) * x(7) + x(8) * x(8) - Scalar(tilt_loose_sq);
        cn(1) = x.segment(10, 3).squaredNorm() -
                Scalar(OMEGA_PHASE0_MAX * OMEGA_PHASE0_MAX);
        cn(2) = trig * (x.segment(3, 3).squaredNorm() -
                        Scalar(SPD_STC_TIGHT * SPD_STC_TIGHT)) -
                Scalar(CTCS_BETA);
        cn(3) = trig * (x(7) * x(7) + x(8) * x(8) - Scalar(tilt_tight_sq)) -
                Scalar(CTCS_BETA);
        cn(4) = trig * (x.segment(10, 3).squaredNorm() -
                        Scalar(OMEGA_STC_TIGHT * OMEGA_STC_TIGHT)) -
                Scalar(CTCS_BETA);
        cn(5) = trig_ac * (Scalar(GS_STC_TIGHT_TAN) * rxy -
                           (x(2) + Scalar(GS_APEX_OFFSET))) -
                Scalar(CTCS_BETA);
        const Scalar tsu = trig * sigma_su;
        cn(6) = tsu * (u(0) - Scalar(T_MAX_AFT)) - Scalar(CTCS_BETA);
        cn(7) = tsu * (Scalar(T_MIN_AFT) - u(0)) - Scalar(CTCS_BETA);
        return cn;
    }

    Matrix<Scalar> cx(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(8, x.size());
        Vector<Scalar> dtrig_dx, dtrig_ac_dx, dsu_dx;
        const Scalar trig =
            triggerScalar(x, r_follow_.x(), r_follow_.y(), r_follow_.z(), phase_latch_,
                          &dtrig_dx);
        const Scalar trig_ac =
            triggerAltCapture(x, r_follow_.x(), r_follow_.y(), r_follow_.z(), phase_latch_,
                              &dtrig_ac_dx);
        const Scalar sigma_su = triggerSlowUpright(x, &dsu_dx);
        static const double tilt_tight_sq =
            std::pow(std::sin(0.5 * ALPHA_THETA_STC * THETA_STC_DEG * M_PI / 180.0), 2);
        const Scalar rxy = std::sqrt(x(0) * x(0) + x(1) * x(1) + Scalar(1e-12));

        J(0, 7) = Scalar(2) * x(7);
        J(0, 8) = Scalar(2) * x(8);
        J(1, 10) = Scalar(2) * x(10);
        J(1, 11) = Scalar(2) * x(11);
        J(1, 12) = Scalar(2) * x(12);

        const Scalar spd_v = x.segment(3, 3).squaredNorm() -
                             Scalar(SPD_STC_TIGHT * SPD_STC_TIGHT);
        J.row(2) = (dtrig_dx * spd_v).transpose();
        J(2, 3) += trig * Scalar(2) * x(3);
        J(2, 4) += trig * Scalar(2) * x(4);
        J(2, 5) += trig * Scalar(2) * x(5);

        const Scalar tilt_v = x(7) * x(7) + x(8) * x(8) - Scalar(tilt_tight_sq);
        J.row(3) = (dtrig_dx * tilt_v).transpose();
        J(3, 7) += trig * Scalar(2) * x(7);
        J(3, 8) += trig * Scalar(2) * x(8);

        const Scalar omega_v = x.segment(10, 3).squaredNorm() -
                               Scalar(OMEGA_STC_TIGHT * OMEGA_STC_TIGHT);
        J.row(4) = (dtrig_dx * omega_v).transpose();
        J(4, 10) += trig * Scalar(2) * x(10);
        J(4, 11) += trig * Scalar(2) * x(11);
        J(4, 12) += trig * Scalar(2) * x(12);

        const Scalar gs_v = Scalar(GS_STC_TIGHT_TAN) * rxy -
                            (x(2) + Scalar(GS_APEX_OFFSET));
        J.row(5) = (dtrig_ac_dx * gs_v).transpose();
        J(5, 0) += trig_ac * Scalar(GS_STC_TIGHT_TAN) * x(0) / rxy;
        J(5, 1) += trig_ac * Scalar(GS_STC_TIGHT_TAN) * x(1) / rxy;
        J(5, 2) -= trig_ac;

        const Scalar h6 = u(0) - Scalar(T_MAX_AFT);
        const Scalar h7 = Scalar(T_MIN_AFT) - u(0);
        J.row(6) = (dtrig_dx * (sigma_su * h6) + dsu_dx * (trig * h6)).transpose();
        J.row(7) = (dtrig_dx * (sigma_su * h7) + dsu_dx * (trig * h7)).transpose();
        return J;
    }

    Matrix<Scalar> cu(const Vector<Scalar>& x, const Vector<Scalar>& u) const override {
        Matrix<Scalar> J = Matrix<Scalar>::Zero(8, u.size());
        const Scalar trig =
            triggerScalar(x, r_follow_.x(), r_follow_.y(), r_follow_.z(), phase_latch_);
        const Scalar sigma_su = triggerSlowUpright(x);
        J(6, 0) = trig * sigma_su;
        J(7, 0) = -trig * sigma_su;
        return J;
    }
};

template<typename Scalar>
class FminCon : public StageConstraintBase<Scalar> {
public:
    FminCon() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }
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
    FmaxCon() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }
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
    Scalar lo_;
    Scalar hi_;

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
    ZFloorCon() {
        this->constraint_type = ConstraintType::NO;
        this->dim_c = 1;
    }
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

inline Param getSolverParams() {
    Param p;
    p.reg1_min = 1e-3;
    p.reg2_min = 0.1;
    p.mu_mul = 0.1;
    p.rho = 1.0;
    p.rhoT = 0.5;
    p.rho_mul = 5.0;
    p.tolerance = 1e-4;
    p.max_iter = 100;
    p.is_quaternion_in_state = false;
    p.use_ddp_terms = false;
    return p;
}

inline std::shared_ptr<OptimalControlProblem<double>> create(
    const Eigen::VectorXd& x0_rel,
    const Eigen::VectorXd& /*terminal_state*/,
    const std::vector<Eigen::VectorXd>& prev_U = {},
    const std::vector<Eigen::VectorXd>& prev_X = {},
    const Eigen::Vector3d& target_accel = Eigen::Vector3d::Zero(),
    const std::vector<Eigen::MatrixXd>& prev_K = {},
    std::shared_ptr<TargetPredictor> predictor = nullptr,
    const Eigen::Vector3d& r_follow = Eigen::Vector3d(-D_BEHIND, 0.0, DZ_ABOVE),
    double phase_latch = 0.0) {
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

    auto tcost = std::make_shared<CaptureAwareTermCost<double>>(
        TERM_W_POS_XY, TERM_W_POS_Z, TERM_W_VEL_XY, TERM_W_VEL_Z, TERM_W_TILT,
        TERM_W_OM, -0.2, r_follow, phase_latch);
    auto cfmin = std::make_shared<FminCon<double>>();
    auto cfmax = std::make_shared<FmaxCon<double>>();
    auto cmom = std::make_shared<MomentCon<double>>();
    auto cth = std::make_shared<ThetaBounds<double>>(THL, THH);
    auto czfl = std::make_shared<ZFloorCon<double>>();
    auto cenv = std::make_shared<CTCSLandingEnvelopeCon<double>>(r_follow, phase_latch);
    auto stage_cost = std::make_shared<CTcSTCStageCost<double>>(
        8.0, W_FOLLOW_XY, W_FOLLOW_Z, r_follow, phase_latch);

    for (int k = 0; k < HORIZON; ++k) {
        problem->setStageDynamics(k, dyn);
        problem->setStageCost(k, stage_cost);
        problem->addStageConstraint(k, cfmin);
        problem->addStageConstraint(k, cfmax);
        problem->addStageConstraint(k, cmom);
        problem->addStageConstraint(k, cth);
        problem->addStageConstraint(k, czfl);
        problem->addStageConstraint(k, cenv);
    }
    problem->setTerminalCost(tcost);

    Eigen::VectorXd x0ss(NX_SS);
    x0ss.segment(0, NX) = x0_rel;
    x0ss(IDX_DT) = 0.0;
    problem->setInitialState(0, x0ss);

    Eigen::VectorXd sim14(NX_SS);
    sim14.segment(0, NX) = x0_rel;
    sim14(IDX_DT) = 0.0;

    const bool use_feedback = (!prev_U.empty() && !prev_X.empty() && !prev_K.empty() &&
                               static_cast<int>(prev_U.size()) >= HORIZON &&
                               static_cast<int>(prev_X.size()) > HORIZON &&
                               static_cast<int>(prev_K.size()) >= HORIZON);

    for (int k = 0; k < HORIZON; ++k) {
        Eigen::VectorXd u0(NU_SS);
        u0.setZero();

        if (use_feedback) {
            const Eigen::VectorXd dx = sim14 - prev_X[k];
            u0 = prev_U[k] + prev_K[k] * dx;
        } else if (!prev_U.empty() && k < static_cast<int>(prev_U.size())) {
            u0 = prev_U[k];
        } else {
            const double phase = std::max(0.0, std::min(1.0, phase_latch));
            const Eigen::Vector3d p0 = sim14.head(3);
            const Eigen::Vector3d v0 = sim14.segment(3, 3);
            const Eigen::Vector3d p_aim = (1.0 - phase) * r_follow;
            double T_g = (HORIZON - k) * TH_INIT;
            if (T_g < TH_INIT) T_g = TH_INIT;
            Eigen::Vector3d a_req = 2.0 * (p_aim - p0 - v0 * T_g) / (T_g * T_g);
            a_req.z() = std::max(a_req.z(), (-VZ_LAND_MAX - v0.z()) / T_g);
            const Eigen::Vector3d a_tgt_ws = predictor->predictAccel(sim14(IDX_DT));
            const Eigen::Vector3d fw = MASS * (a_req - a_tgt_ws - GRAVITY);
            u0(0) = std::max(HOVER_THRUST, std::min(FMAX, fw.norm()));
            u0(IDX_THETA) = TH_INIT;
        }

        if (phase_latch >= 1.0) {
            u0(0) = std::max(T_MIN_AFT, std::min(T_MAX_AFT, static_cast<double>(u0(0))));
        } else {
            u0(0) = std::max(FMIN, std::min(FMAX, static_cast<double>(u0(0))));
        }
        u0(1) = std::max(-TAU_MAX, std::min(TAU_MAX, static_cast<double>(u0(1))));
        u0(2) = std::max(-TAU_MAX, std::min(TAU_MAX, static_cast<double>(u0(2))));
        u0(3) = std::max(-TAU_MAX, std::min(TAU_MAX, static_cast<double>(u0(3))));
        u0(IDX_THETA) = std::max(THL, std::min(THH, static_cast<double>(u0(IDX_THETA))));
        problem->setInitialControl(k, u0);

        Eigen::VectorXd sim14_next = dyn->propagate14(sim14, u0.segment(0, NU), u0(IDX_THETA));
        sim14_next(2) = std::max(sim14_next(2), 0.0);
        problem->setInitialState(k + 1, sim14_next);
        sim14 = sim14_next;
    }

    return problem;
}

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
        ex.predictor->position = tgt_snapshot.position;
        ex.predictor->velocity = tgt_snapshot.velocity;
        ex.predictor->acceleration = tgt_snapshot.acceleration;
        ex.r_follow = computeFollowRef(tgt_snapshot.velocity);
        ex.phase_latch = 0.0;
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
        static bool phase_latched = false;
        static double capture_enter_t = -1.0;
        static double last_t_abs = -1.0;
        StateswitchStcExtra ex;
        if (a.extra.has_value()) {
            try {
                ex = std::any_cast<StateswitchStcExtra>(a.extra);
            } catch (const std::bad_any_cast&) {
            }
        }
        const double now = std::isfinite(a.t_abs) ? a.t_abs : 0.0;
        const bool time_reset = last_t_abs >= 0.0 && now + 0.25 < last_t_abs;
        const bool rearm_from_above =
            phase_latched && a.current_state.size() >= 3 &&
            a.current_state(2) > CAPTURE_LATCH_REARM_Z;
        if (!ex.target_valid || time_reset || rearm_from_above) {
            phase_latched = false;
            capture_enter_t = -1.0;
        } else if (!phase_latched && a.current_state.size() >= 6) {
            const double raw_sigma =
                captureScalarRef<double>(a.current_state, ex.r_follow.x(), ex.r_follow.y(),
                                         ex.r_follow.z(), nullptr);
            if (raw_sigma >= CAPTURE_LATCH_THRESHOLD) {
                if (capture_enter_t < 0.0) {
                    capture_enter_t = now;
                }
                if (now - capture_enter_t >= CAPTURE_LATCH_DWELL_SEC) {
                    phase_latched = true;
                }
            } else {
                capture_enter_t = -1.0;
            }
        }
        last_t_abs = now;
        ex.phase_latch = phase_latched ? 1.0 : 0.0;
        return create(a.current_state, a.terminal_state, a.prev_U, a.prev_X, a.target_accel,
                      a.prev_K, ex.predictor, ex.r_follow, ex.phase_latch);
    };
    return d;
}

}  // namespace StateswitchStcOCP
