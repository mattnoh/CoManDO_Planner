#!/usr/bin/env python3
"""
generate_acados_ocp.py
──────────────────────
Generates the acados C solver code for the quadrotor 6-DOF landing/hover OCP.

Usage:
    source ~/venv/acados/bin/activate
    export ACADOS_SOURCE_DIR=~/programs/acados
    python3 scripts/generate_acados_ocp.py

The generated code lands in:
    <package>/acados_generated/c_generated_code/

Rebuild the ROS2 package afterwards:
    colcon build --packages-select comando_planner
"""

import os
import sys
import numpy as np
from pathlib import Path

# ── acados + CasADi ───────────────────────────────────────────────────────────
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca

# ═══════════════════════════════════════════════════════════════════════════════
#  OCP parameters — must match ocp_landing.hpp exactly
# ═══════════════════════════════════════════════════════════════════════════════
HORIZON  = 100
DT       = 0.1        # seconds per step
MASS     = 0.027      # kg
G        = 9.81       # m/s²

# Inertia (scaled by 1/1.66e-5 ≈ 60240 so diagonal ≈ O(1))
J_SCALE  = 1.0 / 1.66e-5
Jxx = 1.66e-5 * J_SCALE  # ≈ 1.0
Jyy = 1.66e-5 * J_SCALE  # ≈ 1.0
Jzz = 2.92e-5 * J_SCALE  # ≈ 1.759

# Constraints
FMIN        = 0.08     # N
FMAX        = 1.2      # N
GLIDESLOPE  = 70.0     # degrees
THRUST_CONE = 60.0     # degrees

# Cost diagonals (matching standalone that achieved err ≈ 0.0003)
Q_DIAG = np.array([2, 2, 2, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
R_DIAG = np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
P_DIAG = np.array([1000, 1000, 1000, 500, 500, 500, 500, 500, 500, 500, 200, 200, 200])

# Reference state: origin, zero velocity, upright quaternion, zero angular rate
X_REF = np.zeros(13)
X_REF[6] = 1.0   # qw = 1

# Reference control: gravity-compensating hover
U_REF = np.zeros(6)
U_REF[2] = MASS * G

# ═══════════════════════════════════════════════════════════════════════════════
#  CasADi model — Quad6DOF continuous-time dynamics
# ═══════════════════════════════════════════════════════════════════════════════
def quadrotor_model() -> AcadosModel:
    """
    State x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]  (13)
    Control u = [fx, fy, fz, mx, my, mz]                                (6)

    Dynamics (continuous time):
        p_dot = v
        v_dot = (1/m) * R(q) * f + g_vec
        q_dot = 0.5 * Omega(omega) * q
        omega_dot = J^{-1} * (tau - omega x (J * omega))
    """
    model = AcadosModel()
    model.name = "quadrotor"

    # ── Symbolic variables ────────────────────────────────────────────────────
    x = ca.SX.sym("x", 13)
    u = ca.SX.sym("u", 6)
    xdot = ca.SX.sym("xdot", 13)

    # Decompose state
    pos   = x[0:3]
    vel   = x[3:6]
    qw    = x[6]
    qx    = x[7]
    qy    = x[8]
    qz    = x[9]
    omega = x[10:13]

    # Decompose control
    f_body = u[0:3]    # force in body frame
    tau    = u[3:6]    # torque in body frame

    # ── Rotation matrix from quaternion (body → world) ────────────────────────
    R = ca.vertcat(
        ca.horzcat(1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)),
        ca.horzcat(    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)),
        ca.horzcat(    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2))
    )

    # ── Translational dynamics ────────────────────────────────────────────────
    g_vec = ca.vertcat(0, 0, -G)
    pos_dot = vel
    vel_dot = (1.0 / MASS) * R @ f_body + g_vec

    # ── Quaternion kinematics: q_dot = 0.5 * Omega(omega) * q ─────────────────
    wx, wy, wz = omega[0], omega[1], omega[2]
    Omega = ca.vertcat(
        ca.horzcat( 0,  -wx, -wy, -wz),
        ca.horzcat(wx,   0,   wz, -wy),
        ca.horzcat(wy, -wz,   0,   wx),
        ca.horzcat(wz,  wy,  -wx,   0)
    )
    q = ca.vertcat(qw, qx, qy, qz)
    q_dot = 0.5 * Omega @ q

    # ── Rotational dynamics: J * omega_dot = tau - omega x (J * omega) ────────
    J = ca.diag(ca.vertcat(Jxx, Jyy, Jzz))
    J_omega = J @ omega
    omega_dot = ca.solve(J, tau - ca.cross(omega, J_omega))

    # ── Assemble f_expl ───────────────────────────────────────────────────────
    f_expl = ca.vertcat(pos_dot, vel_dot, q_dot, omega_dot)

    model.x    = x
    model.u    = u
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Build the OCP
# ═══════════════════════════════════════════════════════════════════════════════
def create_ocp() -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = quadrotor_model()

    nx = 13
    nu = 6
    N  = HORIZON

    ocp.dims.N = N

    # ── Cost: quadratic (NONLINEAR_LS would also work but LINEAR_LS is simpler)
    # Using EXTERNAL cost so we have full control and can match ALIPDDP exactly.
    # cost = (x - x_ref)^T Q (x - x_ref) + (u - u_ref)^T R (u - u_ref)

    # Stage cost via NONLINEAR_LS: y = [x; u], W = blkdiag(Q, R), y_ref = [x_ref; u_ref]
    ocp.cost.cost_type   = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # Residual: y = [x; u]  (at stages)
    x_sym = ocp.model.x
    u_sym = ocp.model.u
    ocp.model.cost_y_expr   = ca.vertcat(x_sym, u_sym)
    ocp.model.cost_y_expr_e = x_sym  # terminal: only state

    # Weight matrices
    W = np.diag(np.concatenate([Q_DIAG, R_DIAG]))
    W_e = np.diag(P_DIAG)
    ocp.cost.W   = W
    ocp.cost.W_e = W_e

    # Reference
    y_ref = np.concatenate([X_REF, U_REF])
    ocp.cost.yref   = y_ref
    ocp.cost.yref_e = X_REF

    # ── Constraints ───────────────────────────────────────────────────────────
    # Initial state (set to zero — updated at runtime)
    ocp.constraints.x0 = np.zeros(nx)

    # Nonlinear path constraints h(x, u):
    #   h[0] = FMAX - ||f||       ≥ 0   (max thrust)
    #   h[1] = ||f|| - FMIN       ≥ 0   (min thrust)
    #   h[2] = tan(gs) * z       ≥ 0   (glideslope: z > |xy|/tan(gs))
    #   h[3] = (tan(gs)*z)^2 - x^2 - y^2  ≥ 0   (glideslope quadratic form)
    #   h[4] = tan(tc) * fz      ≥ 0   (thrust cone: fz > |fxy|/tan(tc))
    #   h[5] = (tan(tc)*fz)^2 - fx^2 - fy^2  ≥ 0

    # Actually, for acados let's use the simple nonlinear constraint form:
    #   lh ≤ h(x, u) ≤ uh
    #
    # Constraint vector:
    #   h[0] = ||f||              → FMIN ≤ h[0] ≤ FMAX
    #   h[1] = tan(gs)*z - sqrt(x^2 + y^2)  → 0 ≤ h[1] ≤ +inf  (glideslope)
    #   h[2] = tan(tc)*fz - sqrt(fx^2 + fy^2) → 0 ≤ h[2] ≤ +inf  (thrust cone)

    # Use smooth approximation: replace sqrt(a) with sqrt(a + eps) to avoid
    # derivative singularity at zero.
    eps = 1e-8
    tan_gs = np.tan(np.radians(GLIDESLOPE))
    tan_tc = np.tan(np.radians(THRUST_CONE))

    f_body = u_sym[0:3]
    f_norm = ca.sqrt(f_body[0]**2 + f_body[1]**2 + f_body[2]**2 + eps)

    xy_norm = ca.sqrt(x_sym[0]**2 + x_sym[1]**2 + eps)
    glideslope = tan_gs * x_sym[2] - xy_norm

    fxy_norm = ca.sqrt(f_body[0]**2 + f_body[1]**2 + eps)
    thrust_cone = tan_tc * f_body[2] - fxy_norm

    h_expr = ca.vertcat(f_norm, glideslope, thrust_cone)

    ocp.model.con_h_expr = h_expr
    ocp.constraints.lh = np.array([FMIN, 0.0, 0.0])
    ocp.constraints.uh = np.array([FMAX, 1e6, 1e6])

    # ── Solver options ────────────────────────────────────────────────────────
    ocp.solver_options.tf = N * DT                     # total horizon time
    ocp.solver_options.N_horizon = N

    # Integrator: ERK (explicit Runge-Kutta) order 4 — matches Quad6DOF RK4
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4       # RK4
    ocp.solver_options.sim_method_num_steps  = 1       # 1 step per interval

    # NLP solver: SQP_RTI for real-time MPC (1 QP per call)
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 1         # RTI: single iteration

    # QP solver: PARTIAL_CONDENSING_HPIPM (good for medium-size problems)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    # Regularisation
    ocp.solver_options.levenberg_marquardt = 1e-4

    # Tolerances
    ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.nlp_solver_tol_eq   = 1e-4
    ocp.solver_options.nlp_solver_tol_ineq = 1e-4
    ocp.solver_options.nlp_solver_tol_comp = 1e-4

    return ocp


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — generate C code
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Output directory: <package>/acados_generated/
    script_dir = Path(__file__).resolve().parent
    pkg_dir    = script_dir.parent
    out_dir    = pkg_dir / "acados_generated"
    out_dir.mkdir(exist_ok=True)

    # Work inside the output directory so generated files land there
    orig_cwd = os.getcwd()
    os.chdir(str(out_dir))

    ocp = create_ocp()

    # Code-generate the solver
    json_file = str(out_dir / "acados_ocp_quadrotor.json")
    solver = AcadosOcpSolver(ocp, json_file=json_file)

    # ── Quick sanity test ─────────────────────────────────────────────────────
    # Set initial state: hovering at z=1.0
    x0 = np.zeros(13)
    x0[2] = 1.0   # pz = 1m
    x0[6] = 1.0   # qw = 1

    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    # Initialize all stages with hover control
    u_hover = np.zeros(6)
    u_hover[2] = MASS * G
    for k in range(HORIZON):
        solver.set(k, "x", x0)
        solver.set(k, "u", u_hover)
    solver.set(HORIZON, "x", x0)

    # Solve
    status = solver.solve()
    print(f"\n{'='*60}")
    print(f"  Acados solver code generated in: {out_dir}")
    print(f"  Sanity test status: {status} (0 = success)")
    if status == 0:
        x_final = solver.get(HORIZON, "x")
        print(f"  Terminal state: {x_final[:6]}")
    solve_time = solver.get_stats("time_tot")
    print(f"  Solve time: {solve_time*1000:.2f} ms")
    print(f"{'='*60}\n")

    # List generated files
    c_gen = out_dir / "c_generated_code"
    if c_gen.exists():
        print("Generated C files:")
        for f in sorted(c_gen.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(out_dir)}")

    os.chdir(orig_cwd)
    return 0 if status == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
