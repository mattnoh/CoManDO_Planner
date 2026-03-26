"""
animate_landing.py  –  Landing MPC animation (absolute frame only)

CSV inputs (inside --dir folder):
  actual_state.csv  :  timestamp, solve_num, x, y, z, vx, vy, vz,
                        qw, qx, qy, qz, wx, wy, wz
  all_solves.csv    :  solve_num, solve_time_ms, solve_iters, node, t,
                        x, y, z, vx, vy, vz, qw, qx, qy, qz,
                        wx, wy, wz, fz, mx, my, mz

Landing pad is assumed at origin (0, 0, 0).

Usage:
  python animate_landing.py --dir cf_1_landing_mpc_alipddp_20260322_225842
  python animate_landing.py --dir <dir> --out landing.mp4 --fps 30 --speed 1.0
"""

import argparse
import os
import warnings

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")

# ── Colour palette ─────────────────────────────────────────────────────────────
BG            = "#0d1117"
PANE_COL      = "#1a2030"
PANEL_COL     = "#131920"
GROUND_COL    = "#2a442b"
DRONE_COL     = "#58b0f0"
PLAN_COL      = "#8ecaf8"
CONE_COL      = "#c03838"
VIOL_COL      = "#f09040"
CAPTURE_COL   = "#5aba5e"
SOLVETIME_COL = "#8b9cae"
SOLVEITER_COL = "#b07abf"
BODY_X        = "#e84040"
BODY_Y        = "#38c870"
BODY_Z        = "#3898e8"
ARM_COL       = "#3d5068"
RING_COL      = "#5a7090"
TEXT_COL      = "#e6edf3"
GRID_COL      = "#303030"

# ── Physical / constraint parameters ──────────────────────────────────────────
GS_DEG  = 60.0
GS_TAN  = np.tan(np.radians(GS_DEG))
CONE_N  = 40          # number of spokes on glideslope cone
CONE_H  = 3.5        # visualisation height of cone [m]

# ── Drone geometry ─────────────────────────────────────────────────────────────
ARM_LEN    = 0.10
MOTOR_R    = 0.05
N_RING     = 52
RING_T     = np.linspace(0, 2 * np.pi, N_RING)
ARM_ANGLES = np.linspace(np.pi / 4, np.pi / 4 + 2 * np.pi, 4, endpoint=False)


# ── Helpers ────────────────────────────────────────────────────────────────────
def quat_to_rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2) + 1e-12
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
    ])


def _make_seg3d(x, y, z):
    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def _make_drone_artists(ax):
    d = {}
    d["arms"]  = [ax.plot([], [], [], color=ARM_COL, lw=3.5,
                           solid_capstyle="round", zorder=8)[0]
                  for _ in ARM_ANGLES]
    d["rings"] = [ax.plot([], [], [], color=RING_COL, lw=1.4,
                           alpha=0.85, zorder=9)[0]
                  for _ in ARM_ANGLES]
    d["hub_x"], = ax.plot([], [], [], color=ARM_COL, lw=3.0, zorder=10)
    d["hub_y"], = ax.plot([], [], [], color=ARM_COL, lw=3.0, zorder=10)
    return d


def _update_drone(d, pos, R):
    bx = R[:, 0]; by = R[:, 1]
    for k, th in enumerate(ARM_ANGLES):
        tip_b = np.array([np.cos(th), np.sin(th), 0.0]) * ARM_LEN
        tip_w = pos + R @ tip_b
        d["arms"][k].set_data_3d(
            [pos[0], tip_w[0]], [pos[1], tip_w[1]], [pos[2], tip_w[2]])
        rx = tip_w[0] + MOTOR_R * (np.cos(RING_T) * bx[0] + np.sin(RING_T) * by[0])
        ry = tip_w[1] + MOTOR_R * (np.cos(RING_T) * bx[1] + np.sin(RING_T) * by[1])
        rz = tip_w[2] + MOTOR_R * (np.cos(RING_T) * bx[2] + np.sin(RING_T) * by[2])
        d["rings"][k].set_data_3d(rx, ry, rz)
    hs  = ARM_LEN * 0.20
    hx0 = pos + R @ [-hs, 0, 0]; hx1 = pos + R @ [hs, 0, 0]
    hy0 = pos + R @ [0, -hs, 0]; hy1 = pos + R @ [0, hs, 0]
    d["hub_x"].set_data_3d([hx0[0], hx1[0]], [hx0[1], hx1[1]], [hx0[2], hx1[2]])
    d["hub_y"].set_data_3d([hy0[0], hy1[0]], [hy0[1], hy1[1]], [hy0[2], hy1[2]])


def _style_panel(ax, title):
    ax.set_facecolor(PANEL_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
        sp.set_linewidth(1.0)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(data_dir):
    actual = pd.read_csv(os.path.join(data_dir, "actual_state.csv"))
    solves = pd.read_csv(os.path.join(data_dir, "all_solves.csv"))
    return actual, solves


def build_solve_map(solves_df):
    """
    Returns:
      traj_map  : solve_num → (N, 3) planned xyz array
      meta_map  : solve_num → {'ms': float, 'iters': int}
    """
    traj_map = {}
    meta_map = {}
    for sn, grp in solves_df.groupby("solve_num"):
        grp_s = grp.sort_values("node")
        traj_map[int(sn)] = grp_s[["abs_x", "abs_y", "abs_z"]].values
        meta_map[int(sn)] = {
            "ms":    float(grp_s["solve_time_ms"].iloc[0]),
            "iters": int(grp_s["solve_iters"].iloc[0]),
        }
    return traj_map, meta_map


# ── Main animation builder ─────────────────────────────────────────────────────
def build_animation(data_dir, fps=30, speed=1.0, cap=0.15, out_path=None):
    actual_df, solves_df = load_data(data_dir)
    traj_map,  meta_map  = build_solve_map(solves_df)

    n_steps   = len(actual_df)
    dpos      = actual_df[["x", "y", "z"]].values
    vel       = actual_df[["vx", "vy", "vz"]].values
    quats     = actual_df[["qw", "qx", "qy", "qz"]].values
    solve_num = actual_df["solve_num"].values.astype(int)
    time      = actual_df["timestamp"].values
    time      = time - time[0]          # relative time

    # Speed for colouring the trail
    v_mag     = np.linalg.norm(vel, axis=1)
    v_max     = max(float(np.percentile(v_mag, 98)), 0.1)
    vel_cmap  = cm.get_cmap("plasma")
    vel_norm  = mcolors.Normalize(vmin=0, vmax=v_max)

    # Distance to landing pad (origin)
    dist = np.linalg.norm(dpos, axis=1)

    # Glideslope margin  ( ≥ 0 → inside cone )
    dxy       = np.sqrt(dpos[:, 0]**2 + dpos[:, 1]**2)
    dz        = dpos[:, 2]
    gs_margin = GS_TAN * np.maximum(0.0, dz) - dxy
    gs_viol   = gs_margin < 0

    # Solve metadata arrays (sorted by solve number)
    all_sn_ids = sorted(meta_map.keys())
    all_ms     = np.array([meta_map[s]["ms"]    for s in all_sn_ids], dtype=float)
    all_iters  = np.array([meta_map[s]["iters"] for s in all_sn_ids], dtype=float)
    max_ms     = all_ms.max()    * 1.15 if len(all_ms)    else 1.0
    max_iters  = all_iters.max() * 1.15 if len(all_iters) else 1.0

    XY_LIM = max(float(np.abs(dpos[:, :2]).max()) * 1.15, 2.0)
    Z_LIM  = max(float(dpos[:, 2].max())           * 1.15, 2.0)

    # ── Figure / layout ────────────────────────────────────────────────────────
    title_str = os.path.basename(os.path.normpath(data_dir))
    fig = plt.figure(figsize=(22, 11), facecolor=BG)
    fig.suptitle(title_str, color=TEXT_COL, fontsize=11, fontweight="bold", y=0.98)

    gs_main  = fig.add_gridspec(1, 2, width_ratios=[7.5, 2.5],
                                wspace=0.12, left=0.04, right=0.97,
                                top=0.93, bottom=0.07)
    gs_info  = gs_main[1].subgridspec(2, 1, hspace=0.55)
    gs_solve = gs_info[0].subgridspec(1, 2, wspace=0.18, width_ratios=[6, 0.45])

    # ── Absolute 3-D view ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_main[0], projection="3d")
    ax.set_facecolor(BG)
    for attr in ("xaxis", "yaxis", "zaxis"):
        p = getattr(ax, attr).pane
        p.fill = True
        p.set_facecolor(PANE_COL)
        p.set_edgecolor(GRID_COL)
        getattr(ax, attr)._axinfo["grid"]["color"] = GRID_COL
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        lbl.set_color(TEXT_COL); lbl.set_fontsize(10)
    ax.set_title("Absolute Coordinates", color=TEXT_COL, fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("X [m]", labelpad=8)
    ax.set_ylabel("Y [m]", labelpad=8)
    ax.set_zlabel("Z [m]", labelpad=8)
    ax.set_xlim(-XY_LIM, XY_LIM)
    ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_zlim(0, Z_LIM)
    ax.set_box_aspect([2, 2, 1.5])

    # Ground plane
    _gx = np.array([[-XY_LIM, XY_LIM], [-XY_LIM, XY_LIM]])
    _gy = np.array([[-XY_LIM, -XY_LIM], [XY_LIM,  XY_LIM]])
    ax.plot_surface(_gx, _gy, np.zeros_like(_gx),
                    color=GROUND_COL, alpha=0.55, zorder=0,
                    linewidth=0, antialiased=False)

    # Landing pad ring
    _pt = np.linspace(0, 2 * np.pi, 48)
    ax.plot(0.35 * np.cos(_pt), 0.35 * np.sin(_pt), np.zeros(48),
            color=CAPTURE_COL, lw=2.0, alpha=0.80, zorder=2)
    ax.plot([0], [0], [0], "o", color=CAPTURE_COL, markersize=6, zorder=3)

    # Static glideslope cone (apex at origin, opens upward)
    rim_r = GS_TAN * CONE_H
    _rt   = np.linspace(0, 2 * np.pi, 64)
    cone_rim_line, = ax.plot(
        rim_r * np.cos(_rt), rim_r * np.sin(_rt), np.full(64, CONE_H),
        color=CONE_COL, lw=1.6, alpha=0.65, zorder=3)
    _ct = np.linspace(0, 2 * np.pi, CONE_N, endpoint=False)
    cone_spoke_lines = []
    for _th in _ct:
        _rx = rim_r * np.cos(_th); _ry = rim_r * np.sin(_th)
        _line, = ax.plot([0, _rx], [0, _ry], [0, CONE_H],
                          color=CONE_COL, lw=0.9, alpha=0.35, zorder=3)
        cone_spoke_lines.append(_line)

    # Legend
    ax.legend(handles=[
        Line2D([0],[0], color=DRONE_COL,   lw=2.5,           label="Actual trail (speed)"),
        Line2D([0],[0], color=PLAN_COL,    lw=1.5, ls="--",  label="Planned path"),
        Line2D([0],[0], color=CONE_COL,    lw=1.5,           label="Glideslope cone"),
        Line2D([0],[0], color=BODY_X,      lw=2.0,           label="Body X"),
        Line2D([0],[0], color=BODY_Y,      lw=2.0,           label="Body Y"),
        Line2D([0],[0], color=BODY_Z,      lw=2.0,           label="Body Z"),
        Line2D([0],[0], color=CAPTURE_COL, lw=0, marker="o",
               markersize=6,                                  label="Landing pad"),
        Line2D([0],[0], color=VIOL_COL,    lw=0, marker="X",
               markersize=8,                                  label="GS violation"),
    ], facecolor=PANEL_COL, edgecolor=GRID_COL, labelcolor=TEXT_COL,
       fontsize=8, framealpha=0.85, loc="upper right")

    # ── Row 0 left – Solve Time + Iterations (shared plot, twin y-axis) ─────────
    st_ax = fig.add_subplot(gs_solve[0, 0])
    _style_panel(st_ax, "Solve Time & Iterations")
    st_ax.set_xlim(-0.5, max(len(all_sn_ids) - 0.5, 1))
    st_ax.set_ylim(0, max_ms)
    st_ax.set_xlabel("Solve #",   color=TEXT_COL, fontsize=8)
    st_ax.set_ylabel("Time [ms]", color=SOLVETIME_COL, fontsize=8)
    st_ax.tick_params(axis="y", colors=SOLVETIME_COL, labelsize=7)
    st_ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.5)
    bar_time = st_ax.bar(all_sn_ids, all_ms, color=SOLVETIME_COL,
                          alpha=0.55, width=0.7)
    if len(all_ms):
        st_ax.axhline(all_ms.mean(), color=SOLVETIME_COL, lw=1.0, ls="--", alpha=0.5)
    st_cursor = st_ax.axvline(x=0, color=TEXT_COL, lw=1.5, alpha=0.55)
    st_label  = st_ax.text(0.97, 0.97, " ", color=SOLVETIME_COL, fontsize=8,
                            ha="right", va="top", transform=st_ax.transAxes)

    # Twin y-axis for iterations (overlaid on same subplot)
    si_ax = st_ax.twinx()
    si_ax.set_ylim(0, max_iters)
    si_ax.set_ylabel("Iterations", color=SOLVEITER_COL, fontsize=8)
    si_ax.tick_params(axis="y", colors=SOLVEITER_COL, labelsize=7)
    for sp in si_ax.spines.values():
        sp.set_edgecolor(GRID_COL); sp.set_linewidth(1.0)
    bar_iters = si_ax.bar(all_sn_ids, all_iters, color=SOLVEITER_COL,
                           alpha=0.35, width=0.7)
    if len(all_iters):
        si_ax.axhline(all_iters.mean(), color=SOLVEITER_COL, lw=1.0, ls="--", alpha=0.5)
    si_label  = si_ax.text(0.97, 0.82, " ", color=SOLVEITER_COL, fontsize=8,
                            ha="right", va="top", transform=st_ax.transAxes)

    # ── Row 0 right – Speed gradient strip ────────────────────────────────────
    grad_ax = fig.add_subplot(gs_solve[0, 1])
    grad_ax.set_facecolor(PANEL_COL)
    for sp in grad_ax.spines.values():
        sp.set_edgecolor(GRID_COL); sp.set_linewidth(1.0)
    _grad = np.linspace(0, 1, 256).reshape(256, 1)
    grad_ax.imshow(_grad, aspect="auto", cmap=vel_cmap,
                   extent=[0, 1, 0, v_max], origin="lower")
    grad_ax.set_xlim(0, 1); grad_ax.set_ylim(0, v_max)
    grad_ax.set_xticks([])
    _ytv = np.linspace(0, v_max, 5)
    grad_ax.set_yticks(_ytv)
    grad_ax.set_yticklabels([f"{v:.1f}" for v in _ytv], color=TEXT_COL, fontsize=7)
    grad_ax.yaxis.tick_right()
    grad_ax.yaxis.set_tick_params(color=TEXT_COL)
    grad_ax.set_title("Speed\n[m/s]", color=TEXT_COL, fontsize=8,
                       fontweight="bold", pad=4)

    # ── Row 1 – GS Margin + Distance ──────────────────────────────────────────
    gs_ax = fig.add_subplot(gs_info[1, 0])
    _style_panel(gs_ax, "Glideslope Margin & Distance")
    t_end = time[-1] * 1.05 if time[-1] > 0 else 1.0
    gs_ax.set_xlim(0, t_end)
    _gs_lo = min(float(gs_margin.min()) * 1.1, -0.5)
    _gs_hi = max(float(gs_margin.max()) * 1.1,  0.5)
    gs_ax.set_ylim(_gs_lo, _gs_hi)
    gs_ax.set_xlabel("Time [s]",       color=TEXT_COL, fontsize=8)
    gs_ax.set_ylabel("GS Margin [m]",  color=TEXT_COL, fontsize=8)
    gs_ax.axhline(0, color=GRID_COL, lw=1.2, alpha=0.9)
    gs_ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.5)
    gs_line, = gs_ax.plot([], [], color=TEXT_COL, lw=1.5, alpha=0.80, label="GS margin")

    dist_ax = gs_ax.twinx()
    dist_ax.set_ylim(0, dist.max() * 1.1)
    dist_ax.set_ylabel("Distance [m]", color=TEXT_COL, fontsize=8, alpha=0.55)
    dist_ax.tick_params(colors=TEXT_COL, labelsize=7)
    dist_ax.axhline(cap, color=CAPTURE_COL, lw=1.0, linestyle="--", alpha=0.65,
                    label=f"Capture r={cap}m")
    dist_line, = dist_ax.plot([], [], color=TEXT_COL,   lw=1.5, alpha=0.50)
    dist_dot,  = dist_ax.plot([], [], "o", color=DRONE_COL, markersize=4)

    # ── Dynamic 3-D artists ────────────────────────────────────────────────────
    exec_lc = Line3DCollection([], cmap=vel_cmap, norm=vel_norm,
                                lw=2.5, alpha=0.95, zorder=5)
    ax.add_collection3d(exec_lc)

    plan_line, = ax.plot([], [], [], color=PLAN_COL, lw=1.5,
                          alpha=0.55, linestyle="--", zorder=4)

    drone_d  = _make_drone_artists(ax)
    axis_len = 0.80

    bX_line, = ax.plot([], [], [], color=BODY_X, lw=2.2, zorder=11)
    bY_line, = ax.plot([], [], [], color=BODY_Y, lw=2.2, zorder=11)
    bZ_line, = ax.plot([], [], [], color=BODY_Z, lw=2.2, zorder=11)

    viol_dot, = ax.plot([], [], [], "X", color=VIOL_COL,
                         markersize=14, zorder=12, alpha=0.0)

    # ── Animation update ───────────────────────────────────────────────────────
    prev_sn = [-1]

    def update(frame):
        i  = min(frame, n_steps - 1)
        sn = int(solve_num[i])

        qw, qx, qy, qz = quats[i]
        R   = quat_to_rotmat(qw, qx, qy, qz)
        pos = dpos[i]

        # — Drone trail (colour = speed) —
        if i > 1:
            exec_lc.set_segments(
                _make_seg3d(dpos[:i+1, 0], dpos[:i+1, 1], dpos[:i+1, 2]))
            exec_lc.set_array(v_mag[:i+1])

        # — Drone body + body axes —
        _update_drone(drone_d, pos, R)
        endX = pos + axis_len * R[:, 0]
        endY = pos + axis_len * R[:, 1]
        endZ = pos + axis_len * R[:, 2]
        bX_line.set_data_3d([pos[0], endX[0]], [pos[1], endX[1]], [pos[2], endX[2]])
        bY_line.set_data_3d([pos[0], endY[0]], [pos[1], endY[1]], [pos[2], endY[2]])
        bZ_line.set_data_3d([pos[0], endZ[0]], [pos[1], endZ[1]], [pos[2], endZ[2]])

        # — Planned path from current solve —
        if sn in traj_map:
            plan = traj_map[sn]
            plan_line.set_data_3d(plan[:, 0], plan[:, 1], plan[:, 2])

        # — Glideslope cone alpha (fade in as drone approaches) —
        cone_alpha = min(1.0, max(0.1, 1.0 - dist[i] / (XY_LIM * 1.5)))
        cone_rim_line.set_alpha(0.65 * cone_alpha)
        for _sl in cone_spoke_lines:
            _sl.set_alpha(0.35 * cone_alpha)

        # — GS violation marker —
        is_viol = bool(gs_viol[i])
        viol_dot.set_data_3d([pos[0]], [pos[1]], [pos[2]])
        viol_dot.set_alpha(0.90 if is_viol else 0.0)

        # — Solve Time + Iterations bars —
        if sn != prev_sn[0]:
            for ridx, (rt, ri) in enumerate(zip(bar_time, bar_iters)):
                _sn = all_sn_ids[ridx] if ridx < len(all_sn_ids) else ridx
                a = 0.92 if _sn == sn else (0.50 if _sn < sn else 0.15)
                rt.set_alpha(a)
                ri.set_alpha(a)
            prev_sn[0] = sn

        st_cursor.set_xdata([sn, sn])
        ms_now    = meta_map.get(sn, {}).get("ms",    0.0)
        iters_now = meta_map.get(sn, {}).get("iters", 0)
        st_label.set_text(f"{ms_now:.1f} ms")
        si_label.set_text(f"{iters_now} it")

        # — GS margin + distance —
        gs_line.set_data(time[:i+1], gs_margin[:i+1])
        dist_line.set_data(time[:i+1], dist[:i+1])
        dist_dot.set_data([time[i]], [dist[i]])

        return (
            [exec_lc, plan_line,
             bX_line, bY_line, bZ_line, viol_dot,
             cone_rim_line,
             st_cursor, st_label, si_label,
             gs_line, dist_line, dist_dot]
            + list(drone_d["arms"])
            + list(drone_d["rings"])
            + [drone_d["hub_x"], drone_d["hub_y"]]
            + cone_spoke_lines
            + list(bar_time)
            + list(bar_iters)
        )

    interval_ms = 1000.0 / (fps * speed)
    ani = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=interval_ms, blit=False)

    if out_path is None:
        plt.show()
    elif out_path.endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
        print(f"Saving GIF → {out_path}")
        ani.save(out_path, writer=writer, dpi=100,
                 savefig_kwargs={"facecolor": BG})
    elif out_path.lower().endswith((".jpg", ".jpeg")):
        update(n_steps - 1)
        fig.savefig(out_path, dpi=150, facecolor=BG, format="jpeg")
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
        print(f"Saving MP4 → {out_path}")
        ani.save(out_path, writer=writer, dpi=150,
                 savefig_kwargs={"facecolor": BG})

    return ani


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landing MPC animation")
    parser.add_argument("--dir",   required=True,
                        help="Data directory containing actual_state.csv and all_solves.csv")
    parser.add_argument("--out",   default=None,
                        help="Output file (.mp4 / .gif / .jpg). Omit to show interactively.")
    parser.add_argument("--fps",   type=int,   default=30)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (>1 = faster)")
    parser.add_argument("--cap",   type=float, default=0.15,
                        help="Capture radius [m] shown as dashed line")
    args = parser.parse_args()

    actual_df, solves_df = load_data(args.dir)
    print(f"  Actual states  : {len(actual_df)} rows")
    print(f"  Unique solves  : {solves_df['solve_num'].nunique()}")

    build_animation(args.dir, fps=args.fps, speed=args.speed,
                    cap=args.cap, out_path=args.out)