"""
plot_landing.py  –  Static multi-page PDF analysis for landing MPC runs

CSV inputs (inside --dir folder):
  actual_state.csv : timestamp, solve_num, x, y, z, vx, vy, vz,
                     qw, qx, qy, qz, wx, wy, wz
  all_solves.csv   : solve_num, solve_time_ms, solve_iters, node, t,
                     x, y, z, vx, vy, vz, qw, qx, qy, qz,
                     wx, wy, wz, fz, mx, my, mz

Output PDF pages (one per figure):
  Page 1  — 3-D trajectory  (4 azimuth views, 2×2)
  Page 2  — Position states  (x, y, z vs time)
  Page 3  — Velocity states  (vx, vy, vz vs time)
  Page 4  — Quaternion       (qw, qx, qy, qz vs time)
  Page 5  — Angular velocity (wx, wy, wz vs time)
  Page 6  — Position error   (time series + box plot)
  Page 7  — Velocity error   (time series + box plot)
  Page 8  — Solver performance (solve time + iterations)

Usage:
  python plot_landing.py --dir cf_1_landing_mpc_alipddp_20260322_225842
  python plot_landing.py --dir <dir> --out custom_name.pdf --elev 25
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

warnings.filterwarnings("ignore")

# ── Colour palette ─────────────────────────────────────────────────────────────
BG         = "#ffffff"
FIG_BG     = "#f1f1f1"
PANEL_BG   = "#E9E9E9"
PANE_COL   = "#d1d1d1"
GRID_COL   = "#a5a5a5"
GROUND_COL = "#7cb47e"
CONE_COL   = "#c03838"
CAPTURE_COL= "#5aba5e"
TEXT_COL   = "#000000"
SPINE_COL  = "#303848"

# Line colours
C_ACTUAL   = "#58b0f0"   # actual state  – solid
C_CMD      = "#f0c060"   # commanded     – dashed
C_FIRST    = "#a0b8d0"   # first solve   – dotted, faint
C_ERR      = "#f07070"   # error signal

# Body-frame colours
BODY_X = "#e84040"
BODY_Y = "#38c870"
BODY_Z = "#3898e8"

# Component colours for multi-line state plots
COMP_COLS = ["#e84040", "#38c870", "#3898e8", "#f0c060"]   # x/y/z/(w)

GS_DEG  = 60.0
GS_TAN  = np.tan(np.radians(GS_DEG))
CONE_H  = 0.0
BODY_AZSCALE = 0.4    # length of body-frame arrows in 3-D plot [m]

AZIM_VIEWS = [30, 120, -60, -120]   # four azimuth angles for 3-D views
ELEV_VIEW  = 22                     # fixed elevation


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def quat_to_rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2) + 1e-12
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=11)
    ax.xaxis.label.set_color(TEXT_COL); ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_color(TEXT_COL); ax.yaxis.label.set_fontsize(12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, color=TEXT_COL, fontsize=14, fontweight="bold", pad=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL); sp.set_linewidth(1.2)
    ax.grid(True, color=GRID_COL, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _style_3d(ax, title):
    ax.set_facecolor(FIG_BG)
    for attr in ("xaxis", "yaxis", "zaxis"):
        p = getattr(ax, attr).pane
        p.fill = True
        p.set_facecolor(PANE_COL)
        p.set_edgecolor(SPINE_COL)
        getattr(ax, attr)._axinfo["grid"]["color"] = GRID_COL
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        lbl.set_color(TEXT_COL); lbl.set_fontsize(9)
    ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold", pad=8)


def _legend(ax, handles, **kw):
    defaults = dict(facecolor=PANEL_BG, edgecolor=SPINE_COL,
                    labelcolor=TEXT_COL, fontsize=13, framealpha=0.9)
    defaults.update(kw)
    ax.legend(handles=handles, **defaults)


def _fig(suptitle, figsize=(16, 10)):
    fig = plt.figure(figsize=figsize, facecolor=FIG_BG)
    fig.suptitle(suptitle, color=TEXT_COL, fontsize=13,
                 fontweight="bold", y=0.98)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading & preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir):
    actual = pd.read_csv(os.path.join(data_dir, "actual_state.csv"))
    solves = pd.read_csv(os.path.join(data_dir, "all_solves.csv"))
    return actual, solves


def compute_commanded_states(actual_df, solves_df):
    """
    For each row in actual_df (with a given solve_num), pull the corresponding
    planned node from all_solves.csv.  Node counter advances independently for
    each solve, so solve transitions always restart at node 0.
    """
    state_cols = ["x", "y", "z", "vx", "vy", "vz",
                  "qw", "qx", "qy", "qz", "wx", "wy", "wz"]

    # all_solves.csv now uses abs_ prefix for absolute-frame state columns
    _abs = {"x": "abs_x", "y": "abs_y", "z": "abs_z",
            "vx": "abs_vx", "vy": "abs_vy", "vz": "abs_vz",
            "qw": "abs_qw", "qx": "abs_qx", "qy": "abs_qy", "qz": "abs_qz",
            "wx": "abs_wx", "wy": "abs_wy", "wz": "abs_wz"}

    # Build dict: solve_num → sorted DataFrame of nodes
    solve_groups = {int(sn): grp.sort_values("node").reset_index(drop=True)
                    for sn, grp in solves_df.groupby("solve_num")}

    nan_row = {c: np.nan for c in state_cols}
    node_idx = {}
    cmd_rows  = []

    for _, row in actual_df.iterrows():
        sn = int(row["solve_num"])
        if sn not in solve_groups:
            cmd_rows.append(nan_row.copy())
            continue
        if sn not in node_idx:
            node_idx[sn] = 0
        grp  = solve_groups[sn]
        nidx = min(node_idx[sn], len(grp) - 1)
        r    = grp.iloc[nidx]
        cmd_rows.append({c: r[_abs[c]] if _abs[c] in r.index else np.nan for c in state_cols})
        node_idx[sn] += 1

    return pd.DataFrame(cmd_rows)


def get_first_solve_traj(solves_df):
    """Full trajectory from the earliest solve_num."""
    first_sn = solves_df["solve_num"].min()
    grp = solves_df[solves_df["solve_num"] == first_sn].sort_values("node")
    return grp.reset_index(drop=True)


def find_solve_update_indices(actual_df):
    """Indices where solve_num changes (= replanning instants)."""
    sn = actual_df["solve_num"].values
    idx = np.where(np.diff(sn) != 0)[0] + 1
    return np.concatenate([[0], idx])


# ═══════════════════════════════════════════════════════════════════════════════
#  Page 1 – 3-D trajectory (4 views)
# ═══════════════════════════════════════════════════════════════════════════════

def page_3d_views(actual_df, solves_df, cmd_df, first_traj, title_str):
    dpos  = actual_df[["x","y","z"]].values
    quats = actual_df[["qw","qx","qy","qz"]].values
    time  = actual_df["timestamp"].values
    time  = time - time[0]

    cmd_pos   = cmd_df[["x","y","z"]].values
    first_pos = first_traj[["abs_x","abs_y","abs_z"]].values

    # Speed magnitude for colormap
    vel  = actual_df[["vx","vy","vz"]].values
    vmag = np.linalg.norm(vel, axis=1)

    replan_idx = find_solve_update_indices(actual_df)

    XY_LIM = max(float(np.abs(dpos[:, :2]).max()) * 1.2, 1.5)
    Z_LIM  = max(float(dpos[:, 2].max())           * 1.2, 1.5)


    # --- Target circle parameters (for tracking_circle OCP) -------------------
    # Try to infer from all_solves.csv if present, else use defaults
    # (center_x, center_y, center_z, R, omega, phi0)
    center = np.array([0.0, 0.0, 1.0])
    R = 3.0
    omega = 0.5
    phi0 = 0.0
    # Try to infer from solves_df if columns exist
    for col in solves_df.columns:
        if col.startswith("tgt_"):
            # If target columns exist, use first row as reference
            center = np.array([
                solves_df.get("tgt_x", pd.Series([0])).iloc[0],
                solves_df.get("tgt_y", pd.Series([0])).iloc[0],
                1.0,
                ])
            R = solves_df.get("tgt_R", pd.Series([R])).iloc[0] if "tgt_R" in solves_df.columns else R
            omega = solves_df.get("tgt_omega", pd.Series([omega])).iloc[0] if "tgt_omega" in solves_df.columns else omega
            phi0 = solves_df.get("tgt_phi0", pd.Series([phi0])).iloc[0] if "tgt_phi0" in solves_df.columns else phi0
            break

    # Compute target circle trajectory over the time span
    t_span = np.linspace(0, actual_df["timestamp"].values[-1] - actual_df["timestamp"].values[0], 500)
    circ_x = center[0] + R * np.cos(omega * t_span + phi0)
    circ_y = center[1] + R * np.sin(omega * t_span + phi0)
    circ_z = np.full_like(t_span, center[2])

    # --- Glideslope cone geometry (static, apex at origin) --------------------
    rim_r  = GS_TAN * CONE_H
    _rt    = np.linspace(0, 2*np.pi, 64)
    rim_x  = rim_r * np.cos(_rt);  rim_y = rim_r * np.sin(_rt)
    _ct    = np.linspace(0, 2*np.pi, 12, endpoint=False)

    fig = _fig(f"{title_str}  —  3-D Trajectory (4 views)", figsize=(20, 14))
    axs = []
    for k, azim in enumerate(AZIM_VIEWS):
        ax = fig.add_subplot(2, 2, k+1, projection="3d")
        _style_3d(ax, f"Azimuth = {azim}°")
        ax.view_init(elev=ELEV_VIEW, azim=azim)
        ax.set_xlim(-XY_LIM, XY_LIM)
        ax.set_ylim(-XY_LIM, XY_LIM)
        ax.set_zlim(0, Z_LIM)
        ax.set_box_aspect([2, 2, 1.5])
        ax.set_xlabel("X [m]", labelpad=6)
        ax.set_ylabel("Y [m]", labelpad=6)
        ax.set_zlabel("Z [m]", labelpad=6)

        # Ground plane
        _gx = np.array([[-XY_LIM, XY_LIM], [-XY_LIM, XY_LIM]])
        _gy = np.array([[-XY_LIM, -XY_LIM], [XY_LIM, XY_LIM]])
        ax.plot_surface(_gx, _gy, np.zeros_like(_gx),
                        color=GROUND_COL, alpha=0.4, linewidth=0, antialiased=False)

        # Landing pad
        _pt = np.linspace(0, 2*np.pi, 48)
        ax.plot(0.35*np.cos(_pt), 0.35*np.sin(_pt), np.zeros(48),
                color=CAPTURE_COL, lw=2.0, alpha=0.85)
        ax.plot([0], [0], [0], "o", color=CAPTURE_COL, markersize=7)

        # Glideslope cone
        ax.plot(rim_x, rim_y, np.full(64, CONE_H),
                color=CONE_COL, lw=1.4, alpha=0.55)
        for _th in _ct:
            ax.plot([0, rim_r*np.cos(_th)], [0, rim_r*np.sin(_th)], [0, CONE_H],
                    color=CONE_COL, lw=0.7, alpha=0.30)


        # Target circle trajectory (thick magenta)
        ax.plot(circ_x, circ_y, circ_z, color="#d12be6", lw=2.5, ls="-", alpha=0.85, zorder=2, label="Target circle")

        # First-solve reference (dotted, faint)
        ax.plot(first_pos[:, 0], first_pos[:, 1], first_pos[:, 2],
            color=C_FIRST, lw=1.4, ls=":", alpha=0.55, zorder=3)

        # Commanded / stitched path (dashed)
        ax.plot(cmd_pos[:, 0], cmd_pos[:, 1], cmd_pos[:, 2],
            color=C_CMD, lw=1.5, ls="--", alpha=0.70, zorder=4)

        # Actual trajectory (solid, coloured by speed)
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        v_max   = max(float(vmag.max()), 0.1)
        vnorm   = Normalize(vmin=0, vmax=v_max)
        cmap    = cm.plasma
        for j in range(len(dpos) - 1):
            c = cmap(vnorm(vmag[j]))
            ax.plot(dpos[j:j+2, 0], dpos[j:j+2, 1], dpos[j:j+2, 2],
                    color=c, lw=2.2, alpha=0.90, zorder=5)

        # Body-frame axes at replanning instants
        for ridx in replan_idx:
            if ridx >= len(dpos):
                continue
            pos = dpos[ridx]
            R   = quat_to_rotmat(*quats[ridx])
            s   = BODY_AZSCALE
            ax.quiver(*pos, *(R[:, 0]*s), color=BODY_X, lw=1.5,
                       arrow_length_ratio=0.35, alpha=0.85)
            ax.quiver(*pos, *(R[:, 1]*s), color=BODY_Y, lw=1.5,
                       arrow_length_ratio=0.35, alpha=0.85)
            ax.quiver(*pos, *(R[:, 2]*s), color=BODY_Z, lw=1.5,
                       arrow_length_ratio=0.35, alpha=0.85)

        # Start / end markers
        ax.plot(*dpos[0], "o", color=C_ACTUAL, markersize=9, zorder=10)
        ax.plot(*dpos[-1], "*", color=C_ACTUAL, markersize=12, zorder=10)


        # Legend (only on first subplot)
        if k == 0:
            handles = [
                Line2D([0],[0], color=C_ACTUAL, lw=2.2,           label="Actual trajectory"),
                Line2D([0],[0], color="#d12be6", lw=2.5,         label="Target circle"),
                Line2D([0],[0], color=C_CMD,    lw=1.5, ls="--",  label="Commanded (stitched)"),
                Line2D([0],[0], color=C_FIRST,  lw=1.4, ls=":",   label="First solve plan"),
                Line2D([0],[0], color=CONE_COL, lw=1.4,           label="Glideslope cone"),
                Line2D([0],[0], color=BODY_X,   lw=2.0,           label="Body X (at replan)"),
                Line2D([0],[0], color=BODY_Y,   lw=2.0,           label="Body Y (at replan)"),
                Line2D([0],[0], color=BODY_Z,   lw=2.0,           label="Body Z (at replan)"),
                Line2D([0],[0], color=CAPTURE_COL, lw=0, marker="o",
                       markersize=8,                               label="Landing pad"),
            ]
            _legend(ax, handles, loc="upper right", fontsize=10)

        axs.append(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Pages 2-5 – State plots
# ═══════════════════════════════════════════════════════════════════════════════

def _state_page(time, actual_vals, cmd_vals, first_time, first_vals,
                labels, units, page_title, fig_title):
    """
    Generic state-plot page.

    Parameters
    ----------
    time         : (N,) actual timestamps (seconds, relative to start)
    actual_vals  : (N, k)  actual state components
    cmd_vals     : (N, k)  commanded state components
    first_time   : (M,)    time axis for first solve
    first_vals   : (M, k)  first solve planned state
    labels       : list[str]  component names, e.g. ['x','y','z']
    units        : str        y-axis unit label
    page_title   : str        suptitle
    fig_title    : str        axis title
    """
    fig = _fig(page_title, figsize=(16, 8))
    ax  = fig.add_subplot(1, 1, 1)
    _style_ax(ax, xlabel="Time [s]", ylabel=units, title=fig_title)

    n_comp = actual_vals.shape[1]
    handles = []
    for k in range(n_comp):
        col = COMP_COLS[k % len(COMP_COLS)]
        lbl = labels[k]

        # First solve (dotted, faint)
        if first_vals is not None and k < first_vals.shape[1]:
            ax.plot(first_time, first_vals[:, k],
                    color=col, lw=1.5, ls=":", alpha=0.40)

        # Commanded (dashed)
        ax.plot(time, cmd_vals[:, k],
                color=col, lw=1.8, ls="--", alpha=0.70)

        # Actual (solid)
        ax.plot(time, actual_vals[:, k],
                color=col, lw=2.2, ls="-", alpha=0.95)

        handles += [
            Line2D([0],[0], color=col, lw=2.2, ls="-",  label=f"{lbl}  actual"),
            Line2D([0],[0], color=col, lw=1.8, ls="--", label=f"{lbl}  commanded"),
            Line2D([0],[0], color=col, lw=1.5, ls=":",
                   alpha=0.55,                           label=f"{lbl}  first solve"),
        ]

    _legend(ax, handles, loc="best", ncol=max(1, n_comp))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Pages 6-7 – Error plots (time series + box plot)
# ═══════════════════════════════════════════════════════════════════════════════

def _error_page(time, actual_vals, cmd_vals, labels, units,
                page_title, fig_title):
    """
    Two-panel layout:
      Top    – error time series for each component
      Bottom – box-and-whisker for the error distribution
    """
    err = actual_vals - cmd_vals          # (N, k)
    # drop NaN rows (where commanded is unavailable)
    valid = ~np.isnan(err).any(axis=1)
    err_clean = err[valid]
    t_clean   = time[valid]

    fig  = _fig(page_title, figsize=(16, 11))
    gs   = gridspec.GridSpec(2, 1, hspace=0.45, figure=fig,
                             top=0.92, bottom=0.07, left=0.08, right=0.95)
    ax_ts  = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1])
    _style_ax(ax_ts,  "Time [s]", f"Error [{units}]",
              title=f"{fig_title}  –  Error Time Series")
    _style_ax(ax_box, "State component", f"Error [{units}]",
              title=f"{fig_title}  –  Error Distribution")

    n_comp = actual_vals.shape[1]
    handles_ts = []

    for k in range(n_comp):
        col = COMP_COLS[k % len(COMP_COLS)]
        # Time series
        ax_ts.plot(t_clean, err_clean[:, k], color=col, lw=1.8, alpha=0.85,
                   label=labels[k])
        handles_ts.append(Line2D([0],[0], color=col, lw=2.0, label=labels[k]))

    ax_ts.axhline(0, color=TEXT_COL, lw=1.0, alpha=0.6, ls="--")
    _legend(ax_ts, handles_ts, loc="best", fontsize=13)

    # Box plot
    bp_data = [err_clean[:, k] for k in range(n_comp)]
    bplot = ax_box.boxplot(
        bp_data,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color=TEXT_COL, lw=2.5),
        whiskerprops=dict(color=TEXT_COL, lw=1.5),
        capprops=dict(color=TEXT_COL, lw=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, k in zip(bplot["boxes"], range(n_comp)):
        col = COMP_COLS[k % len(COMP_COLS)]
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor(TEXT_COL)
    for flier, k in zip(bplot["fliers"], range(n_comp)):
        flier.set_markerfacecolor(COMP_COLS[k % len(COMP_COLS)])
        flier.set_markeredgecolor(COMP_COLS[k % len(COMP_COLS)])

    ax_box.set_xticks(range(1, n_comp+1))
    ax_box.set_xticklabels(labels, fontsize=13, color=TEXT_COL)
    ax_box.axhline(0, color=TEXT_COL, lw=1.0, alpha=0.6, ls="--")

    # Annotate MAE
    for k in range(n_comp):
        mae = float(np.mean(np.abs(err_clean[:, k])))
        ax_box.text(k + 1, ax_box.get_ylim()[1] * 0.90,
                    f"MAE\n{mae:.3f}", ha="center", va="top",
                    color=TEXT_COL, fontsize=10)

    handles_box = [Line2D([0],[0], color=COMP_COLS[k % len(COMP_COLS)],
                           lw=0, marker="s", markersize=12,
                           label=f"{labels[k]}  (MAE={np.mean(np.abs(err_clean[:,k])):.3f} {units})")
                   for k in range(n_comp)]
    _legend(ax_box, handles_box, loc="upper left", fontsize=12)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Page 8 – Solver performance  (solve time [ms] + iterations, twin y-axes)
# ═══════════════════════════════════════════════════════════════════════════════

SOLVETIME_COL = "#8b9cae"
SOLVEITER_COL = "#b07abf"

def page_solver_stats(solves_df, title_str):
    """
    Bar chart with two overlaid series sharing the Solve # x-axis:
      left  y-axis  – solve_time_ms   (blue-grey bars)
      right y-axis  – solve_iters     (purple bars, more transparent)
    Mean lines are drawn for each series.
    """
    # One row per solve (first row inside each group carries the metadata)
    meta      = solves_df.groupby("solve_num").first().reset_index()
    sn_ids    = meta["solve_num"].values.astype(int)
    ms_vals   = meta["solve_time_ms"].values.astype(float)
    iter_vals = meta["solve_iters"].values.astype(float)

    max_ms    = ms_vals.max()   * 1.15 if len(ms_vals)   else 1.0
    max_iters = iter_vals.max() * 1.15 if len(iter_vals) else 1.0

    fig = _fig(f"{title_str}  —  Solver Performance", figsize=(16, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    _style_ax(ax1, xlabel="Solve #", ylabel="Solve Time [ms]",
              title="Solve Time & Iterations per Solve")
    ax1.set_xlim(-0.5, max(len(sn_ids) - 0.5, 1))
    ax1.set_ylim(0, max_ms)
    ax1.yaxis.label.set_color(SOLVETIME_COL)
    ax1.tick_params(axis="y", colors=SOLVETIME_COL)

    ax1.bar(sn_ids, ms_vals, color=SOLVETIME_COL, alpha=0.60, width=0.7,
            zorder=3, label="Solve time [ms]")
    if len(ms_vals):
        ax1.axhline(ms_vals.mean(), color=SOLVETIME_COL,
                    lw=1.4, ls="--", alpha=0.70, zorder=4)

    ax2 = ax1.twinx()
    ax2.set_ylim(0, max_iters)
    ax2.set_ylabel("Iterations", color=SOLVEITER_COL, fontsize=12)
    ax2.tick_params(axis="y", colors=SOLVEITER_COL)
    for sp in ax2.spines.values():
        sp.set_edgecolor(SPINE_COL); sp.set_linewidth(1.2)

    ax2.bar(sn_ids, iter_vals, color=SOLVEITER_COL, alpha=0.35, width=0.7,
            zorder=2, label="Iterations")
    if len(iter_vals):
        ax2.axhline(iter_vals.mean(), color=SOLVEITER_COL,
                    lw=1.4, ls="--", alpha=0.70, zorder=4)

    handles = [
        Line2D([0],[0], color=SOLVETIME_COL, lw=8, alpha=0.60,
               label=f"Solve time [ms]  (mean = {ms_vals.mean():.1f} ms)"),
        Line2D([0],[0], color=SOLVEITER_COL, lw=8, alpha=0.50,
               label=f"Iterations  (mean = {iter_vals.mean():.1f})"),
    ]
    _legend(ax1, handles, loc="upper left")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Master builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_plots(data_dir, out_path=None, elev=22):
    global ELEV_VIEW
    ELEV_VIEW = elev

    actual_df, solves_df = load_data(data_dir)
    cmd_df               = compute_commanded_states(actual_df, solves_df)
    first_traj           = get_first_solve_traj(solves_df)

    title_str = os.path.basename(os.path.normpath(data_dir))

    if out_path is None:
        out_path = os.path.join(data_dir, f"{title_str}_analysis.pdf")

    # ── Time axes ──────────────────────────────────────────────────────────────
    time = actual_df["timestamp"].values.astype(float)
    time = time - time[0]

    first_t = first_traj["t"].values.astype(float)
    if first_t[0] != 0:
        first_t = first_t - first_t[0]

    # ── State arrays ───────────────────────────────────────────────────────────
    actual_pos  = actual_df[["x","y","z"]].values
    actual_vel  = actual_df[["vx","vy","vz"]].values
    actual_quat = actual_df[["qw","qx","qy","qz"]].values
    actual_angv = actual_df[["wx","wy","wz"]].values

    cmd_pos     = cmd_df[["x","y","z"]].values
    cmd_vel     = cmd_df[["vx","vy","vz"]].values
    cmd_quat    = cmd_df[["qw","qx","qy","qz"]].values
    cmd_angv    = cmd_df[["wx","wy","wz"]].values

    first_pos   = first_traj[["abs_x","abs_y","abs_z"]].values
    first_vel   = first_traj[["abs_vx","abs_vy","abs_vz"]].values
    first_quat  = first_traj[["abs_qw","abs_qx","abs_qy","abs_qz"]].values
    first_angv  = first_traj[["abs_wx","abs_wy","abs_wz"]].values

    print(f"  Building PDF → {out_path}")
    with PdfPages(out_path) as pdf:

        # ── Page 1 : 3-D views ─────────────────────────────────────────────────
        print("  [1/8] 3-D views …")
        fig1 = page_3d_views(actual_df, solves_df, cmd_df, first_traj, title_str)
        pdf.savefig(fig1, facecolor=FIG_BG)
        plt.close(fig1)

        # ── Page 2 : Position ─────────────────────────────────────────────────
        print("  [2/8] Position states …")
        fig2 = _state_page(
            time, actual_pos, cmd_pos,
            first_t, first_pos,
            labels=["x", "y", "z"], units="m",
            page_title=f"{title_str}  —  Position",
            fig_title="Position  [x, y, z]")
        pdf.savefig(fig2, facecolor=FIG_BG); plt.close(fig2)

        # ── Page 3 : Velocity ─────────────────────────────────────────────────
        print("  [3/7] Velocity states …")
        fig3 = _state_page(
            time, actual_vel, cmd_vel,
            first_t, first_vel,
            labels=["vx", "vy", "vz"], units="m/s",
            page_title=f"{title_str}  —  Velocity",
            fig_title="Velocity  [vx, vy, vz]")
        pdf.savefig(fig3, facecolor=FIG_BG); plt.close(fig3)

        # ── Page 4 : Quaternion ───────────────────────────────────────────────
        print("  [4/7] Quaternion states …")
        fig4 = _state_page(
            time, actual_quat, cmd_quat,
            first_t, first_quat,
            labels=["qw", "qx", "qy", "qz"], units="–",
            page_title=f"{title_str}  —  Quaternion",
            fig_title="Attitude quaternion  [qw, qx, qy, qz]")
        pdf.savefig(fig4, facecolor=FIG_BG); plt.close(fig4)

        # ── Page 5 : Angular velocity ─────────────────────────────────────────
        print("  [5/7] Angular velocity states …")
        fig5 = _state_page(
            time, actual_angv, cmd_angv,
            first_t, first_angv,
            labels=["wx", "wy", "wz"], units="rad/s",
            page_title=f"{title_str}  —  Angular Velocity",
            fig_title="Angular velocity  [wx, wy, wz]")
        pdf.savefig(fig5, facecolor=FIG_BG); plt.close(fig5)

        # ── Page 6 : Position error ───────────────────────────────────────────
        print("  [6/7] Position error …")
        fig6 = _error_page(
            time, actual_pos, cmd_pos,
            labels=["x", "y", "z"], units="m",
            page_title=f"{title_str}  —  Position Tracking Error",
            fig_title="Position error  (actual − commanded)")
        pdf.savefig(fig6, facecolor=FIG_BG); plt.close(fig6)

        # ── Page 7 : Velocity error ───────────────────────────────────────────
        print("  [7/7] Velocity error …")
        fig7 = _error_page(
            time, actual_vel, cmd_vel,
            labels=["vx", "vy", "vz"], units="m/s",
            page_title=f"{title_str}  —  Velocity Tracking Error",
            fig_title="Velocity error  (actual − commanded)")
        pdf.savefig(fig7, facecolor=FIG_BG); plt.close(fig7)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = f"Landing MPC analysis: {title_str}"
        d["Subject"] = "MPC landing trajectory analysis"

    print(f"  ✓  Saved {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-page PDF analysis for landing MPC run")
    parser.add_argument("--dir",  required=True,
                        help="Data directory containing actual_state.csv and all_solves.csv")
    parser.add_argument("--out",  default=None,
                        help="Output PDF path (default: <dir>/<run_name>_analysis.pdf)")
    parser.add_argument("--elev", type=float, default=22,
                        help="3-D view elevation angle [deg] (default: 22)")
    args = parser.parse_args()

    actual_df, solves_df = load_data(args.dir)
    print(f"  Actual states  : {len(actual_df)} rows")
    print(f"  Unique solves  : {solves_df['solve_num'].nunique()}")

    build_plots(args.dir, out_path=args.out, elev=args.elev)
    