"""
plot_landing.py  –  Static multi-page PDF analysis for landing MPC runs

CSV inputs (inside --dir folder):
  actual_state.csv : timestamp, solve_num, x, y, z, vx, vy, vz,
                     qw, qx, qy, qz, wx, wy, wz
  all_solves.csv   : solve_num, solve_time_ms, solve_iters, node, t,
                     x, y, z, vx, vy, vz, qw, qx, qy, qz,
                     wx, wy, wz, fz, mx, my, mz

Output PDF pages (one per figure):
  Page 1  — 3-D trajectory  (2 × 2: top = Absolute coords, bottom = Relative coords)
  Page 2  — Position states  (absolute: x, y, z vs time)
  Page 3  — Velocity states  (absolute: vx, vy, vz vs time)
  Page 4  — Quaternion       (qw, qx, qy, qz vs time)
  Page 5  — Angular velocity (wx, wy, wz vs time)
  Page 6  — Relative position & velocity states  (rx, ry, rz  and  rvx, rvy, rvz)
  Page 7  — Target Tracking  (actual vs target position components)
  Page 8  — Position error   (time series + box plot)
  Page 9  — Velocity error   (time series + box plot)
  Page 10 — Solver performance (solve time + iterations)
  Page 11 — Control Inputs & Timing (fz, moments, node intervals)

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
CONE_H  = 1.5           # matches target height
BODY_AZSCALE = 0.4      # length of body-frame arrows in 3-D plot [m]

# --- OCP Parameters from ocp_tracking_circle.hpp ---
MASS        = 0.027     # [kg]
VZ_REF      = -0.15     # [m/s] reference descent
VZ_LAND_MAX = 2.5       # [m/s]
FMIN        = 0.08      # [N]
FMAX        = 0.60      # [N]
TH_INIT     = 0.1       # [s] default DT
THL         = 0.05      # [s] min DT
THH         = 0.2       # [s] max DT

# Relative-component colours (softer variants)
REL_COLS  = ["#f5a0a0", "#90e8b0", "#a898f8", "#f8e090"]   # rx/ry/rz

AZ_PAIR    = [30, 120]              # two azimuths shown per frame
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
                  "qw", "qx", "qy", "qz", "wx", "wy", "wz",
                  "fz", "mx", "my", "mz", "theta",
                  "tgt_x", "tgt_y", "tgt_z", "tgt_vx", "tgt_vy", "tgt_vz"]

    # all_solves.csv uses abs_ and rel_ prefixes for state, but not for controls
    _abs = {"x": "abs_x", "y": "abs_y", "z": "abs_z",
            "vx": "abs_vx", "vy": "abs_vy", "vz": "abs_vz",
            "qw": "abs_qw", "qx": "abs_qx", "qy": "abs_qy", "qz": "abs_qz",
            "wx": "abs_wx", "wy": "abs_wy", "wz": "abs_wz",
            "fz": "fz", "mx": "mx", "my": "my", "mz": "mz", "theta": "theta",
            "tgt_x": "tgt_x", "tgt_y": "tgt_y", "tgt_z": "tgt_z",
            "tgt_vx": "tgt_vx", "tgt_vy": "tgt_vy", "tgt_vz": "tgt_vz"}
    
    _rel = {"x": "rel_x", "y": "rel_y", "z": "rel_z",
            "vx": "rel_vx", "vy": "rel_vy", "vz": "rel_vz",
            "qw": "rel_qw", "qx": "rel_qx", "qy": "rel_qy", "qz": "rel_qz",
            "wx": "rel_wx", "wy": "rel_wy", "wz": "rel_wz",
            "fz": "fz", "mx": "mx", "my": "my", "mz": "mz", "theta": "theta",
            "tgt_x": "tgt_x", "tgt_y": "tgt_y", "tgt_z": "tgt_z",
            "tgt_vx": "tgt_vx", "tgt_vy": "tgt_vy", "tgt_vz": "tgt_vz"}

    # Build dict: solve_num → sorted DataFrame of nodes
    solve_groups = {int(sn): grp.sort_values("node").reset_index(drop=True)
                    for sn, grp in solves_df.groupby("solve_num")}

    nan_row_abs = {c: np.nan for c in state_cols}
    nan_row_rel = {c: np.nan for c in state_cols}
    node_idx = {}
    cmd_rows_abs = []
    cmd_rows_rel = []

    for _, row in actual_df.iterrows():
        sn = int(row["solve_num"])
        if sn not in solve_groups:
            cmd_rows_abs.append(nan_row_abs.copy())
            cmd_rows_rel.append(nan_row_rel.copy())
            continue
        if sn not in node_idx:
            node_idx[sn] = 0
        grp  = solve_groups[sn]
        nidx = min(node_idx[sn], len(grp) - 1)
        r    = grp.iloc[nidx]
        cmd_rows_abs.append({c: r[_abs[c]] if _abs[c] in r.index else np.nan for c in state_cols})
        cmd_rows_rel.append({c: r[_rel[c]] if _rel[c] in r.index else np.nan for c in state_cols})
        node_idx[sn] += 1

    return pd.DataFrame(cmd_rows_abs), pd.DataFrame(cmd_rows_rel)


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

def page_3d_views(actual_df, solves_df, cmd_df_abs, cmd_df_rel, first_traj, title_str):
    dpos  = actual_df[["x","y","z"]].values
    quats = actual_df[["qw","qx","qy","qz"]].values
    time_actual = actual_df["timestamp"].values
    t_rel = time_actual - time_actual[0]

    # --- Target trajectory ----------------------------------------------------
    # Extract target from the commanded node directly!
    tgt_df = cmd_df_abs[["tgt_x", "tgt_y", "tgt_z"]]
    tgt_df = tgt_df.fillna(method="bfill").fillna(method="ffill")
    tgt_pos = tgt_df.values

    # Relative actual position
    rel_pos = dpos - tgt_pos

    cmd_pos_abs = cmd_df_abs[["x","y","z"]].values
    cmd_pos_rel = cmd_df_rel[["x","y","z"]].values
    
    first_pos_abs = first_traj[["abs_x","abs_y","abs_z"]].values
    first_pos_rel = first_traj[["rel_x","rel_y","rel_z"]].values

    # Speed magnitude for colormap
    vel  = actual_df[["vx","vy","vz"]].values
    vmag = np.linalg.norm(vel, axis=1)

    replan_idx = find_solve_update_indices(actual_df)

    # ── Scene limits ──────────────────────────────────────────────────────────
    XY_LIM_ABS = max(float(np.abs(dpos[:, :2]).max()) * 1.2, float(np.abs(tgt_pos[:, :2]).max()) * 1.2, 1.5)
    Z_LIM_ABS  = max(float(dpos[:, 2].max()) * 1.2, 1.5)
    
    R = float(solves_df.iloc[0].get("circle_radius", 0.0))
    XY_LIM_REL = max(float(np.abs(rel_pos[:, :2]).max()) * 1.2, R * 1.2 if R > 0 else 1.5)
    Z_LIM_REL  = max(float(rel_pos[:, 2].max()) * 1.2, 1.5)
    
    # We want z=0 in the relative frame plot to be the ground ONLY IF the user wants it,
    # but in relative frame z=0 IS the target.
    # The actual ground is at rz = -target_z.
    # We will center the z-axis to show both ground and target.
    avg_tgt_z = float(np.mean(tgt_pos[:, 2]))
    max_rel_z = max(float(rel_pos[:, 2].max()) * 1.2, 1.5)
    min_rel_z = min(float(rel_pos[:, 2].min()) * 1.2, -avg_tgt_z)

    # ── Cone geometry ─────────────────────────────────────────────────────────
    rim_r  = GS_TAN * CONE_H
    _rt    = np.linspace(0, 2*np.pi, 64)
    rim_x  = rim_r * np.cos(_rt);  rim_y = rim_r * np.sin(_rt)
    _ct    = np.linspace(0, 2*np.pi, 12, endpoint=False)

    fig = _fig(f"{title_str}  —  3-D Trajectory (Absolute vs Relative)", figsize=(20, 16))
    
    for row_idx, is_rel in enumerate([False, True]):
        frame_label = "Relative" if is_rel else "Absolute"
        lim_xy = XY_LIM_REL if is_rel else XY_LIM_ABS
        lim_z  = Z_LIM_REL if is_rel else Z_LIM_ABS
        traj   = rel_pos if is_rel else dpos
        ctraj  = cmd_pos_rel if is_rel else cmd_pos_abs
        ftraj  = first_pos_rel if is_rel else first_pos_abs
        
        xlbl, ylbl, zlbl = ("rx [m]", "ry [m]", "rz [m]") if is_rel else ("X [m]", "Y [m]", "Z [m]")

        for col_idx, azim in enumerate(AZ_PAIR):
            subplot_idx = row_idx * 2 + col_idx + 1
            ax = fig.add_subplot(2, 2, subplot_idx, projection="3d")
            _style_3d(ax, f"{frame_label} Frame  |  Azimuth = {azim}°")
            ax.view_init(elev=ELEV_VIEW, azim=azim)
            ax.set_xlim(-lim_xy, lim_xy)
            ax.set_ylim(-lim_xy, lim_xy)
            
            if is_rel:
                ax.set_zlim(min_rel_z, max_rel_z)
            else:
                ax.set_zlim(0, lim_z)
                
            ax.set_box_aspect([2, 2, 1.5])
            ax.set_xlabel(xlbl, labelpad=6)
            ax.set_ylabel(ylbl, labelpad=6)
            ax.set_zlabel(zlbl, labelpad=6)

            # Ground Plane
            _gx = np.array([[-lim_xy, lim_xy], [-lim_xy, lim_xy]])
            _gy = np.array([[-lim_xy, -lim_xy], [lim_xy, lim_xy]])
            ground_z = -avg_tgt_z if is_rel else 0.0
            ax.plot_surface(_gx, _gy, np.full_like(_gx, ground_z),
                            color=GROUND_COL, 
                            alpha=0.3, linewidth=0, antialiased=False)
                            
            if is_rel:
                # Plot the target plane at rz=0
                ax.plot_surface(_gx, _gy, np.zeros_like(_gx),
                                color="#dcdcdc", 
                                alpha=0.15, linewidth=0, antialiased=False)

            if not is_rel:
                # Target circle trajectory (thick magenta) in absolute frame
                ax.plot(tgt_pos[:, 0], tgt_pos[:, 1], tgt_pos[:, 2], color="#d12be6", lw=2.5, ls="-", alpha=0.85, zorder=2, label="Target trajectory")
                # Target current position
                ax.plot([tgt_pos[-1, 0]], [tgt_pos[-1, 1]], [tgt_pos[-1, 2]], "o", color="#d12be6", markersize=8)
            else:
                # Target at origin in relative frame
                ax.plot([0], [0], [0], "o", color="#d12be6", markersize=10, label="Target (origin)")
                # Show circle rim in relative frame (where drone wants to be)
                if R > 1e-6:
                    ax.plot(R*np.cos(_rt), R*np.sin(_rt), np.zeros_like(_rt), color="#d12be6", lw=1.0, ls="--", alpha=0.3)

            # Glideslope cone
            apex = np.array([0, 0, 0]) if is_rel else tgt_pos[-1]
            z_apex = apex[2]
            z_rim = z_apex + CONE_H
            
            # Draw rim
            ax.plot(apex[0] + rim_r * np.cos(_rt), 
                    apex[1] + rim_r * np.sin(_rt), 
                    np.full_like(_rt, z_rim), 
                    color=CONE_COL, lw=1.2, alpha=0.4, ls="-", zorder=3)
            # Draw spokes
            for th in _ct:
                ax.plot([apex[0], apex[0] + rim_r * np.cos(th)],
                        [apex[1], apex[1] + rim_r * np.sin(th)],
                        [z_apex, z_rim],
                        color=CONE_COL, lw=0.8, alpha=0.3, zorder=3)

            # First-solve reference (dotted, faint)
            ax.plot(ftraj[:, 0], ftraj[:, 1], ftraj[:, 2],
                color=C_FIRST, lw=1.4, ls=":", alpha=0.55, zorder=3)

            # Commanded / stitched path (dashed)
            ax.plot(ctraj[:, 0], ctraj[:, 1], ctraj[:, 2],
                color=C_CMD, lw=1.5, ls="--", alpha=0.70, zorder=4)

            # Actual trajectory (solid, coloured by speed)
            from matplotlib.colors import Normalize
            import matplotlib.cm as cm
            v_max   = max(float(vmag.max()), 0.1)
            vnorm   = Normalize(vmin=0, vmax=v_max)
            cmap    = cm.plasma
            for j in range(len(traj) - 1):
                c = cmap(vnorm(vmag[j]))
                ax.plot(traj[j:j+2, 0], traj[j:j+2, 1], traj[j:j+2, 2],
                        color=c, lw=2.2, alpha=0.90, zorder=5)

            # Body-frame axes at 10 sampled points along the trajectory
            body_indices = np.linspace(0, len(traj) - 1, 10, dtype=int)
            for ridx in body_indices:
                if ridx >= len(traj): continue
                pos = traj[ridx]
                R_mat = quat_to_rotmat(*quats[ridx])
                s_val = BODY_AZSCALE
                ax.quiver(*pos, *(R_mat[:, 0]*s_val), color=BODY_X, lw=1.5, arrow_length_ratio=0.35, alpha=0.8)
                ax.quiver(*pos, *(R_mat[:, 1]*s_val), color=BODY_Y, lw=1.5, arrow_length_ratio=0.35, alpha=0.8)
                ax.quiver(*pos, *(R_mat[:, 2]*s_val), color=BODY_Z, lw=1.5, arrow_length_ratio=0.35, alpha=0.8)

            # Start / end markers
            ax.plot(*traj[0], "o", color=C_ACTUAL, markersize=8, zorder=10)
            ax.plot(*traj[-1], "*", color=C_ACTUAL, markersize=10, zorder=10)

            if row_idx == 0 and col_idx == 0:
                handles = [
                    Line2D([0],[0], color=C_ACTUAL, lw=2.2,           label="Actual path"),
                    Line2D([0],[0], color="#d12be6", lw=2.5,         label="Target"),
                    Line2D([0],[0], color=C_CMD,    lw=1.5, ls="--",  label="Commanded"),
                    Line2D([0],[0], color=C_FIRST,  lw=1.4, ls=":",   label="First plan"),
                    Line2D([0],[0], color=CONE_COL, lw=1.2,           label="Glideslope"),
                ]
                _legend(ax, handles, loc="upper right", fontsize=9)

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
#  Page 11 – Control Inputs & Timing
# ═══════════════════════════════════════════════════════════════════════════════

def page_controls(time, actual_ctrl, cmd_ctrl, title_str):
    """
    Subplots for thrust, moments, and node-time (theta).
    """
    fig = _fig(f"{title_str}  —  Control Inputs & Timing", figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 1, hspace=0.4)
    
    ax_fz = fig.add_subplot(gs[0])
    _style_ax(ax_fz, "Time [s]", "Thrust [N]", "Vertical Thrust (fz)")
    ax_fz.plot(time, cmd_ctrl[:, 0], color=COMP_COLS[2], lw=1.8, ls="--", label="cmd fz")
    ax_fz.axhline(FMIN, color="r", lw=1.2, ls=":", label="FMIN")
    ax_fz.axhline(FMAX, color="r", lw=1.2, ls=":", label="FMAX")
    ax_fz.legend(loc="upper right", fontsize=10)

    ax_m = fig.add_subplot(gs[1])
    _style_ax(ax_m, "Time [s]", "Moment [Nm?]", "Control Moments (mx, my, mz)")
    ax_m.plot(time, cmd_ctrl[:, 1], color=COMP_COLS[0], lw=1.8, ls="--", label="mx")
    ax_m.plot(time, cmd_ctrl[:, 2], color=COMP_COLS[1], lw=1.8, ls="--", label="my")
    ax_m.plot(time, cmd_ctrl[:, 3], color=COMP_COLS[2], lw=1.8, ls="--", label="mz")
    ax_m.legend(loc="upper right", fontsize=10)

    ax_th = fig.add_subplot(gs[2])
    _style_ax(ax_th, "Time [s]", "DT [s]", "Node Interval (theta)")
    # Extract theta if available (often stored in a specific column or meta)
    # Since we are reading from all_solves stitched to actual, we need to check columns
    if "theta" in actual_ctrl.columns:
        ax_th.plot(time, actual_ctrl["theta"].values, color="#606060", lw=2.0, label="actual theta")
    ax_th.axhline(THL, color="r", lw=1.2, ls=":", label="THL")
    ax_th.axhline(THH, color="r", lw=1.2, ls=":", label="THH")
    ax_th.axhline(TH_INIT, color="g", lw=1.0, ls="--", label="Initial")
    ax_th.legend(loc="upper right", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def build_plots(data_dir, out_path=None, elev=22):
    global ELEV_VIEW
    ELEV_VIEW = elev

    actual_df, solves_df = load_data(data_dir)
    cmd_df_abs, cmd_df_rel = compute_commanded_states(actual_df, solves_df)
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

    cmd_pos_abs = cmd_df_abs[["x","y","z"]].values
    cmd_vel_abs = cmd_df_abs[["vx","vy","vz"]].values
    cmd_quat_abs= cmd_df_abs[["qw","qx","qy","qz"]].values
    cmd_angv_abs= cmd_df_abs[["wx","wy","wz"]].values

    first_pos_abs = first_traj[["abs_x","abs_y","abs_z"]].values
    first_vel_abs = first_traj[["abs_vx","abs_vy","abs_vz"]].values
    first_quat_abs= first_traj[["abs_qw","abs_qx","abs_qy","abs_qz"]].values
    first_angv_abs= first_traj[["abs_wx","abs_wy","abs_wz"]].values

    print(f"  Building PDF → {out_path}")
    with PdfPages(out_path) as pdf:

        # ── Page 1 : 3-D views ─────────────────────────────────────────────────
        print("  [1/10] 3-D views (Abs/Rel) …")
        fig1 = page_3d_views(actual_df, solves_df, cmd_df_abs, cmd_df_rel, first_traj, title_str)
        pdf.savefig(fig1, facecolor=FIG_BG); plt.close(fig1)

        # ── Page 2 : Position (Absolute) ──────────────────────────────────────
        print("  [2/10] Position states (Absolute) …")
        fig2 = _state_page(
            time, actual_pos, cmd_pos_abs,
            first_t, first_pos_abs,
            labels=["x", "y", "z"], units="m",
            page_title=f"{title_str}  —  Position (Absolute)",
            fig_title="Absolute Position  [x, y, z]")
        pdf.savefig(fig2, facecolor=FIG_BG); plt.close(fig2)

        # ── Page 3 : Velocity (Absolute) ──────────────────────────────────────
        print("  [3/11] Velocity states (Absolute) …")
        fig3 = _state_page(
            time, actual_vel, cmd_vel_abs,
            first_t, first_vel_abs,
            labels=["vx", "vy", "vz"], units="m/s",
            page_title=f"{title_str}  —  Velocity (Absolute)",
            fig_title="Absolute Velocity  [vx, vy, vz]")
        # Reference vz_ref
        for ax in fig3.get_axes():
            if ax.get_title().startswith("Absolute Velocity"):
                ax.axhline(VZ_REF, color="r", lw=1.2, ls="--", alpha=0.6, label="VZ_REF")
                _legend(ax, ax.get_lines(), loc="best")
        pdf.savefig(fig3, facecolor=FIG_BG); plt.close(fig3)

        # ── Page 4 : Quaternion ───────────────────────────────────────────────
        print("  [4/10] Quaternion states …")
        fig4 = _state_page(
            time, actual_quat, cmd_quat_abs,
            first_t, first_quat_abs,
            labels=["qw", "qx", "qy", "qz"], units="–",
            page_title=f"{title_str}  —  Quaternion",
            fig_title="Attitude quaternion  [qw, qx, qy, qz]")
        pdf.savefig(fig4, facecolor=FIG_BG); plt.close(fig4)

        # ── Page 5 : Angular velocity ─────────────────────────────────────────
        print("  [5/10] Angular velocity states …")
        fig5 = _state_page(
            time, actual_angv, cmd_angv_abs,
            first_t, first_angv_abs,
            labels=["wx", "wy", "wz"], units="rad/s",
            page_title=f"{title_str}  —  Angular Velocity",
            fig_title="Angular velocity  [wx, wy, wz]")
        pdf.savefig(fig5, facecolor=FIG_BG); plt.close(fig5)

        # ── Page 6 : Relative Position & Velocity ─────────────────────────────
        # Extract target trajectory from commanded states
        tgt_df = cmd_df_abs[["tgt_x", "tgt_y", "tgt_z", "tgt_vx", "tgt_vy", "tgt_vz"]]
        tgt_df = tgt_df.fillna(method="bfill").fillna(method="ffill")
        
        tgt_x = tgt_df["tgt_x"].values
        tgt_y = tgt_df["tgt_y"].values
        tgt_z = tgt_df["tgt_z"].values
        tgt_vx = tgt_df["tgt_vx"].values
        tgt_vy = tgt_df["tgt_vy"].values
        tgt_vz = tgt_df["tgt_vz"].values
        
        actual_rel_pos = actual_pos - np.stack([tgt_x, tgt_y, tgt_z], axis=1)
        actual_rel_vel = actual_vel - np.stack([tgt_vx, tgt_vy, tgt_vz], axis=1)
        
        fig6 = _fig(f"{title_str}  —  Relative States", figsize=(16, 10))
        gs6 = gridspec.GridSpec(2, 1, hspace=0.3)
        ax6a = fig6.add_subplot(gs6[0]); _style_ax(ax6a, "Time [s]", "Pos [m]", "Relative Position (Drone - Target)")
        ax6b = fig6.add_subplot(gs6[1]); _style_ax(ax6b, "Time [s]", "Vel [m/s]", "Relative Velocity (Drone - Target)")
        
        for k in range(3):
            ax6a.plot(time, actual_rel_pos[:, k], color=COMP_COLS[k], lw=2.0, label=["rx","ry","rz"][k])
            ax6b.plot(time, actual_rel_vel[:, k], color=COMP_COLS[k], lw=2.0, label=["rvx","rvy","rvz"][k])
        ax6b.axhline(VZ_REF, color="r", lw=1.2, ls="--", alpha=0.6, label="VZ_REF")
        ax6a.legend(); ax6b.legend()
        pdf.savefig(fig6, facecolor=FIG_BG); plt.close(fig6)

        # ── Page 7 : Target Tracking ──────────────────────────────────────────
        print("  [7/10] Target tracking …")
        fig7 = _fig(f"{title_str}  —  Target Tracking", figsize=(16, 10))
        # Plot x, y components vs target
        ax7a = fig7.add_subplot(2, 1, 1); _style_ax(ax7a, "Time [s]", "X [m]", "X Tracking")
        ax7a.plot(time, tgt_x, color="#d12be6", lw=1.5, ls="--", label="Target X")
        ax7a.plot(time, actual_pos[:, 0], color=COMP_COLS[0], lw=2.0, label="Drone X")
        ax7a.legend()
        
        ax7b = fig7.add_subplot(2, 1, 2); _style_ax(ax7b, "Time [s]", "Y [m]", "Y Tracking")
        ax7b.plot(time, tgt_y, color="#d12be6", lw=1.5, ls="--", label="Target Y")
        ax7b.plot(time, actual_pos[:, 1], color=COMP_COLS[1], lw=2.0, label="Drone Y")
        ax7b.legend()
        pdf.savefig(fig7, facecolor=FIG_BG); plt.close(fig7)

        # ── Page 8 : Position error ───────────────────────────────────────────
        print("  [8/10] Position error …")
        fig8 = _error_page(
            time, actual_pos, cmd_pos_abs,
            labels=["x", "y", "z"], units="m",
            page_title=f"{title_str}  —  Position Tracking Error",
            fig_title="Position error  (actual − commanded)")
        pdf.savefig(fig8, facecolor=FIG_BG); plt.close(fig8)

        # ── Page 9 : Velocity error ───────────────────────────────────────────
        print("  [9/10] Velocity error …")
        fig9 = _error_page(
            time, actual_vel, cmd_vel_abs,
            labels=["vx", "vy", "vz"], units="m/s",
            page_title=f"{title_str}  —  Velocity Tracking Error",
            fig_title="Velocity error  (actual − commanded)")
        pdf.savefig(fig9, facecolor=FIG_BG); plt.close(fig9)

        # ── Page 10 : Solver Stats ───────────────────────────────────────────
        print("  [10/11] Solver stats …")
        fig10 = page_solver_stats(solves_df, title_str)
        pdf.savefig(fig10, facecolor=FIG_BG); plt.close(fig10)

        # ── Page 11 : Control Inputs ─────────────────────────────────────────
        print("  [11/11] Control inputs …")
        # Extract controls fz, mx, my, mz from cmd_df_abs (they are logged in all_solves)
        cmd_ctrl = cmd_df_abs[["fz", "mx", "my", "mz"]].values
        # For theta, we might need to find it in solves_df if it's there
        fig11 = page_controls(time, cmd_df_abs, cmd_ctrl, title_str)
        pdf.savefig(fig11, facecolor=FIG_BG); plt.close(fig11)

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
    