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

STATE_COLS = [
    "x", "y", "z", "vx", "vy", "vz",
    "qw", "qx", "qy", "qz", "wx", "wy", "wz",
]
CONTROL_COLS = ["fz", "mx", "my", "mz"]
TARGET_COLS = ["tgt_x", "tgt_y", "tgt_z", "tgt_vx", "tgt_vy", "tgt_vz"]
STREAM_COLS = STATE_COLS + CONTROL_COLS + ["theta"] + TARGET_COLS

# ═══════════════════════════════════════════════════════════════════════════════
#  Dynamic schema detection — reads column structure from all_solves.csv headers
# ═══════════════════════════════════════════════════════════════════════════════

_FIXED_PREFIX = {"solve_num", "solve_time_ms", "solve_iters", "coord_mode", "node", "t", "theta"}
_FIXED_SUFFIX = {"tgt_x", "tgt_y", "tgt_z", "tgt_vx", "tgt_vy", "tgt_vz"}
_CONTROL_STARTERS = {"fz", "T", "thrust", "mx", "omx"}


def detect_schema(solves_df):
    """Infer state_cols, control_cols from all_solves.csv column headers.

    Returns:
        state_cols      : list of logical state column names
        control_cols    : list of logical control column names (excluding Theta)
        theta_col       : "theta" if present, else None
        abs_state_cols  : list of abs_* prefixed columns (empty for non-relative OCPs)
        rel_state_cols  : list of rel_* prefixed columns
    """
    middle = [c for c in solves_df.columns
              if c not in _FIXED_PREFIX and c not in _FIXED_SUFFIX]
    theta_col = "theta" if "theta" in solves_df.columns else None

    abs_state_cols = [c for c in middle if c.startswith("abs_")]
    rel_state_cols = [c for c in middle if c.startswith("rel_")]
    plain_cols = [c for c in middle
                  if not c.startswith("abs_") and not c.startswith("rel_")]

    if abs_state_cols:
        # Relative OCP: abs_x/rel_x encoding; controls are in plain_cols
        state_cols = [c[4:] for c in abs_state_cols]  # strip "abs_"
        control_cols = [c for c in plain_cols if c not in ("theta", "Theta")]
    else:
        # Absolute/body-frame OCP: plain state columns then plain control columns
        split = len(plain_cols)
        for i, c in enumerate(plain_cols):
            if c in _CONTROL_STARTERS:
                split = i
                break
        state_cols = plain_cols[:split]
        control_cols = [c for c in plain_cols[split:]
                        if c not in ("theta", "Theta")]

    return state_cols, control_cols, theta_col, abs_state_cols, rel_state_cols


def group_state_cols(state_cols):
    """Group detected state columns by semantic type for page generation.

    Returns list of (group_label, col_list, units) tuples.
    """
    _POS  = {"px","py","pz","x","y","z","p0","p1","p2"}
    _VEL  = {"vx","vy","vz","vdx","vdy","vdz","v0","v1","v2"}
    _QUAT = {"qw","qx","qy","qz"}
    _ANGV = {"wx","wy","wz"}
    _DT   = {"DT"}

    pos  = [c for c in state_cols if c in _POS  or c.startswith("p_") or c[:2] in ("px","py","pz")]
    vel  = [c for c in state_cols if c in _VEL  or c.startswith("v_") or c[:2] in ("vx","vy","vz")]
    quat = [c for c in state_cols if c in _QUAT]
    angv = [c for c in state_cols if c in _ANGV]
    skip = set(pos) | set(vel) | set(quat) | set(angv) | _DT
    aug  = [c for c in state_cols if c not in skip]

    groups = []
    if pos:  groups.append(("Position",       pos,  "m"))
    if vel:  groups.append(("Velocity",        vel,  "m/s"))
    if quat: groups.append(("Quaternion",      quat, "—"))
    if angv: groups.append(("Angular Rate",    angv, "rad/s"))
    if aug:  groups.append(("Augmented State", aug,  "mixed"))
    return groups


def _fc(df, *candidates):
    """Return the first column from candidates that exists in df, else empty series."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)

def _normalise_solves(df):
    """Rename columns from newer logging schema to the names the rest of the script expects."""
    renames = {"solve_id": "solve_num", "solve_ms": "solve_time_ms", "t_acc": "t"}
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    if "solve_iters" not in df.columns:
        df["solve_iters"] = 0
    if "coord_mode" not in df.columns:
        df["coord_mode"] = ""
    return df


def load_data(data_dir):
    actual = pd.read_csv(os.path.join(data_dir, "actual_state.csv"))
    solves = _normalise_solves(pd.read_csv(os.path.join(data_dir, "all_solves.csv")))
    cmd_path = os.path.join(data_dir, "commanded_state.csv")
    commanded = pd.read_csv(cmd_path) if os.path.exists(cmd_path) else None
    return actual, solves, commanded


def compute_solve_aligned_states(actual_df, solves_df,
                                  state_cols=None, control_cols=None,
                                  abs_state_cols=None, rel_state_cols=None,
                                  theta_col=None):
    """
    For each row in actual_df (with a given solve_num), pull the corresponding
    planned node from all_solves.csv.  Node counter advances independently for
    each solve, so solve transitions always restart at node 0.

    Works with any column layout detected by detect_schema().
    Returns (cmd_df_abs, cmd_df_rel) — both have the same set of output columns.
    Output columns are: state_cols + control_cols + ['theta'] + TARGET_COLS
    """
    # Detect schema if not provided
    if state_cols is None:
        state_cols, control_cols, theta_col, abs_state_cols, rel_state_cols = \
            detect_schema(solves_df)

    has_abs_rel = len(abs_state_cols) > 0
    all_out_cols = list(state_cols) + list(control_cols) + \
                   (["theta"] if theta_col else []) + list(TARGET_COLS)
    nan_row = {c: np.nan for c in all_out_cols}

    # Build mapping: logical_name → csv_column_name
    def _make_abs_map():
        m = {}
        for c in abs_state_cols:
            m[c[4:]] = c   # "abs_x" → "x" → "abs_x"
        for c in control_cols:
            m[c] = c
        if theta_col:
            m["theta"] = theta_col
        for c in TARGET_COLS:
            m[c] = c
        return m

    def _make_rel_map():
        m = {}
        for c in rel_state_cols:
            m[c[4:]] = c
        for c in control_cols:
            m[c] = c
        if theta_col:
            m["theta"] = theta_col
        for c in TARGET_COLS:
            m[c] = c
        return m

    def _make_plain_map():
        m = {}
        for c in state_cols:
            m[c] = c
        for c in control_cols:
            m[c] = c
        if theta_col:
            m["theta"] = theta_col
        for c in TARGET_COLS:
            m[c] = c
        return m

    if has_abs_rel:
        map_abs = _make_abs_map()
        map_rel = _make_rel_map()
    else:
        map_abs = _make_plain_map()
        map_rel = _make_plain_map()   # no relative version for plain OCPs

    # Build dict: solve_num → sorted DataFrame of nodes
    solve_groups = {int(sn): grp.sort_values("node").reset_index(drop=True)
                    for sn, grp in solves_df.groupby("solve_num")}

    node_idx = {}
    cmd_rows_abs = []
    cmd_rows_rel = []

    for _, row in actual_df.iterrows():
        sn = int(row["solve_num"])
        if sn not in solve_groups:
            cmd_rows_abs.append(nan_row.copy())
            cmd_rows_rel.append(nan_row.copy())
            continue
        if sn not in node_idx:
            node_idx[sn] = 0
        grp  = solve_groups[sn]
        nidx = min(node_idx[sn], len(grp) - 1)
        r    = grp.iloc[nidx]
        cmd_rows_abs.append(
            {c: r[map_abs[c]] if c in map_abs and map_abs[c] in r.index else np.nan
             for c in all_out_cols})
        cmd_rows_rel.append(
            {c: r[map_rel[c]] if c in map_rel and map_rel[c] in r.index else np.nan
             for c in all_out_cols})
        node_idx[sn] += 1

    return pd.DataFrame(cmd_rows_abs), pd.DataFrame(cmd_rows_rel)


def align_published_command(actual_df, commanded_df):
    if commanded_df is None:
        return None
    req_cols = ["timestamp"] + STATE_COLS + CONTROL_COLS
    if any(col not in commanded_df.columns for col in req_cols):
        return None

    left = actual_df[["timestamp"]].copy().sort_values("timestamp").reset_index()
    right = commanded_df[req_cols].copy().sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="nearest")
    merged = merged.set_index("index").sort_index()
    return merged[STATE_COLS + CONTROL_COLS]


def infer_actual_frame(actual_df, solve_abs_df, solve_rel_df, target_df, aligned_cmd=None):
    if "coord_mode" in actual_df.columns:
        modes = actual_df["coord_mode"].dropna().astype(str).str.lower().unique().tolist()
        modes = [m for m in modes if m in ("absolute", "relative")]
        if len(modes) == 1:
            return modes[0], f"explicit coord_mode={modes[0]}"

    has_abs_cols = all(f"abs_{c}" in actual_df.columns for c in STATE_COLS)
    has_rel_cols = all(f"rel_{c}" in actual_df.columns for c in STATE_COLS)
    if has_abs_cols and has_rel_cols:
        return "absolute", "explicit abs_*/rel_* columns present"

    if aligned_cmd is not None:
        raw_xyz = actual_df[["x", "y", "z"]].to_numpy(dtype=float)
        cmd_xyz = aligned_cmd[["x", "y", "z"]].to_numpy(dtype=float)
        tgt_xyz = target_df[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(dtype=float)
        valid = np.isfinite(raw_xyz).all(axis=1) & np.isfinite(cmd_xyz).all(axis=1) & np.isfinite(tgt_xyz).all(axis=1)
        if np.any(valid):
            mae_abs = float(np.mean(np.linalg.norm(raw_xyz[valid] - cmd_xyz[valid], axis=1)))
            mae_rel = float(np.mean(np.linalg.norm((raw_xyz[valid] + tgt_xyz[valid]) - cmd_xyz[valid], axis=1)))
            inferred = "relative" if mae_rel < 0.8 * mae_abs else "absolute"
            reason = f"inferred from published command proximity (mae_abs={mae_abs:.3f}, mae_rel={mae_rel:.3f})"
            return inferred, reason

    raw_xyz = actual_df[["x", "y", "z"]].to_numpy(dtype=float)
    abs_xyz = solve_abs_df[["x", "y", "z"]].to_numpy(dtype=float)
    rel_xyz = solve_rel_df[["x", "y", "z"]].to_numpy(dtype=float)
    valid = np.isfinite(raw_xyz).all(axis=1) & np.isfinite(abs_xyz).all(axis=1) & np.isfinite(rel_xyz).all(axis=1)
    if not np.any(valid):
        return "absolute", "fallback (insufficient overlap for frame inference)"

    mae_abs = float(np.mean(np.linalg.norm(raw_xyz[valid] - abs_xyz[valid], axis=1)))
    mae_rel = float(np.mean(np.linalg.norm(raw_xyz[valid] - rel_xyz[valid], axis=1)))
    inferred = "relative" if mae_rel < 0.7 * mae_abs else "absolute"
    reason = f"inferred from solve-node proximity (mae_abs={mae_abs:.3f}, mae_rel={mae_rel:.3f})"
    return inferred, reason


def normalize_actual_absolute(actual_df, frame_mode, target_df):
    has_abs_cols = all(f"abs_{c}" in actual_df.columns for c in STATE_COLS)
    if has_abs_cols:
        actual_abs = actual_df[[f"abs_{c}" for c in STATE_COLS]].copy()
        actual_abs.columns = STATE_COLS
        return actual_abs

    actual_abs = actual_df[STATE_COLS].copy()
    if frame_mode == "relative":
        actual_abs[["x", "y", "z"]] = actual_abs[["x", "y", "z"]].to_numpy(dtype=float) + target_df[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(dtype=float)
        actual_abs[["vx", "vy", "vz"]] = actual_abs[["vx", "vy", "vz"]].to_numpy(dtype=float) + target_df[["tgt_vx", "tgt_vy", "tgt_vz"]].to_numpy(dtype=float)
    return actual_abs


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

def page_3d_views(actual_df, solves_df, cmd_df_abs, cmd_df_rel, first_traj, title_str, command_label,
                   pos_cols=None, actual_pos_cols=None):
    if pos_cols is None:
        pos_cols = ["x", "y", "z"]
    pc  = pos_cols[:3]
    apc = actual_pos_cols[:3] if actual_pos_cols else pc

    def _safe_arr(df, cols):
        return np.stack([df[c].values.astype(float) if c in df.columns
                         else np.zeros(len(df)) for c in cols], axis=1)

    dpos  = _safe_arr(actual_df, apc)
    # Quaternion for body-frame arrows (optional — only if columns exist)
    _qcols = ["qw","qx","qy","qz"]
    has_quat = all(c in actual_df.columns for c in _qcols)
    quats = actual_df[_qcols].values if has_quat else None
    time_actual = actual_df["timestamp"].values
    t_rel = time_actual - time_actual[0]  # noqa: F841 (kept for future use)

    # --- Target trajectory ----------------------------------------------------
    tgt_df = cmd_df_abs[["tgt_x", "tgt_y", "tgt_z"]]
    tgt_df = tgt_df.ffill().bfill()
    tgt_pos = tgt_df.values

    # Relative actual position
    rel_pos = dpos - tgt_pos

    cmd_pos_abs = _safe_arr(cmd_df_abs, pc)
    cmd_pos_rel = _safe_arr(cmd_df_rel, pc)

    # First trajectory: prefer abs_* prefix, then plain
    abs_pc = [f"abs_{c}" for c in pc]
    rel_pc = [f"rel_{c}" for c in pc]
    first_pos_abs = (_safe_arr(first_traj, abs_pc)
                     if all(c in first_traj.columns for c in abs_pc)
                     else _safe_arr(first_traj, pc))
    first_pos_rel = (_safe_arr(first_traj, rel_pc)
                     if all(c in first_traj.columns for c in rel_pc)
                     else first_pos_abs)

    # Speed magnitude for colormap
    _vcols = ["vx","vy","vz"]
    vel  = _safe_arr(actual_df, _vcols)
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
            # ax.plot(ctraj[:, 0], ctraj[:, 1], ctraj[:, 2],
            #     color=C_CMD, lw=1.5, ls="--", alpha=0.70, zorder=4)

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

            # Body-frame axes at 10 sampled points (only when quaternion data present)
            if has_quat:
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
                    # Line2D([0],[0], color=C_CMD,    lw=1.5, ls="--",  label=command_label),
                    Line2D([0],[0], color=C_FIRST,  lw=1.4, ls=":",   label="Planned solve nodes"),
                    Line2D([0],[0], color=CONE_COL, lw=1.2,           label="Glideslope"),
                ]
                _legend(ax, handles, loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Pages 2-5 – State plots
# ═══════════════════════════════════════════════════════════════════════════════

def _state_page(time, actual_vals, cmd_vals, first_time, first_vals,
                labels, units, page_title, fig_title,
                command_label="published command", first_label="planned solve nodes"):
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
             Line2D([0],[0], color=col, lw=1.8, ls="--", label=f"{lbl}  {command_label}"),
            Line2D([0],[0], color=col, lw=1.5, ls=":",
                 alpha=0.55,                           label=f"{lbl}  {first_label}"),
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


def page_controls_dynamic(time, cmd_ctrl_df, control_cols, theta_col, title_str):
    """Dynamic control page — one subplot per control column, works with any OCP."""
    has_theta = theta_col is not None and theta_col in cmd_ctrl_df.columns
    n_plots = len(control_cols) + (1 if has_theta else 0)
    if n_plots == 0:
        return None

    fig = _fig(f"{title_str}  —  Control Inputs", figsize=(16, max(6, 3 * n_plots)))
    gs  = gridspec.GridSpec(n_plots, 1, hspace=0.4)

    for i, col in enumerate(control_cols):
        ax = fig.add_subplot(gs[i])
        _col = col
        if col in ("fz", "thrust"):
            unit = "[N]"
        elif col == "T":
            unit = "[m/s²]"
        elif col.startswith("om"):
            unit = "[rad/s]"
        else:
            unit = ""
        _style_ax(ax, "Time [s]", f"{_col} {unit}", _col)
        if col in cmd_ctrl_df.columns:
            ax.plot(time, cmd_ctrl_df[col].values, color=COMP_COLS[i % 4],
                    lw=1.8, ls="--", label=f"cmd {col}")
            if col in ("fz", "thrust"):
                ax.axhline(FMIN, color="r", lw=1.2, ls=":", label="FMIN")
                ax.axhline(FMAX, color="r", lw=1.2, ls=":", label="FMAX")
        ax.legend(loc="upper right", fontsize=10)

    if has_theta:
        ax = fig.add_subplot(gs[len(control_cols)])
        _style_ax(ax, "Time [s]", "DT [s]", "Node Interval (theta)")
        ax.plot(time, cmd_ctrl_df[theta_col].values, color="#606060", lw=2.0, label="theta")
        ax.axhline(THL, color="r", lw=1.2, ls=":", label="THL")
        ax.axhline(THH, color="r", lw=1.2, ls=":", label="THH")
        ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def build_plots(data_dir, out_path=None, elev=22):
    global ELEV_VIEW
    ELEV_VIEW = elev

    actual_raw_df, solves_df, commanded_df = load_data(data_dir)

    # ── Dynamic schema from all_solves.csv headers ────────────────────────────
    state_cols, control_cols, theta_col, abs_state_cols, rel_state_cols = \
        detect_schema(solves_df)
    state_groups = group_state_cols(state_cols)

    solve_df_abs, solve_df_rel = compute_solve_aligned_states(
        actual_raw_df, solves_df,
        state_cols, control_cols, abs_state_cols, rel_state_cols, theta_col)

    # For absolute_shifted OCPs (e.g. stateswitch with valid target), the solver
    # trajectory is in (drone − target) frame. Add target back to get world frame.
    if "coord_mode" in solves_df.columns and (solves_df["coord_mode"] == "absolute_shifted").any():
        for _sc, _tc in [("px","tgt_x"),("py","tgt_y"),("pz","tgt_z"),
                          ("vx","tgt_vx"),("vy","tgt_vy"),("vz","tgt_vz")]:
            if _sc in solve_df_abs.columns and _tc in solve_df_abs.columns:
                solve_df_abs[_sc] = solve_df_abs[_sc].values + solve_df_abs[_tc].values

    first_traj = get_first_solve_traj(solves_df)
    target_df  = solve_df_abs[TARGET_COLS].copy().ffill().bfill()

    # Helper: safely extract an (N, k) float array from a DataFrame
    def _get_arr(df, cols):
        return np.stack([df[c].values.astype(float) if c in df.columns
                         else np.full(len(df), np.nan) for c in cols], axis=1)

    # Helper: extract first-trajectory array, preferring abs_* prefix columns
    def _get_first_arr(cols):
        abs_cols = [f"abs_{c}" for c in cols]
        if all(c in first_traj.columns for c in abs_cols):
            return _get_arr(first_traj, abs_cols)
        return _get_arr(first_traj, cols)

    # ── Actual-state normalisation (13D absolute OCPs only) ───────────────────
    _standard_pos = ("x", "y", "z")
    has_world_pos = all(c in actual_raw_df.columns for c in _standard_pos)

    if has_world_pos:
        aligned_cmd = align_published_command(actual_raw_df, commanded_df)
        frame_mode, frame_reason = infer_actual_frame(
            actual_raw_df, solve_df_abs, solve_df_rel, target_df, aligned_cmd)
        actual_abs_state = normalize_actual_absolute(actual_raw_df, frame_mode, target_df)
        actual_df = actual_raw_df.copy()
        for col in actual_abs_state.columns:
            if col in actual_df.columns:
                actual_df[col] = actual_abs_state[col].values
        if frame_mode == "relative":
            print("  Applied relative→absolute conversion using target from all_solves.csv")
    else:
        aligned_cmd = None
        frame_mode  = "body_frame"
        frame_reason = "no world-frame position columns in actual_state.csv"
        actual_df   = actual_raw_df.copy()

    # Inject px/py/pz aliases so 13D OCP dynamic-state pages can match solve columns
    for _alias, _canon in {"px": "x", "py": "y", "pz": "z"}.items():
        if _alias not in actual_df.columns and _canon in actual_df.columns:
            actual_df[_alias] = actual_df[_canon].values

    # ── Commanded state: align published or fall back to solve trajectory ─────
    if aligned_cmd is not None:
        common = [c for c in STATE_COLS + CONTROL_COLS
                  if c in aligned_cmd.columns and c in solve_df_abs.columns]
        if common:
            solve_df_abs.loc[:, common] = aligned_cmd[common].to_numpy()
        command_label = "published command"
    else:
        command_label = "reconstructed command"
        if commanded_df is not None:
            print("  Warning: commanded_state.csv columns incompatible; using solve trajectory.")

    print(f"  Schema: state={state_cols}, controls={control_cols}"
          f"{', theta' if theta_col else ''}")
    print(f"  State groups: {[g[0] for g in state_groups]}")
    print(f"  Frame mode:   {frame_mode} ({frame_reason})")

    title_str = os.path.basename(os.path.normpath(data_dir))
    if out_path is None:
        out_path = os.path.join(data_dir, f"{title_str}_analysis.pdf")

    # ── Time axis ─────────────────────────────────────────────────────────────
    time = actual_df["timestamp"].values.astype(float)
    time = time - time[0]

    first_t = first_traj["t"].values.astype(float)
    if len(first_t) and first_t[0] != 0:
        first_t = first_t - first_t[0]

    # ── Detect whether world-frame 3D page is possible ────────────────────────
    pos_group  = next((g for g in state_groups if g[0] == "Position"), None)
    vel_group  = next((g for g in state_groups if g[0] == "Velocity"), None)
    pos_cols_3 = pos_group[1][:3] if pos_group else []
    # actual_state.csv may use "x","y","z" while all_solves.csv uses "px","py","pz"
    _ACTUAL_XYZ = ["x","y","z"]
    actual_pos_cols_3 = (pos_cols_3 if all(c in actual_df.columns for c in pos_cols_3)
                         else [c for c in _ACTUAL_XYZ if c in actual_df.columns][:3])
    has_3d = (bool(pos_cols_3)
              and all(c in solve_df_abs.columns for c in pos_cols_3)
              and len(actual_pos_cols_3) == 3)

    # Target columns for relative / tracking pages
    tgt_x  = _fc(solve_df_abs, "tgt_x").values.astype(float)
    tgt_y  = _fc(solve_df_abs, "tgt_y").values.astype(float)
    tgt_z  = _fc(solve_df_abs, "tgt_z").values.astype(float)
    tgt_vx = _fc(solve_df_abs, "tgt_vx").values.astype(float)
    tgt_vy = _fc(solve_df_abs, "tgt_vy").values.astype(float)
    tgt_vz = _fc(solve_df_abs, "tgt_vz").values.astype(float)
    has_target = np.isfinite(tgt_x).any()

    page_num = 1
    print(f"  Building PDF → {out_path}")
    with PdfPages(out_path) as pdf:

        # ── 3-D trajectory page ───────────────────────────────────────────────
        if has_3d:
            print(f"  [{page_num}] 3-D views …")
            fig_3d = page_3d_views(actual_df, solves_df, solve_df_abs, solve_df_rel,
                                   first_traj, title_str, command_label,
                                   pos_cols=pos_cols_3,
                                   actual_pos_cols=actual_pos_cols_3)
            pdf.savefig(fig_3d, facecolor=FIG_BG); plt.close(fig_3d)
            page_num += 1

        # ── Dynamic state pages (one per semantic group) ──────────────────────
        for grp_label, grp_cols, grp_units in state_groups:
            # Use columns that exist in actual_df (some OCPs log a subset)
            avail = [c for c in grp_cols if c in actual_df.columns]
            if not avail:
                continue
            print(f"  [{page_num}] {grp_label} ({', '.join(avail)}) …")
            actual_arr = _get_arr(actual_df,   avail)
            cmd_arr    = _get_arr(solve_df_abs, avail)
            first_arr  = _get_first_arr(avail)
            fig_s = _state_page(
                time, actual_arr, cmd_arr,
                first_t, first_arr,
                labels=avail, units=grp_units,
                page_title=f"{title_str}  —  {grp_label}",
                fig_title=f"{grp_label}  [{', '.join(avail)}]",
                command_label=command_label)
            pdf.savefig(fig_s, facecolor=FIG_BG); plt.close(fig_s)
            page_num += 1

        # ── Relative position & velocity page ─────────────────────────────────
        if has_target and has_3d and pos_group and vel_group:
            vel_cols_3 = vel_group[1][:3]
            actual_vel_cols_3 = (vel_cols_3 if all(c in actual_df.columns for c in vel_cols_3)
                                 else [c for c in ["vx","vy","vz"] if c in actual_df.columns][:3])
            if len(actual_vel_cols_3) == 3:
                print(f"  [{page_num}] Relative states …")
                actual_pos = _get_arr(actual_df, actual_pos_cols_3)
                actual_vel = _get_arr(actual_df, actual_vel_cols_3)
                rel_pos = actual_pos - np.stack([tgt_x, tgt_y, tgt_z], axis=1)
                rel_vel = actual_vel - np.stack([tgt_vx, tgt_vy, tgt_vz], axis=1)

                fig_rel = _fig(f"{title_str}  —  Relative States", figsize=(16, 10))
                gs_r = gridspec.GridSpec(2, 1, hspace=0.3)
                ax_rp = fig_rel.add_subplot(gs_r[0])
                ax_rv = fig_rel.add_subplot(gs_r[1])
                _style_ax(ax_rp, "Time [s]", "Pos [m]",   "Relative Position (Drone − Target)")
                _style_ax(ax_rv, "Time [s]", "Vel [m/s]", "Relative Velocity (Drone − Target)")
                for k in range(3):
                    ax_rp.plot(time, rel_pos[:, k], color=COMP_COLS[k], lw=2.0,
                               label=f"r{pos_cols_3[k]}")
                    ax_rv.plot(time, rel_vel[:, k], color=COMP_COLS[k], lw=2.0,
                               label=f"r{vel_cols_3[k]}")
                ax_rp.legend(); ax_rv.legend()
                pdf.savefig(fig_rel, facecolor=FIG_BG); plt.close(fig_rel)
                page_num += 1

        # ── Target tracking page ──────────────────────────────────────────────
        if has_target and has_3d and pos_group:
            avail_pos = [c for c in pos_cols_3 if c in actual_df.columns]
            if len(avail_pos) >= 2:
                print(f"  [{page_num}] Target tracking …")
                actual_pos = _get_arr(actual_df, avail_pos[:2])
                fig_trk = _fig(f"{title_str}  —  Target Tracking", figsize=(16, 10))
                for ki, (lbl, tgt) in enumerate(zip(avail_pos[:2], [tgt_x, tgt_y])):
                    ax = fig_trk.add_subplot(2, 1, ki + 1)
                    _style_ax(ax, "Time [s]", f"{lbl} [m]", f"{lbl} Tracking")
                    ax.plot(time, tgt,              color="#d12be6",    lw=1.5, ls="--", label=f"Target {lbl}")
                    ax.plot(time, actual_pos[:, ki], color=COMP_COLS[ki], lw=2.0, label=f"Drone {lbl}")
                    ax.legend()
                pdf.savefig(fig_trk, facecolor=FIG_BG); plt.close(fig_trk)
                page_num += 1

        # ── Position error page ───────────────────────────────────────────────
        if has_3d and pos_group:
            avail_pos = [c for c in pos_cols_3 if c in actual_df.columns and c in solve_df_abs.columns]
            if avail_pos:
                print(f"  [{page_num}] Position error …")
                fig_ep = _error_page(
                    time,
                    _get_arr(actual_df,   avail_pos),
                    _get_arr(solve_df_abs, avail_pos),
                    labels=avail_pos, units="m",
                    page_title=f"{title_str}  —  Position Tracking Error",
                    fig_title=f"Position error  (actual − {command_label})")
                pdf.savefig(fig_ep, facecolor=FIG_BG); plt.close(fig_ep)
                page_num += 1

        # ── Velocity error page ───────────────────────────────────────────────
        if vel_group:
            vel_cols_3 = vel_group[1][:3]
            avail_vel = [c for c in vel_cols_3
                         if c in actual_df.columns and c in solve_df_abs.columns]
            if avail_vel:
                print(f"  [{page_num}] Velocity error …")
                fig_ev = _error_page(
                    time,
                    _get_arr(actual_df,   avail_vel),
                    _get_arr(solve_df_abs, avail_vel),
                    labels=avail_vel, units="m/s",
                    page_title=f"{title_str}  —  Velocity Tracking Error",
                    fig_title=f"Velocity error  (actual − {command_label})")
                pdf.savefig(fig_ev, facecolor=FIG_BG); plt.close(fig_ev)
                page_num += 1

        # ── Solver stats ──────────────────────────────────────────────────────
        print(f"  [{page_num}] Solver stats …")
        fig_sv = page_solver_stats(solves_df, title_str)
        pdf.savefig(fig_sv, facecolor=FIG_BG); plt.close(fig_sv)
        page_num += 1

        # ── Controls page ─────────────────────────────────────────────────────
        print(f"  [{page_num}] Control inputs …")
        fig_ctrl = page_controls_dynamic(time, solve_df_abs, control_cols, theta_col, title_str)
        if fig_ctrl is not None:
            pdf.savefig(fig_ctrl, facecolor=FIG_BG); plt.close(fig_ctrl)
            page_num += 1

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = f"CoManDO analysis: {title_str}"
        d["Subject"] = "MPC trajectory analysis"

    print(f"  ✓  Saved {out_path}  ({page_num - 1} pages)")
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

    actual_df, solves_df, _ = load_data(args.dir)
    print(f"  Actual states  : {len(actual_df)} rows")
    print(f"  Unique solves  : {solves_df['solve_num'].nunique()}")

    build_plots(args.dir, out_path=args.out, elev=args.elev)