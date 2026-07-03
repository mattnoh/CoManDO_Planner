#!/usr/bin/env python3
"""GIF animation for the quad single-horizon STC landing plots.

This is intentionally separate from ``plot_stc_paper.py``.  It borrows the
rocket landing script's frame-to-GIF workflow while reusing the quad paper-plot
colors, glyphs, cones, and panel conventions.  It writes GIFs only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

HERE = Path(__file__).resolve().parent
# animate.py and plot_stc_paper.py live alongside this script in analysis/.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_stc_paper as paper  # noqa: E402


def _first_active_idx(y: np.ndarray, eps: float = 1e-9) -> int | None:
    idx = np.flatnonzero(np.asarray(y, dtype=float) > eps)
    return int(idx[0]) if len(idx) else None


def _last_active_idx(y: np.ndarray, eps: float = 1e-9) -> int | None:
    idx = np.flatnonzero(np.asarray(y, dtype=float) > eps)
    return int(idx[-1]) if len(idx) else None


def load_animation_data(result_dir: Path, csv_name: str, kind: str) -> dict:
    detected_kind, csv_path = paper.detect_dataset(result_dir, csv_name, kind)
    df, data = paper.read_and_normalize_dataset(result_dir, csv_path, detected_kind)
    bounds = paper.load_current_bounds(detected_kind, result_dir)

    t = data["t"]
    rxyz = data["rxyz"]
    rvxyz = data["rvxyz"]
    quat = data["quat"]
    omega = data["omega"]
    speed = (
        df["rel_speed"].to_numpy(dtype=float)
        if "rel_speed" in df.columns
        else np.linalg.norm(rvxyz, axis=1)
    )
    omega_n = (
        df["omega_norm"].to_numpy(dtype=float)
        if "omega_norm" in df.columns
        else np.linalg.norm(omega, axis=1)
    )
    rxy = np.linalg.norm(rxyz[:, :2], axis=1)
    gs_angle = np.degrees(
        np.arctan2(rxy, np.maximum(rxyz[:, 2] + bounds["gs_apex_offset"], 1e-9))
    )
    q_perp = (
        df["tilt_sin"].to_numpy(dtype=float)
        if "tilt_sin" in df.columns
        else np.linalg.norm(quat[:, 1:3], axis=1)
    )
    tilt_deg = 2.0 * np.degrees(np.arcsin(np.clip(q_perp, 0.0, 1.0)))
    a_land = (
        df["ctcs_a_altitude_state"].to_numpy(dtype=float)
        if "ctcs_a_altitude_state" in df.columns
        else df.get("trig_env", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    )
    a_stage = (
        df["ctcs_a_stage"].to_numpy(dtype=float)
        if "ctcs_a_stage" in df.columns
        else np.maximum(0.0, (rxy - bounds["stage_radius"]) / bounds["stage_trigger_scale"])
    )
    a_los_margin = (bounds["los_alt_trig"] - rxyz[:, 2]) / bounds["los_trigger_scale"]
    a_los_geom = np.where(
        a_los_margin > 0.0,
        bounds.get("los_trigger_floor", 0.0) + a_los_margin,
        0.0,
    )
    if "ctcs_a_los" in df.columns:
        a_los_csv = df["ctcs_a_los"].to_numpy(dtype=float)
        a_los = a_los_csv if np.nanmax(a_los_csv) > 1e-12 else a_los_geom
    else:
        a_los = a_los_geom

    land_t = paper.first_trigger_time(t, a_land)
    los_t = paper.first_trigger_time(t, a_los)
    stage_t = paper.stage_constraint_end_time(t, a_stage)
    land_idx = _first_active_idx(a_land)
    los_idx = _first_active_idx(a_los)
    stage_idx = _last_active_idx(a_stage)
    return {
        "kind": detected_kind,
        "csv_path": csv_path,
        "df": df,
        "bounds": bounds,
        "t": t,
        "rxyz": rxyz,
        "rvxyz": rvxyz,
        "quat": quat,
        "speed": speed,
        "omega_n": omega_n,
        "tilt_deg": tilt_deg,
        "gs_angle": gs_angle,
        "los_angle": paper.compute_los_angle_deg(rxyz, quat),
        "a_land": a_land,
        "a_los": a_los,
        "a_stage": a_stage,
        "land_t": land_t,
        "los_t": los_t,
        "stage_t": stage_t,
        "land_idx": land_idx,
        "los_idx": los_idx,
        "stage_idx": stage_idx,
    }


def frame_indices(n: int, step: int) -> list[int]:
    idx = list(range(0, n, max(1, step)))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def camera_axis(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(q, dtype=float)
    bz_x = 2.0 * (qx * qz + qw * qy)
    bz_y = 2.0 * (qy * qz - qw * qx)
    bz_z = qw * qw - qx * qx - qy * qy + qz * qz
    cam = -np.array([bz_x, bz_y, bz_z], dtype=float)
    n = float(np.linalg.norm(cam))
    return cam / max(n, 1e-12)


def draw_projection_trigger(ax, lo: np.ndarray, hi: np.ndarray,
                            rxyz: np.ndarray, idx: int | None,
                            color: str, dash_offset: float = 0.0):
    if idx is None:
        return
    xtr, ytr, ztr = map(float, rxyz[idx])
    seg_x = 0.10 * float(hi[0] - lo[0])
    seg_y = 0.10 * float(hi[1] - lo[1])
    style = (dash_offset, (5, 3))
    ax.plot([xtr - seg_x, xtr + seg_x], [hi[1], hi[1]], [ztr, ztr],
            color=color, linestyle=style, lw=1.2, zorder=1)
    ax.plot([lo[0], lo[0]], [ytr - seg_y, ytr + seg_y], [ztr, ztr],
            color=color, linestyle=style, lw=1.2, zorder=1)


def draw_active_los_cone(ax, data: dict, k: int, extent: float):
    if data["a_los"][k] <= 1e-9:
        return
    b = data["bounds"]
    center = data["rxyz"][k]
    direction = camera_axis(data["quat"][k])
    # The cone starts on the vehicle and follows the camera boresight, as in
    # the rocket reference animation. Limit the height so it remains readable.
    if abs(direction[2]) > 1e-9:
        height = abs(float(center[2]) / float(direction[2]))
    else:
        height = float(np.linalg.norm(center))
    height = float(np.clip(height, 0.18, min(max(0.28, 0.12 * extent), 0.55)))
    if np.linalg.norm(np.cross([0.0, 0.0, 1.0], direction)) < 1e-9 and direction[2] < 0.0:
        direction = np.array([1e-6, 0.0, -1.0])
    paper.plot_cone_wireframe(
        ax, center=center, direction=direction,
        angle_deg=float(b["los_cone_deg"]), height=height,
        cc=paper.C_TRIG_THR, lw=0.42, zorder=8, num_points=34,
    )


def static_3d_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=paper.C_PROJ, lw=1.0, alpha=0.7,
               label="Trajectory Projections"),
        Line2D([0], [0], color=paper.C_CONE, lw=1.3, linestyle="dotted",
               label="Glideslope cone"),
        Line2D([0], [0], color=paper.C_TRIG_THR, lw=1.2, linestyle="dotted",
               label="LOS cone"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=5,
               markerfacecolor=paper.anim.TARGET, markeredgecolor=paper.anim.TARGET,
               label="Landing point"),
        Line2D([0], [0], color=paper.C_TRIG_STAGE, lw=1.25,
               linestyle=(0, (5, 3)), label="$a_{\\mathrm{stage}}$"),
        Line2D([0], [0], color=paper.C_TRIG_THR, lw=1.25,
               linestyle=(1.4, (5, 3)), label="$a_{\\mathrm{los}}$"),
        Line2D([0], [0], color=paper.C_TRIG_ALT, lw=1.25,
               linestyle=(2.8, (5, 3)), label="$a_{\\mathrm{land}}$"),
    ]


def draw_partial_3d(ax, data: dict, k: int):
    b = data["bounds"]
    rxyz = data["rxyz"]
    quat = data["quat"]
    speed = data["speed"]
    kk = slice(0, k + 1)

    paper.anim.style_3d(ax, "")
    ax.tick_params(labelsize=12)
    lo, hi = paper.set_3d_limits(ax, rxyz, np.zeros(3))
    extent = float(np.max(hi - lo))

    pts = rxyz[kk].reshape(-1, 1, 3)
    if len(pts) > 1:
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(
            segs,
            cmap=plt.cm.rainbow,
            norm=mcolors.Normalize(0.0, max(float(np.nanmax(speed)), 1e-6)),
            array=speed[:k],
            lw=2.8,
            zorder=5,
        )
        ax.add_collection3d(lc)
    else:
        lc = None

    n = k + 1
    ax.plot(rxyz[kk, 0], np.full(n, hi[1]), rxyz[kk, 2],
            color=paper.C_PROJ, lw=0.9, alpha=0.55, zorder=0,
            label="Trajectory Projections")
    ax.plot(np.full(n, lo[0]), rxyz[kk, 1], rxyz[kk, 2],
            color=paper.C_PROJ, lw=0.9, alpha=0.55, zorder=0)

    draw_projection_trigger(ax, lo, hi, rxyz, data["stage_idx"], paper.C_TRIG_STAGE, 0.0)
    draw_projection_trigger(ax, lo, hi, rxyz, data["los_idx"], paper.C_TRIG_THR, 1.4)
    draw_projection_trigger(ax, lo, hi, rxyz, data["land_idx"], paper.C_TRIG_ALT, 2.8)

    target = np.zeros(3)
    apex_z = -float(b["gs_apex_offset"])
    tan_a = float(b.get("gs_post_tan", np.tan(np.radians(float(b["gs_post_deg"])))))
    cone_h = min(max(float(b["alt_trig"]) + b["gs_apex_offset"], 0.85), 1.15)
    z_top = apex_z + cone_h
    radius = tan_a * cone_h
    ax.plot([-radius, 0.0, radius], [hi[1], hi[1], hi[1]], [z_top, apex_z, z_top],
            color=paper.C_CONE, ls="dotted", lw=1.0, zorder=0)
    ax.plot([lo[0], lo[0], lo[0]], [-radius, 0.0, radius], [z_top, apex_z, z_top],
            color=paper.C_CONE, ls="dotted", lw=1.0, zorder=0)

    old_h, old_cone = paper.anim.CONE_H, paper.anim.CONE
    try:
        paper.anim.GS_APEX_OFFSET = float(b["gs_apex_offset"])
        paper.anim.GS_TAN = tan_a
        paper.anim.CONE_H = cone_h
        paper.anim.CONE = paper.C_CONE
        paper.anim.draw_constraint(ax, "gs", target, alpha=0.42)
    finally:
        paper.anim.CONE_H = old_h
        paper.anim.CONE = old_cone

    draw_active_los_cone(ax, data, k, extent)
    paper.anim.draw_target_pad(ax, target, yaw=0.0, alpha=0.58)
    pos = rxyz[k].copy()
    if k == len(rxyz) - 1:
        pos[2] = max(float(pos[2]), 0.04)
    paper.anim.draw_drone(ax, pos, quat[k], scale=0.30, alpha=0.65)

    ax.scatter([0.0], [hi[1]], [0.0], s=24, c=paper.anim.TARGET,
               depthshade=False, zorder=2, label="Landing point")
    ax.scatter([lo[0]], [0.0], [0.0], s=24, c=paper.anim.TARGET,
               depthshade=False, zorder=2)
    ax.set_xlabel("$r_x$ [m]", fontsize=13, labelpad=7)
    ax.set_ylabel("$r_y$ [m]", fontsize=13, labelpad=7)
    ax.set_zlabel("$r_z$ [m]", fontsize=13, labelpad=12)
    try:
        ax.set_box_aspect((1, 1, 0.85))
    except Exception:
        pass
    ax.view_init(elev=13, azim=-58)
    return lc


def y_limits(series: np.ndarray, *bounds: float | None, floor_zero: bool = True) -> tuple[float, float]:
    vals = [np.asarray(series, dtype=float)]
    vals.extend(np.array([b], dtype=float) for b in bounds if b is not None)
    finite = np.concatenate(vals)
    finite = finite[np.isfinite(finite)]
    lo = float(np.nanmin(finite)) if len(finite) else 0.0
    hi = float(np.nanmax(finite)) if len(finite) else 1.0
    if floor_zero:
        lo = min(0.0, lo)
    pad = max(0.08 * (hi - lo), 0.05)
    return lo - pad, hi + pad


def trigger_line(ax, t_now: float, t_trigger: float | None, color: str, label: str | None = None):
    if t_trigger is not None and np.isfinite(t_trigger):
        ax.axvline(t_trigger, c=color, linestyle=(0, (5, 3)), lw=1.25, label=label)


def draw_panel(ax, data: dict, k: int, y: np.ndarray, ylabel: str,
               switch_t: float | None = None,
               switch_color: str = paper.C_TRIG_ALT,
               switch_label: str | None = None,
               upper_pre: float | None = None,
               upper_post: float | None = None,
               lower_pre: float | None = None,
               lower_post: float | None = None):
    t = data["t"]
    kk = slice(0, k + 1)
    ax.plot(t[kk], y[kk], c=paper.C_STATE, lw=1.5, label="State")

    t0, tf = float(t[0]), float(t[-1])
    t_sw = float(switch_t) if switch_t is not None and np.isfinite(switch_t) else tf
    if upper_pre is not None:
        ax.plot([t0, min(t_sw, tf)], [upper_pre, upper_pre],
                c=paper.C_UPPER, linestyle=(0, (5, 3)), lw=1.25, label="Upper bound")
    if lower_pre is not None:
        ax.plot([t0, min(t_sw, tf)], [lower_pre, lower_pre],
                c=paper.C_LOWER, linestyle=(0, (5, 3)), lw=1.25, label="Lower bound")
    if switch_t is not None:
        trigger_line(ax, float(t[k]), switch_t, switch_color, switch_label)
        if upper_post is not None:
            ax.plot([t_sw, tf], [upper_post, upper_post],
                    c=paper.C_UPPER, linestyle=(0, (5, 3)), lw=1.25)
        if lower_post is not None:
            ax.plot([t_sw, tf], [lower_post, lower_post],
                    c=paper.C_LOWER, linestyle=(0, (5, 3)), lw=1.25)

    ax.set_xlim(t0, tf)
    ax.set_ylim(*y_limits(y, upper_pre, upper_post, lower_pre, lower_post))
    ax.set_ylabel(ylabel, fontsize=11, labelpad=1)
    ax.tick_params(labelsize=9)
    ax.grid(True, color="0.88", lw=0.45, ls="--", alpha=0.55)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))


def draw_state_panels(fig, data: dict, k: int):
    b = data["bounds"]
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.96, bottom=0.07, top=0.39,
                          wspace=0.48, hspace=0.48)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    draw_panel(axes[0, 0], data, k, data["speed"],
               "Speed [m/s]", data["land_t"], paper.C_TRIG_ALT,
               "$a_{\\mathrm{land}}$", b["v_pre"], b["v_post"])
    draw_panel(axes[0, 1], data, k, data["los_angle"],
               "LOS angle [deg]", data["los_t"], paper.C_TRIG_THR,
               "$a_{\\mathrm{los}}$", None, b["los_cone_deg"])
    draw_panel(axes[0, 2], data, k, data["gs_angle"],
               "Glideslope [deg]", data["land_t"], paper.C_TRIG_ALT,
               None, None, b["gs_post_deg"])
    draw_panel(axes[1, 0], data, k, data["omega_n"],
               "Angular rate [rad/s]", data["land_t"], paper.C_TRIG_ALT,
               None, b["omega_pre"], b["omega_post"])
    draw_panel(axes[1, 1], data, k, data["tilt_deg"],
               "Tilt [deg]", data["land_t"], paper.C_TRIG_ALT,
               None, b["tilt_pre_deg"], b["tilt_post_deg"])
    draw_panel(axes[1, 2], data, k, data["rxyz"][:, 2],
               "Stage altitude [m]", data["stage_t"], paper.C_TRIG_STAGE,
               "$a_{\\mathrm{stage}}$", None, None, b["z_stage"], None)
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]", fontsize=11)

    handles = [
        Line2D([0], [0], color=paper.C_STATE, lw=1.5, label="State"),
        Line2D([0], [0], color=paper.C_UPPER, lw=1.25, linestyle=(0, (5, 3)), label="Upper bound"),
        Line2D([0], [0], color=paper.C_LOWER, lw=1.25, linestyle=(0, (5, 3)), label="Lower bound"),
        Line2D([0], [0], color=paper.C_TRIG_THR, lw=1.25, linestyle=(0, (5, 3)), label="$a_{\\mathrm{los}}$"),
        Line2D([0], [0], color=paper.C_TRIG_ALT, lw=1.25, linestyle=(0, (5, 3)), label="$a_{\\mathrm{land}}$"),
        Line2D([0], [0], color=paper.C_TRIG_STAGE, lw=1.25, linestyle=(0, (5, 3)), label="$a_{\\mathrm{stage}}$"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=6, fontsize=10,
               bbox_to_anchor=(0.08, 0.405, 0.88, 0.04), mode="expand",
               frameon=True, framealpha=0.92, edgecolor="0.7",
               handlelength=1.6, borderpad=0.35)


def render_frame(data: dict, k: int, frame_path: Path, dpi: int):
    fig = plt.figure(figsize=(9.0, 8.6), facecolor="white")
    ax3d = fig.add_axes([0.04, 0.47, 0.70, 0.43], projection="3d", computed_zorder=False)
    lc = draw_partial_3d(ax3d, data, k)
    cax = fig.add_axes([0.82, 0.54, 0.025, 0.30])
    if lc is None:
        lc = mpl.cm.ScalarMappable(
            cmap=plt.cm.rainbow,
            norm=mcolors.Normalize(0.0, max(float(np.nanmax(data["speed"])), 1e-6)),
        )
        lc.set_array([])
    cbar = fig.colorbar(lc, cax=cax)
    cbar.set_label("Speed, $\\|v\\|_2$ [m s$^{-1}$]", fontsize=11, labelpad=5)
    cbar.ax.tick_params(labelsize=10)

    handles = static_3d_legend_handles()
    ax3d.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.00, 1.18),
                bbox_transform=ax3d.transAxes, fontsize=10, framealpha=0.92,
                edgecolor="0.7", handlelength=1.7, borderpad=0.45,
                labelspacing=0.35)

    draw_state_panels(fig, data, k)
    fig.savefig(frame_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def make_gif(data: dict, out_path: Path, step: int, dpi: int,
             duration_ms: int, hold_frames: int, keep_frames: bool):
    frame_dir = out_path.with_suffix("").with_name(out_path.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()

    indices = frame_indices(len(data["t"]), step)
    frame_paths: list[Path] = []
    for frame_no, k in enumerate(indices):
        print(f"animation frame {frame_no + 1}/{len(indices)} source_index={k}")
        frame_path = frame_dir / f"frame_{frame_no:03d}.png"
        render_frame(data, k, frame_path, dpi)
        frame_paths.append(frame_path)

    for extra in range(max(0, hold_frames)):
        duplicate = frame_dir / f"frame_{len(frame_paths) + extra:03d}.png"
        duplicate.write_bytes(frame_paths[-1].read_bytes())
        frame_paths.append(duplicate)

    frames = [Image.open(path) for path in frame_paths]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for frame in frames:
        frame.close()
    if not keep_frames:
        for path in frame_paths:
            path.unlink(missing_ok=True)
        try:
            frame_dir.rmdir()
        except OSError:
            pass
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=HERE / "on",
                        help="Result directory containing the quad CSV and constraints.csv.")
    parser.add_argument("--csv", default="clean_terminal_eq_dense.csv",
                        help="CSV inside --dir. Dense CSV gives smoother GIFs.")
    parser.add_argument("--kind", default="single_horizon_stc",
                        choices=("auto", "single", "single_horizon_stc", "stateswitch_stc"))
    parser.add_argument("--out", type=Path, default=None,
                        help="GIF output path. Defaults to <dir>/stc_paper_los.gif.")
    parser.add_argument("--step", type=int, default=40,
                        help="Use every Nth CSV row as an animation frame.")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--duration-ms", type=int, default=110)
    parser.add_argument("--hold-frames", type=int, default=10)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    result_dir = args.dir.resolve()
    out_path = args.out if args.out is not None else result_dir / "stc_paper_los.gif"
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    data = load_animation_data(result_dir, args.csv, args.kind)
    make_gif(
        data,
        out_path=out_path,
        step=args.step,
        dpi=args.dpi,
        duration_ms=args.duration_ms,
        hold_frames=args.hold_frames,
        keep_frames=args.keep_frames,
    )
    print(f"dataset={data['kind']}  csv={data['csv_path']}")


if __name__ == "__main__":
    main()
