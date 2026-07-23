#!/usr/bin/env python3
"""Plot SITL planned-trajectory horizons over executed flight.

Inputs:
  - moving_executed.csv from the SITL result folder or its planner_phase subdir
  - a ROS bag containing /px4_drone/planned_trajectory as nav_msgs/Path

This script is intentionally SITL/log specific. It does not depend on the
unrelated benchmark animator.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


PLANNED = "#afdbf7"
EXECUTED = "#1f77b4"
TARGET = "#d62728"
GRID = "#d8d8d8"
CMAP = "rainbow"
NODE = "#659abb"


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{path} is empty")

    result: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (TypeError, ValueError):
                vals.append(math.nan)
        result[key] = np.asarray(vals, dtype=float)
    return result


def resolve_exec_csv(base: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else base / path

    candidates = [
        base / "planner_phase" / "moving_executed.csv",
        base / "moving_executed.csv",
    ]
    if base.name == "planner_phase":
        candidates = [base / "moving_executed.csv", base.parent / "moving_executed.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No moving_executed.csv found under {base}")


def resolve_bag(base: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else base / path

    search_dirs = [base]
    if base.name == "planner_phase":
        search_dirs.append(base.parent)
    else:
        search_dirs.append(base / "planner_phase")

    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in ("flight.bag", "stc_landing_sitl_2m.bag"):
            candidate = directory / name
            if candidate.exists():
                return candidate
        bags = sorted(directory.glob("*.bag"))
        if bags:
            return bags[0]
    raise FileNotFoundError(f"No .bag found under {base}")


def landing_start_time(bag_path: Path) -> float | None:
    """Bag time of the landing OCP activation ('Starting OCP=*landing*' in
    /rosout), so hover-to-staging-point horizons can be excluded. None if the
    bag has no landing activation (single-phase bag: keep everything)."""
    import rosbag

    with rosbag.Bag(str(bag_path)) as bag:
        for _, msg, t in bag.read_messages(topics=["/rosout"]):
            if "Starting OCP=" in msg.msg and "landing" in msg.msg:
                return t.to_sec()
    return None


def load_planned_paths(bag_path: Path, topic: str,
                       t_min: float | None = None) -> list[np.ndarray]:
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("rosbag import failed. Source ROS first, e.g. source /opt/ros/noetic/setup.zsh") from exc

    paths: list[np.ndarray] = []
    with rosbag.Bag(str(bag_path)) as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            if t_min is not None and t.to_sec() < t_min:
                continue
            pts = np.asarray(
                [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
                dtype=float,
            )
            if len(pts) > 1:
                paths.append(pts)
    if not paths:
        raise RuntimeError(f"No planned paths found on {topic} in {bag_path}")
    return paths


def seg3d(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def equalish_limits(ax, arrays: list[np.ndarray]):
    pts = np.vstack(arrays)
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    pad = np.maximum((hi - lo) * 0.12, np.array([0.35, 0.35, 0.20]))
    lo -= pad
    hi += pad
    lo[2] = min(0.0, lo[2])
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])


def plot_overlay(args):
    base = Path(args.dir)
    exec_csv = resolve_exec_csv(base, args.exec_csv)
    bag_path = resolve_bag(base, args.bag)
    data = load_csv(exec_csv)
    t_min = None if args.all_horizons else landing_start_time(bag_path)
    planned_paths = load_planned_paths(bag_path, args.topic, t_min)

    dpos = np.column_stack([data["x"], data["y"], data["z"]])
    tpos = np.column_stack([data["tx"], data["ty"], data["tz"]])
    dvel = np.column_stack([data["vx"], data["vy"], data["vz"]])
    speed = np.linalg.norm(dvel, axis=1)

    fig = plt.figure(figsize=(7.2, 5.2), facecolor="white")
    ax = fig.add_axes([0.055, 0.060, 0.700, 0.885], projection="3d", computed_zorder=False)
    cax = fig.add_axes([0.865, 0.220, 0.024, 0.560])

    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.75

    for pts in planned_paths:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=PLANNED, alpha=0.72, linewidth=1.15, zorder=2)

    lc = Line3DCollection(
        seg3d(dpos),
        cmap=plt.get_cmap(CMAP),
        norm=plt.Normalize(0.0, max(float(np.nanmax(speed)), 1e-6)),
        array=speed,
        linewidth=2.20,
        zorder=6,
    )
    ax.add_collection3d(lc)
    cbar = fig.colorbar(lc, cax=cax)
    cbar.set_label(r"Speed, $\|v\|_2$ [m s$^{-1}$]")

    ax.plot(tpos[:, 0], tpos[:, 1], tpos[:, 2], color=TARGET, linewidth=1.25,
            linestyle="--", alpha=0.62, zorder=3)

    starts = np.asarray([pts[0] for pts in planned_paths])
    ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2], s=6, c=NODE,
               depthshade=False, zorder=10)
    ax.scatter([dpos[0, 0]], [dpos[0, 1]], [dpos[0, 2]], s=12, c="black",
               depthshade=False, zorder=11)
    ax.scatter([dpos[-1, 0]], [dpos[-1, 1]], [dpos[-1, 2]], s=32, c=TARGET,
               depthshade=False, zorder=11)
    ax.scatter([tpos[-1, 0]], [tpos[-1, 1]], [tpos[-1, 2]], s=42, marker="s",
               c=TARGET, depthshade=False, zorder=11)

    equalish_limits(ax, [dpos, tpos, *planned_paths])
    try:
        ax.set_box_aspect((1.15, 1.15, 0.85))
    except Exception:
        pass

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    if args.title:
        ax.set_title(args.title, fontsize=10, pad=6)
    ax.view_init(elev=args.elev, azim=args.azim if args.azim is not None else (122.0 if args.flip_view else -58.0))

    handles = [
        Line2D([0], [0], color=EXECUTED, linewidth=2.2, label="Executed (SITL)"),
        Line2D([0], [0], color=PLANNED, linewidth=1.5,
               label="Full MPC predictions (first segment executed)"),
        Line2D([0], [0], color=TARGET, linewidth=1.25, linestyle="--", label="Target path"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TARGET,
               markeredgecolor=TARGET, markersize=5, label="Target at touchdown"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.88)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(out.resolve())


def main():
    parser = argparse.ArgumentParser(description="Plot SITL planned horizons over executed flight.")
    parser.add_argument("--dir", required=True, help="SITL result dir or planner_phase dir")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--exec-csv", default=None, help="Override executed CSV path")
    parser.add_argument("--bag", default=None, help="Override bag path")
    parser.add_argument("--topic", default="/px4_drone/planned_trajectory")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--flip-view", action="store_true", help="Use opposite-side 3D camera")
    parser.add_argument("--azim", type=float, default=None, help="Explicit matplotlib azimuth")
    parser.add_argument("--elev", type=float, default=13.0, help="Matplotlib elevation")
    parser.add_argument("--title", default=None, help="Optional plot title; omitted by default")
    parser.add_argument("--all-horizons", action="store_true",
                        help="Also plot pre-landing (hover) planned horizons")
    plot_overlay(parser.parse_args())


if __name__ == "__main__":
    main()
