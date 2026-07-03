#!/usr/bin/env python3
"""Post-flight one-shot: collect everything for a run into
CoManDO_planner/logs/<run_name>/ and generate the paper figures there.

Given a flight bag, this script:
  1. finds the planner node's own log folder (newest under ~/.ros/logs —
     that's where roslaunch-started nodes write ./logs) and copies its
     all_solves.csv + solver_events.csv into logs/<run_name>/ here,
  2. copies the bag there as flight.bag,
  3. converts it to moving_executed.csv + constraints.csv (bag_to_stc_csv),
     trimmed to [landing OCP activation, touchdown at relative z = 0],
  4. renders stc_flight_3d.png + stc_flight_states.png (plot_stc_paper.py)
     and stc_flight.gif (animate_stc_paper.py) into the same folder.

So one folder in this directory holds everything for a run — same layout as
the existing stc_landing_sitl_2m/ and hardware/ results.

Usage:
  python3 analyze_flight.py --bag /tmp/flight.bag --name stc_landing_sitl_1p8
  python3 analyze_flight.py --bag ...              # name = planner run folder
  python3 analyze_flight.py --bag ... --t1 12 --no-gif
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import rosbag

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bag_to_stc_csv  # noqa: E402

# Where the planner node writes its run folders. Since the log_dir param was
# added, planner_launch.launch points it here (CoManDO_planner/logs); older
# runs (or nodes launched without the param) are under ~/.ros/logs.
PLANNER_LOGS = [HERE, Path.home() / ".ros" / "logs"]
# Where runs are collected: this directory (CoManDO_planner/logs).
OUT_ROOT = HERE


def newest_planner_run(pattern: str) -> Path:
    """Newest planner log folder by all_solves.csv mtime — the planner
    appends to that file during flight, so it identifies the run that
    actually just flew (dir mtime alone is unreliable)."""
    dirs = [d for root in PLANNER_LOGS for d in root.glob(pattern)
            if d.is_dir() and (d / "all_solves.csv").exists()]
    if not dirs:
        raise SystemExit(f"no run dir matching {pattern} under "
                         f"{' or '.join(str(r) for r in PLANNER_LOGS)} — "
                         "was the planner profile actually triggered?")
    return max(dirs, key=lambda d: (d / "all_solves.csv").stat().st_mtime)


def auto_trim_window(bag_path: Path):
    """[t0, t1] relative to the first drone-odom stamp: from the landing OCP
    activation (planner's 'Starting OCP=' line in /rosout; falls back to the
    first setpoint) to the last arm→disarm transition (touchdown stop)."""
    t_odom0 = None
    t_sp0 = None
    t_ocp = None
    disarm_t = None
    prev_armed = None
    with rosbag.Bag(str(bag_path)) as bag:
        # Same stamp policy as bag_to_stc_csv.read_bag: trust the header
        # stamp only if it agrees with the bag record time, else use the
        # bag record time (mavros stamps are garbage under broken time sync).
        def stamp_of(m, bag_t):
            h = m.header.stamp.to_sec()
            return h if abs(h - bag_t.to_sec()) < 10.0 else bag_t.to_sec()

        for _, m, bt in bag.read_messages(topics=["/mavros/local_position/odom"]):
            t_odom0 = stamp_of(m, bt)
            break
        for _, m, bt in bag.read_messages(topics=["/mavros/setpoint_raw/local"]):
            t_sp0 = stamp_of(m, bt)
            break
        for _, m, t in bag.read_messages(topics=["/mavros/state"]):
            if prev_armed and not m.armed:
                disarm_t = t.to_sec()
            prev_armed = m.armed
        starts = []
        for _, m, t in bag.read_messages(topics=["/rosout"]):
            if "Starting OCP=" in m.msg:
                starts.append((t.to_sec(), m.msg))
        # A session is usually hover-to-staging-point first, then the landing
        # OCP. The figures are about the landing: trim from the landing
        # activation so the hover phase stays out of the plots.
        landing = [ts for ts, msg in starts if "landing" in msg]
        if landing:
            t_ocp = landing[0]
        elif starts:
            t_ocp = starts[0][0]
    if t_odom0 is None:
        return None, None
    t_start = t_ocp if t_ocp is not None else t_sp0
    t0 = max(0.0, t_start - t_odom0 - 1.0) if t_start is not None else None
    t1 = disarm_t - t_odom0 + 0.5 if disarm_t is not None else None
    return t0, t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--name", default=None,
                    help="run folder name under logs/ (default: the planner "
                         "run folder's own name)")
    ap.add_argument("--planner-run", type=Path, default=None,
                    help="planner log folder (default: newest under ~/.ros/logs)")
    # The planner's CsvLogger initializes once per node lifetime and names the
    # folder after the FIRST profile applied — a hover-then-land session logs
    # everything under a *_hover_* name, so match any run dir and take the
    # newest by mtime (the planner appends to all_solves.csv during flight).
    ap.add_argument("--run-glob", default="*_*_*_*_*",
                    help="glob for planner run auto-detection")
    ap.add_argument("--t0", type=float, default=None,
                    help="override auto trim start [s from first odom]")
    ap.add_argument("--t1", type=float, default=None,
                    help="override auto trim end [s from first odom]")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not args.bag.exists():
        raise SystemExit(f"bag not found: {args.bag}")

    planner_run = (args.planner_run or newest_planner_run(args.run_glob)).resolve()
    run_dir = OUT_ROOT / (args.name or planner_run.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"planner logs: {planner_run}")
    print(f"run dir:      {run_dir}")

    for f in ("all_solves.csv", "solver_events.csv"):
        src = planner_run / f
        if src.exists() and src.resolve() != (run_dir / f).resolve():
            shutil.copy2(src, run_dir / f)

    bag_dest = run_dir / "flight.bag"
    if args.bag.resolve() != bag_dest:
        shutil.copy2(args.bag, bag_dest)
        print(f"copied bag -> {bag_dest}")

    t0, t1 = args.t0, args.t1
    if t0 is None and t1 is None:
        t0, t1 = auto_trim_window(bag_dest)
        print(f"auto trim: t0={t0} t1={t1}")
    bag_to_stc_csv.convert(bag_dest, run_dir, t0, t1)

    py = sys.executable or "python3"
    subprocess.run([py, str(HERE / "plot_stc_paper.py"),
                    "--dir", str(run_dir),
                    "--out", str(run_dir / "stc_flight.png"),
                    "--dpi", str(args.dpi)], check=True)
    if not args.no_gif:
        subprocess.run([py, str(HERE / "animate_stc_paper.py"),
                        "--dir", str(run_dir),
                        "--csv", "moving_executed.csv",
                        "--kind", "stateswitch_stc",
                        "--out", str(run_dir / "stc_flight.gif"),
                        "--step", "10", "--dpi", "90"], check=True)

    print("\nrun folder contents:")
    for p in sorted(run_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
