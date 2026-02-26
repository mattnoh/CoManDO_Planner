#!/usr/bin/env python3
"""
Plot OCP constraint violations from logged flight data.

Usage:
    python3 plot_constraints.py /path/to/flight_folder [--cutoff TIME] [--latest]

CSV files used:
    commanded_state.csv
    first_solve_trajectory.csv (optional)

Constraints (from the OCP header):
    - Max thrust         : ‖f‖ ≤ FMAX
    - Glideslope         : √(x²+y²) ≤ tan(θ_gs) · z   (θ_gs = 70°)
    - Thrust cone        : √(fx²+fy²) ≤ tan(θ_tc) · fz   (θ_tc = 60°)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from math import tan, radians

# ── OCP constraint parameters (mirroring the header) ─────────────────────
FMAX           = 1.2          # N
GLIDESLOPE_DEG = 70.0         # degrees
THRUST_CONE_DEG = 60.0        # degrees

# ── CSV column indices (same as logger.py) ───────────────────────────────
# commanded_state.csv
C_TS  = 0
C_X,  C_Y,  C_Z  = 1,  2,  3
C_VX, C_VY, C_VZ = 4,  5,  6
C_QW, C_QX, C_QY, C_QZ = 7, 8, 9, 10
C_WX, C_WY, C_WZ = 11, 12, 13
C_FX, C_FY, C_FZ = 14, 15, 16
C_MX, C_MY, C_MZ = 17, 18, 19
C_TNORM          = 20

# first_solve_trajectory.csv
F_NODE, F_T      = 0, 1
F_X,  F_Y,  F_Z  = 2,  3,  4
F_VX, F_VY, F_VZ = 5,  6,  7
F_QW, F_QX, F_QY, F_QZ = 8, 9, 10, 11
F_WX, F_WY, F_WZ = 12, 13, 14
F_FX, F_FY, F_FZ = 15, 16, 17
F_MX, F_MY, F_MZ = 18, 19, 20
F_TNORM          = 21


# ── Helper functions (from logger.py) ────────────────────────────────────
def load_csv(filepath, cutoff_time=None):
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data is None or data.ndim < 2 or data.shape[0] == 0:
            return None
        if cutoff_time is not None:
            rel  = data[:, 0] - data[0, 0]
            mask = rel <= cutoff_time
            if not np.any(mask):
                print(f"Warning: cutoff {cutoff_time}s is before all data in {filepath}")
                return None
            data = data[mask]
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_first_traj(filepath):
    """Load first_solve_trajectory.csv — no cutoff, uses horizon time column."""
    try:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data is None or data.ndim < 2 or data.shape[0] == 0:
            return None
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def find_latest_flight_folder(logs_dir):
    folders = [p for p in Path(logs_dir).iterdir()
               if p.is_dir() and "_flight_" in p.name]
    return str(max(folders, key=lambda p: p.stat().st_mtime)) if folders else None


# ── Main plotting function ───────────────────────────────────────────────
def plot_constraint_violations(folder_path, cutoff_time=None):
    # Load commanded state (required)
    cmd = load_csv(os.path.join(folder_path, "commanded_state.csv"), cutoff_time)
    if cmd is None:
        print("Error: could not load commanded_state.csv")
        return

    # Load first‑solve trajectory (optional)
    fs = load_first_traj(os.path.join(folder_path, "first_solve_trajectory.csv"))
    if fs is None:
        print("Note: first_solve_trajectory.csv not found – first‑solve overlay disabled")

    cmd_t = cmd[:, C_TS] - cmd[0, C_TS]          # relative time

    # Extract commanded quantities
    x = cmd[:, C_X]
    y = cmd[:, C_Y]
    z = cmd[:, C_Z]
    fx = cmd[:, C_FX]
    fy = cmd[:, C_FY]
    fz = cmd[:, C_FZ]
    thrust_norm = cmd[:, C_TNORM]

    # Constraint parameters
    tan_gs = tan(radians(GLIDESLOPE_DEG))
    tan_tc = tan(radians(THRUST_CONE_DEG))

    # Compute residuals (positive ⟹ violation)
    thrust_viol = thrust_norm - FMAX
    horiz_dist = np.sqrt(x**2 + y**2)
    gs_viol = horiz_dist - tan_gs * z
    lateral_thrust = np.sqrt(fx**2 + fy**2)
    tc_viol = lateral_thrust - tan_tc * fz   # works even when fz <= 0

    # Prepare figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ── Max thrust ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(cmd_t, thrust_viol, 'r-', lw=2, label='Commanded')
    if fs is not None:
        fs_t = fs[:, F_T]
        fs_fn = fs[:, F_TNORM]
        fs_viol = fs_fn - FMAX
        ax.plot(fs_t, fs_viol, 'r--', lw=1.5, alpha=0.7, label='First‑solve')
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.fill_between(cmd_t, 0, thrust_viol, where=thrust_viol>0,
                    color='red', alpha=0.3, label='Violation')
    ax.set_ylabel('Thrust norm – Fmax (N)')
    ax.set_title(f'Max Thrust Constraint (Fmax = {FMAX} N)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # ── Glideslope ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(cmd_t, gs_viol, 'b-', lw=2, label='Commanded')
    if fs is not None:
        fs_x = fs[:, F_X]; fs_y = fs[:, F_Y]; fs_z = fs[:, F_Z]
        fs_horiz = np.sqrt(fs_x**2 + fs_y**2)
        fs_gs_viol = fs_horiz - tan_gs * fs_z
        ax.plot(fs_t, fs_gs_viol, 'b--', lw=1.5, alpha=0.7, label='First‑solve')
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.fill_between(cmd_t, 0, gs_viol, where=gs_viol>0,
                    color='red', alpha=0.3, label='Violation')
    ax.set_ylabel('√(x²+y²) – tan(gs)·z (m)')
    ax.set_title(f'Glideslope Constraint (gs = {GLIDESLOPE_DEG}°)')
    ax.legend(loc='upper right')
    ax.grid(True)

    # ── Thrust cone ───────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(cmd_t, tc_viol, 'g-', lw=2, label='Commanded')
    if fs is not None:
        fs_fx = fs[:, F_FX]; fs_fy = fs[:, F_FY]; fs_fz = fs[:, F_FZ]
        fs_lat = np.sqrt(fs_fx**2 + fs_fy**2)
        fs_tc_viol = fs_lat - tan_tc * fs_fz
        ax.plot(fs_t, fs_tc_viol, 'g--', lw=1.5, alpha=0.7, label='First‑solve')
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.fill_between(cmd_t, 0, tc_viol, where=tc_viol>0,
                    color='red', alpha=0.3, label='Violation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('√(fx²+fy²) – tan(tc)·fz (N)')
    ax.set_title(f'Thrust Cone Constraint (tc = {THRUST_CONE_DEG}°)')
    ax.legend(loc='upper right')
    ax.grid(True)

    fig.suptitle(f'OCP Constraint Violations: {os.path.basename(folder_path)}'
                 + (f' (Cutoff: {cutoff_time}s)' if cutoff_time else ''),
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    suffix = f"_cutoff_{cutoff_time}s" if cutoff_time else ""
    out = os.path.join(folder_path, f"constraint_violations{suffix}.pdf")
    plt.savefig(out, bbox_inches='tight')
    print(f"Constraint violations plot saved to: {out}")
    print_constraint_summary(cmd_t, thrust_viol, gs_viol, tc_viol, z)
    plt.show()

def print_constraint_summary(t, thrust_viol, gs_viol, tc_viol, z):
    """Print a summary of constraint violations."""
    print("\n=== Constraint Violation Summary ===")

    # Thrust
    max_thrust_viol = np.max(thrust_viol)
    if max_thrust_viol > 0:
        idx_max = np.argmax(thrust_viol)
        time_max = t[idx_max]
        print(f"Max thrust violation: {max_thrust_viol:.4f} N at t = {time_max:.2f} s")
        viol_mask = thrust_viol > 0
        total_viol_time = np.sum(np.diff(t, prepend=t[0]) * viol_mask)
        percent = 100 * total_viol_time / (t[-1] - t[0])
        print(f"  Total time in violation: {total_viol_time:.2f} s ({percent:.1f}%)")
    else:
        print("Thrust constraint: OK (no violation)")

    # Glideslope
    max_gs_viol = np.max(gs_viol)
    if max_gs_viol > 0:
        idx_max = np.argmax(gs_viol)
        time_max = t[idx_max]
        print(f"Max glideslope violation: {max_gs_viol:.4f} m at t = {time_max:.2f} s")
        viol_mask = gs_viol > 0
        total_viol_time = np.sum(np.diff(t, prepend=t[0]) * viol_mask)
        percent = 100 * total_viol_time / (t[-1] - t[0])
        print(f"  Total time in violation: {total_viol_time:.2f} s ({percent:.1f}%)")
        if np.any(z <= 0):
            print("  Warning: z <= 0 detected (glideslope constraint may be ill‑defined)")
    else:
        print("Glideslope constraint: OK (no violation)")

    # Thrust cone
    max_tc_viol = np.max(tc_viol)
    if max_tc_viol > 0:
        idx_max = np.argmax(tc_viol)
        time_max = t[idx_max]
        print(f"Max thrust‑cone violation: {max_tc_viol:.4f} N at t = {time_max:.2f} s")
        viol_mask = tc_viol > 0
        total_viol_time = np.sum(np.diff(t, prepend=t[0]) * viol_mask)
        percent = 100 * total_viol_time / (t[-1] - t[0])
        print(f"  Total time in violation: {total_viol_time:.2f} s ({percent:.1f}%)")
    else:
        print("Thrust‑cone constraint: OK (no violation)")

    print("=====================================\n")

# ── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot OCP constraint violations')
    parser.add_argument('path', nargs='?', help='Flight folder or logs directory')
    parser.add_argument('--cutoff', type=float, help='Cutoff time (seconds)')
    parser.add_argument('--latest', action='store_true',
                        help='Use the most recent flight folder in the given path (interpreted as logs dir)')
    args = parser.parse_args()

    if args.latest:
        logs_dir = args.path or "./logs"
        if not os.path.exists(logs_dir):
            logs_dir = "../logs"
        folder_path = find_latest_flight_folder(logs_dir)
        if folder_path is None:
            print(f"Error: no flight folders found in {logs_dir}")
            sys.exit(1)
        print(f"Using latest flight folder: {folder_path}")
    elif args.path:
        folder_path = args.path
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a directory")
            sys.exit(1)
    else:
        print("Usage: python3 plot_constraints.py [path] [--cutoff TIME] [--latest]")
        sys.exit(1)

    if not os.path.exists(os.path.join(folder_path, "commanded_state.csv")):
        print(f"Error: commanded_state.csv missing from {folder_path}")
        sys.exit(1)

    plot_constraint_violations(folder_path, args.cutoff)
    print("\nDone!")