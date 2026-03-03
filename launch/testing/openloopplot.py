#!/usr/bin/env python3
"""
plot_open_loop_test.py
─────────────────────────────────────────────────────────────────────────────
Plots commanded_vs_actual.csv produced by open_loop_test_node.

Usage:
    python3 plot_open_loop_test.py ./logs/<folder>/commanded_vs_actual.csv

Produces:
    - Position comparison (x, y, z)
    - Velocity comparison (vx, vy, vz)
    - Position error magnitude over time
    - Attitude (quaternion) comparison
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python3 plot_open_loop_test.py <path_to_commanded_vs_actual.csv>")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

t = df["t"].values

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Open-Loop Test: Commanded vs Actual\n" + csv_path,
             fontsize=11, y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.4)

colors = {"cmd": "#1f77b4", "act": "#d62728"}

# ── Row 0: Position ──────────────────────────────────────────────────────────
for col_idx, axis in enumerate(["x", "y", "z"]):
    ax = fig.add_subplot(gs[0, col_idx])
    ax.plot(t, df[f"cmd_{axis}"].to_numpy(), label="commanded", color=colors["cmd"], lw=1.5)
    ax.plot(t, df[f"act_{axis}"].to_numpy(), label="actual",    color=colors["act"], lw=1.5, ls="--")
    ax.set_title(f"Position {axis.upper()} (m)")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# ── Row 1: Velocity ──────────────────────────────────────────────────────────
for col_idx, axis in enumerate(["vx", "vy", "vz"]):
    ax = fig.add_subplot(gs[1, col_idx])
    ax.plot(t, df[f"cmd_{axis}"].to_numpy(), label="commanded", color=colors["cmd"], lw=1.5)
    ax.plot(t, df[f"act_{axis}"].to_numpy(), label="actual",    color=colors["act"], lw=1.5, ls="--")
    ax.set_title(f"Velocity {axis.upper()} (m/s)")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# ── Row 2: Quaternion ────────────────────────────────────────────────────────
for col_idx, comp in enumerate(["qw", "qx", "qy"]):
    ax = fig.add_subplot(gs[2, col_idx])
    ax.plot(t, df[f"cmd_{comp}"].to_numpy(), label="commanded", color=colors["cmd"], lw=1.5)
    ax.plot(t, df[f"act_{comp}"].to_numpy(), label="actual",    color=colors["act"], lw=1.5, ls="--")
    ax.set_title(f"Quaternion {comp}")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# ── Row 3: Position error magnitude + qz ─────────────────────────────────────
ax_err = fig.add_subplot(gs[3, 0:2])
ax_err.plot(t, df["pos_err_norm"].to_numpy(), color="darkorange", lw=1.8, label="|pos error| (m)")
ax_err.axhline(0.05, color="gray", ls=":", lw=1, label="5 cm threshold")
ax_err.set_title("Position Error Magnitude (m)")
ax_err.set_xlabel("t (s)")
ax_err.set_ylabel("||p_act - p_cmd|| (m)")
ax_err.legend(fontsize=8)
ax_err.grid(True, alpha=0.3)

ax_qz = fig.add_subplot(gs[3, 2])
ax_qz.plot(t, df["cmd_qz"].to_numpy(), label="commanded", color=colors["cmd"], lw=1.5)
ax_qz.plot(t, df["act_qz"].to_numpy(), label="actual",    color=colors["act"], lw=1.5, ls="--")
ax_qz.set_title("Quaternion qz")
ax_qz.set_xlabel("t (s)")
ax_qz.legend(fontsize=7)
ax_qz.grid(True, alpha=0.3)

# ── Summary stats ─────────────────────────────────────────────────────────────
max_err  = df["pos_err_norm"].max()
mean_err = df["pos_err_norm"].mean()
final_err = df["pos_err_norm"].iloc[-1]
print(f"\n── Position Error Summary ──────────────────────────")
print(f"  max  : {max_err:.4f} m")
print(f"  mean : {mean_err:.4f} m")
print(f"  final: {final_err:.4f} m")
print(f"────────────────────────────────────────────────────")

if max_err < 0.05:
    print("  ✓  Actual closely tracks commanded — first-solve OK.")
    print("     Investigate the receding-horizon feedback loop.")
else:
    print("  ✗  Actual DIVERGES from commanded — OCP model / parameters are wrong.")
    print("     Check: mass, inertia, gravity, frame convention, deg2rad conversion.")

out_path = csv_path.replace(".csv", "_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")
plt.show()