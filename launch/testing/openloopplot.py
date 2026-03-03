#!/usr/bin/env python3
"""
plot_open_loop_test.py
─────────────────────────────────────────────────────────────────────────────
Plots commanded_vs_actual.csv produced by open_loop_test_node.

Usage:
    python3 plot_open_loop_test.py ./logs/<folder>/commanded_vs_actual.csv

Produces comparison plots for every state:
    - Position (x, y, z)
    - Velocity (vx, vy, vz)
    - Quaternion (qw, qx, qy, qz)
    - Angular rates (wx, wy, wz)
    - Position error magnitude
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

colors = {"cmd": "#1f77b4", "act": "#d62728"}

def make_row(fig, gs, row, labels, cmd_keys, act_keys, unit=""):
    for col, (label, ck, ak) in enumerate(zip(labels, cmd_keys, act_keys)):
        ax = fig.add_subplot(gs[row, col])
        ax.plot(t, df[ck].to_numpy(), label="cmd", color=colors["cmd"], lw=1.5)
        ax.plot(t, df[ak].to_numpy(), label="act", color=colors["act"], lw=1.5, ls="--")
        ax.set_title(f"{label}{' (' + unit + ')' if unit else ''}", fontsize=9)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

# ── Figure 1: Position, Velocity, Quaternion, Angular rates ──────────────────
fig1 = plt.figure(figsize=(16, 20))
fig1.suptitle("Open-Loop Test: Commanded vs Actual — All States\n" + csv_path,
              fontsize=11, y=0.99)

gs1 = gridspec.GridSpec(5, 3, figure=fig1, hspace=0.55, wspace=0.4)

# Row 0: Position
make_row(fig1, gs1, 0,
         ["Position X", "Position Y", "Position Z"],
         ["cmd_x",  "cmd_y",  "cmd_z"],
         ["act_x",  "act_y",  "act_z"],
         unit="m")

# Row 1: Velocity
make_row(fig1, gs1, 1,
         ["Velocity VX", "Velocity VY", "Velocity VZ"],
         ["cmd_vx", "cmd_vy", "cmd_vz"],
         ["act_vx", "act_vy", "act_vz"],
         unit="m/s")

# Row 2: Quaternion qw, qx, qy
make_row(fig1, gs1, 2,
         ["Quaternion qw", "Quaternion qx", "Quaternion qy"],
         ["cmd_qw", "cmd_qx", "cmd_qy"],
         ["act_qw", "act_qx", "act_qy"])

# Row 3: Quaternion qz + Angular rates wx, wy
make_row(fig1, gs1, 3,
         ["Quaternion qz", "Angular Rate wx", "Angular Rate wy"],
         ["cmd_qz", "cmd_wx", "cmd_wy"],
         ["act_qz", "act_wx", "act_wy"],
         unit="")

# Row 4: Angular rate wz + position error magnitude
ax_wz = fig1.add_subplot(gs1[4, 0])
ax_wz.plot(t, df["cmd_wz"].to_numpy(), label="cmd", color=colors["cmd"], lw=1.5)
ax_wz.plot(t, df["act_wz"].to_numpy(), label="act", color=colors["act"], lw=1.5, ls="--")
ax_wz.set_title("Angular Rate wz (rad/s)", fontsize=9)
ax_wz.set_xlabel("t (s)", fontsize=7)
ax_wz.legend(fontsize=7)
ax_wz.grid(True, alpha=0.3)

ax_err = fig1.add_subplot(gs1[4, 1:3])
ax_err.plot(t, df["pos_err_norm"].to_numpy(), color="darkorange", lw=1.8, label="|pos error| (m)")
ax_err.axhline(0.05, color="gray", ls=":", lw=1, label="5 cm threshold")
ax_err.set_title("Position Error Magnitude (m)", fontsize=9)
ax_err.set_xlabel("t (s)", fontsize=7)
ax_err.set_ylabel("||p_act - p_cmd|| (m)", fontsize=7)
ax_err.legend(fontsize=8)
ax_err.grid(True, alpha=0.3)

# ── Figure 2: Per-state error (act - cmd) ────────────────────────────────────
state_pairs = [
    ("x",  "cmd_x",  "act_x",  "m"),
    ("y",  "cmd_y",  "act_y",  "m"),
    ("z",  "cmd_z",  "act_z",  "m"),
    ("vx", "cmd_vx", "act_vx", "m/s"),
    ("vy", "cmd_vy", "act_vy", "m/s"),
    ("vz", "cmd_vz", "act_vz", "m/s"),
    ("qw", "cmd_qw", "act_qw", ""),
    ("qx", "cmd_qx", "act_qx", ""),
    ("qy", "cmd_qy", "act_qy", ""),
    ("qz", "cmd_qz", "act_qz", ""),
    ("wx", "cmd_wx", "act_wx", "rad/s"),
    ("wy", "cmd_wy", "act_wy", "rad/s"),
    ("wz", "cmd_wz", "act_wz", "rad/s"),
]

fig2, axes2 = plt.subplots(5, 3, figsize=(16, 18))
fig2.suptitle("Open-Loop Test: State Errors (act − cmd)\n" + csv_path,
              fontsize=11, y=0.99)
axes2 = axes2.flatten()

for i, (name, ck, ak, unit) in enumerate(state_pairs):
    err = df[ak].to_numpy() - df[ck].to_numpy()
    axes2[i].plot(t, err, color="purple", lw=1.5)
    axes2[i].axhline(0, color="gray", ls=":", lw=1)
    title = f"err {name}"
    if unit:
        title += f" ({unit})"
    axes2[i].set_title(title, fontsize=9)
    axes2[i].set_xlabel("t (s)", fontsize=7)
    axes2[i].grid(True, alpha=0.3)

# Hide unused subplot (13 states, 15 slots)
for j in range(len(state_pairs), len(axes2)):
    fig2.delaxes(axes2[j])

fig2.tight_layout(rect=[0, 0, 1, 0.97])

# ── Summary stats ─────────────────────────────────────────────────────────────
max_err   = df["pos_err_norm"].max()
mean_err  = df["pos_err_norm"].mean()
final_err = df["pos_err_norm"].iloc[-1]

print(f"\n── Position Error Summary ──────────────────────────")
print(f"  max  : {max_err:.4f} m")
print(f"  mean : {mean_err:.4f} m")
print(f"  final: {final_err:.4f} m")
print(f"────────────────────────────────────────────────────")

print("\n── Per-State Max Absolute Error ────────────────────")
for name, ck, ak, unit in state_pairs:
    err = np.abs(df[ak].to_numpy() - df[ck].to_numpy())
    print(f"  {name:>4s}: {err.max():.6f} {unit}")
print(f"────────────────────────────────────────────────────")

if max_err < 0.05:
    print("\n  ✓  Position closely tracks commanded.")
    print("     Investigate receding-horizon feedback loop.")
else:
    print("\n  ✗  Position DIVERGES from commanded.")
    print("     Check: mass, inertia, gravity, frame convention, deg2rad.")

out1 = csv_path.replace(".csv", "_states_plot.png")
out2 = csv_path.replace(".csv", "_errors_plot.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"\nPlots saved to:\n  {out1}\n  {out2}")
plt.show()