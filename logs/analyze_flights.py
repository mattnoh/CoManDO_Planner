#!/usr/bin/env python3
"""Analyze both flight logs to diagnose issues."""
import csv
import math
import os

def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

def float_or_none(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

# ============================================================
# FLIGHT 1: Landing open loop
# ============================================================
print("=" * 80)
print("FLIGHT 1: gogogo_landing_open_loop_alipddp_20260327_154544")
print("=" * 80)

base1 = "gogogo_landing_open_loop_alipddp_20260327_154544"
cmd1 = read_csv(os.path.join(base1, "commanded_state.csv"))
act1 = read_csv(os.path.join(base1, "actual_state.csv"))
solves1 = read_csv(os.path.join(base1, "all_solves.csv"))

print(f"\nCommanded states: {len(cmd1)} rows")
print(f"Actual states:   {len(act1)} rows")
print(f"Solve entries:   {len(solves1)} rows")

# Initial states comparison
print("\n--- INITIAL STATE COMPARISON (t=0) ---")
c0 = cmd1[0]
a0 = act1[0]
print(f"  Commanded initial: x={c0['x']}, y={c0['y']}, z={c0['z']}")
print(f"  Actual initial:    x={a0['x']}, y={a0['y']}, z={a0['z']}")
print(f"  Commanded initial vel: vx={c0['vx']}, vy={c0['vy']}, vz={c0['vz']}")
print(f"  Actual initial vel:    vx={a0['vx']}, vy={a0['vy']}, vz={a0['vz']}")
print(f"  Commanded initial quat: qw={c0['qw']}, qx={c0['qx']}, qy={c0['qy']}, qz={c0['qz']}")
print(f"  Actual initial quat:    qw={a0['qw']}, qx={a0['qx']}, qy={a0['qy']}, qz={a0['qz']}")

dx = float(c0['x']) - float(a0['x'])
dy = float(c0['y']) - float(a0['y'])
dz = float(c0['z']) - float(a0['z'])
pos_err = math.sqrt(dx**2 + dy**2 + dz**2)
print(f"\n  Initial position error: {pos_err:.6f} m")

dvx = float(c0['vx']) - float(a0['vx'])
dvy = float(c0['vy']) - float(a0['vy'])
dvz = float(c0['vz']) - float(a0['vz'])
vel_err = math.sqrt(dvx**2 + dvy**2 + dvz**2)
print(f"  Initial velocity error: {vel_err:.6f} m/s")

# Final states
print("\n--- FINAL STATE ---")
cf = cmd1[-1]
af = act1[-1]
print(f"  Commanded final: x={cf['x']}, y={cf['y']}, z={cf['z']}")
print(f"  Actual final:    x={af['x']}, y={af['y']}, z={af['z']}")
print(f"  Timestamp cmd: {cf['timestamp']}, actual: {af['timestamp']}")

# Target from solve data
s0 = solves1[0]
print(f"\n--- SOLVE INFO ---")
print(f"  Solve 0: time={s0['solve_time_ms']}ms, iters={s0['solve_iters']}, coord_mode={s0['coord_mode']}")
print(f"  Target: x={s0['tgt_x']}, y={s0['tgt_y']}, z={s0['tgt_z']}")

# Glideslope analysis for landing
print("\n--- GLIDESLOPE CONE ANALYSIS ---")
tgt_x = float(s0['tgt_x'])
tgt_y = float(s0['tgt_y'])
tgt_z = float(s0['tgt_z'])
print(f"  Target position: ({tgt_x}, {tgt_y}, {tgt_z})")

# Analyze both commanded and actual trajectories for glideslope
# The glideslope cone is typically: sqrt((x-xt)^2 + (y-yt)^2) <= k * (z - zt)  for some k
# Let's compute the ratio for both trajectories
print("\n  Timestep | CMD r/h ratio | ACT r/h ratio | CMD(x,y,z) | ACT(x,y,z)")
print("  " + "-" * 120)

max_cmd_ratio = 0
max_act_ratio = 0
max_cmd_ratio_idx = 0
max_act_ratio_idx = 0

for i in range(min(len(cmd1), len(act1))):
    cx, cy, cz = float(cmd1[i]['x']), float(cmd1[i]['y']), float(cmd1[i]['z'])
    ax, ay, az = float(act1[i]['x']), float(act1[i]['y']), float(act1[i]['z'])
    
    cr = math.sqrt((cx - tgt_x)**2 + (cy - tgt_y)**2)
    ch = cz - tgt_z
    ar = math.sqrt((ax - tgt_x)**2 + (ay - tgt_y)**2)
    ah = az - tgt_z
    
    c_ratio = cr / ch if ch > 0.001 else float('inf')
    a_ratio = ar / ah if ah > 0.001 else float('inf')
    
    if c_ratio < 100 and c_ratio > max_cmd_ratio:
        max_cmd_ratio = c_ratio
        max_cmd_ratio_idx = i
    if a_ratio < 100 and a_ratio > max_act_ratio:
        max_act_ratio = a_ratio
        max_act_ratio_idx = i
    
    if i < 10 or i > len(cmd1) - 5 or i % 10 == 0:
        print(f"  {i:5d}    | {c_ratio:13.4f} | {a_ratio:13.4f} | ({cx:.4f},{cy:.4f},{cz:.4f}) | ({ax:.4f},{ay:.4f},{az:.4f})")

print(f"\n  Max CMD glideslope ratio: {max_cmd_ratio:.4f} at step {max_cmd_ratio_idx}")
print(f"  Max ACT glideslope ratio: {max_act_ratio:.4f} at step {max_act_ratio_idx}")

# Position error over time
print("\n--- TRACKING ERROR OVER TIME (every 10 steps) ---")
print("  Step | dt(s) | pos_err(m) | vel_err(m/s) | z_cmd | z_act | vz_cmd | vz_act")
print("  " + "-" * 100)
for i in range(0, min(len(cmd1), len(act1)), 5):
    cx, cy, cz = float(cmd1[i]['x']), float(cmd1[i]['y']), float(cmd1[i]['z'])
    ax, ay, az = float(act1[i]['x']), float(act1[i]['y']), float(act1[i]['z'])
    cvx, cvy, cvz = float(cmd1[i]['vx']), float(cmd1[i]['vy']), float(cmd1[i]['vz'])
    avx, avy, avz = float(act1[i]['vx']), float(act1[i]['vy']), float(act1[i]['vz'])
    
    pe = math.sqrt((cx-ax)**2 + (cy-ay)**2 + (cz-az)**2)
    ve = math.sqrt((cvx-avx)**2 + (cvy-avy)**2 + (cvz-avz)**2)
    dt = float(cmd1[i]['timestamp']) - float(cmd1[0]['timestamp'])
    
    print(f"  {i:5d} | {dt:5.2f} | {pe:10.4f} | {ve:12.4f} | {cz:7.4f} | {az:7.4f} | {cvz:8.4f} | {avz:8.4f}")

# Check for the "gap" hypothesis - look at first few timesteps for sudden z drop
print("\n--- INITIAL Z AND VZ ANALYSIS (first 10 steps) ---")
print("  Step | dt(s) | z_cmd | z_act | dz | vz_cmd | vz_act | dvz")
for i in range(min(10, len(cmd1))):
    cx, cy, cz = float(cmd1[i]['x']), float(cmd1[i]['y']), float(cmd1[i]['z'])
    ax, ay, az = float(act1[i]['x']), float(act1[i]['y']), float(act1[i]['z'])
    cvz = float(cmd1[i]['vz'])
    avz = float(act1[i]['vz'])
    dt = float(cmd1[i]['timestamp']) - float(cmd1[0]['timestamp'])
    print(f"  {i:5d} | {dt:5.3f} | {cz:.6f} | {az:.6f} | {cz-az:.6f} | {cvz:8.5f} | {avz:8.5f} | {cvz-avz:8.5f}")

# ============================================================
# FLIGHT 2: Tracking circle target
# ============================================================
print("\n\n" + "=" * 80)
print("FLIGHT 2: gogogo_tracking_circle_target_open_loop_alipddp_20260327_164635")
print("=" * 80)

base2 = "gogogo_tracking_circle_target_open_loop_alipddp_20260327_164635"
cmd2 = read_csv(os.path.join(base2, "commanded_state.csv"))
act2 = read_csv(os.path.join(base2, "actual_state.csv"))
solves2 = read_csv(os.path.join(base2, "all_solves.csv"))

print(f"\nCommanded states: {len(cmd2)} rows")
print(f"Actual states:   {len(act2)} rows")
print(f"Solve entries:   {len(solves2)} rows")

# Initial states comparison
print("\n--- INITIAL STATE COMPARISON (t=0) ---")
c0 = cmd2[0]
a0 = act2[0]
print(f"  Commanded initial: x={c0['x']}, y={c0['y']}, z={c0['z']}")
print(f"  Actual initial:    x={a0['x']}, y={a0['y']}, z={a0['z']}")
print(f"  Commanded initial vel: vx={c0['vx']}, vy={c0['vy']}, vz={c0['vz']}")
print(f"  Actual initial vel:    vx={a0['vx']}, vy={a0['vy']}, vz={a0['vz']}")

dx = float(c0['x']) - float(a0['x'])
dy = float(c0['y']) - float(a0['y'])
dz = float(c0['z']) - float(a0['z'])
pos_err = math.sqrt(dx**2 + dy**2 + dz**2)
print(f"\n  Initial position error: {pos_err:.6f} m")

dvx = float(c0['vx']) - float(a0['vx'])
dvy = float(c0['vy']) - float(a0['vy'])
dvz = float(c0['vz']) - float(a0['vz'])
vel_err = math.sqrt(dvx**2 + dvy**2 + dvz**2)
print(f"  Initial velocity error: {vel_err:.6f} m/s")

# Solve data analysis
print("\n--- SOLVE INFO ---")
solve_nums = set()
for s in solves2:
    solve_nums.add(s['solve_num'])
print(f"  Number of solves: {len(solve_nums)}")

s0 = solves2[0]
print(f"  Solve 0: time={s0['solve_time_ms']}ms, iters={s0['solve_iters']}, coord_mode={s0['coord_mode']}")
print(f"  Target node 0: x={s0['tgt_x']}, y={s0['tgt_y']}, z={s0['tgt_z']}")
print(f"  Target vel node 0: vx={s0['tgt_vx']}, vy={s0['tgt_vy']}, vz={s0['tgt_vz']}")
print(f"  Circle: center=({s0['circle_center_x']},{s0['circle_center_y']},{s0['circle_center_z']})")
print(f"          radius={s0['circle_radius']}, omega={s0['circle_omega']}, phi0={s0['circle_phi0']}")
print(f"          t_abs={s0['circle_t_abs']}")
print(f"  Abs state: x={s0['abs_x']}, y={s0['abs_y']}, z={s0['abs_z']}")
print(f"  Rel state: x={s0['rel_x']}, y={s0['rel_y']}, z={s0['rel_z']}")

# Check circle_phi0 and circle_t_abs for sanity
phi0 = float(s0['circle_phi0'])
t_abs = float(s0['circle_t_abs'])
print(f"\n  *** CRITICAL: circle_phi0 = {phi0:.6f}")
print(f"  *** CRITICAL: circle_t_abs = {t_abs:.6f}")
if abs(phi0) > 1e6 or abs(t_abs) > 1e6:
    print(f"  *** WARNING: These values look EXTREMELY LARGE - possible overflow or uninitialized data!")

# Verify if target position makes sense given circle params
cx_circ = float(s0['circle_center_x'])
cy_circ = float(s0['circle_center_y'])
cz_circ = float(s0['circle_center_z'])
r_circ = float(s0['circle_radius'])
omega_circ = float(s0['circle_omega'])

# target should be: center + radius * [cos(omega*t + phi0), sin(omega*t + phi0), 0]
# at t=0 of the solve
theta0 = omega_circ * 0 + phi0  # phi0 is the initial phase
expected_tgt_x = cx_circ + r_circ * math.cos(theta0)
expected_tgt_y = cy_circ + r_circ * math.sin(theta0)
print(f"\n  Expected target at theta={theta0:.4f}: x={expected_tgt_x:.4f}, y={expected_tgt_y:.4f}")
print(f"  Actual target:                        x={s0['tgt_x']}, y={s0['tgt_y']}")
print(f"  (Note: theta is {theta0 % (2*math.pi):.4f} mod 2pi)")

# Check trajectory divergence
print("\n--- TRAJECTORY DIVERGENCE ANALYSIS ---")
print("  Step | dt(s) | x_cmd | y_cmd | z_cmd | x_act | y_act | z_act | pos_err")
print("  " + "-" * 100)
for i in range(len(cmd2)):
    cx, cy, cz = float(cmd2[i]['x']), float(cmd2[i]['y']), float(cmd2[i]['z'])
    if i < len(act2):
        ax, ay, az = float(act2[i]['x']), float(act2[i]['y']), float(act2[i]['z'])
        pe = math.sqrt((cx-ax)**2 + (cy-ay)**2 + (cz-az)**2)
    else:
        ax, ay, az = 0, 0, 0
        pe = -1
    dt = float(cmd2[i]['timestamp']) - float(cmd2[0]['timestamp'])
    if i < 15 or i > len(cmd2) - 5 or i % 10 == 0:
        print(f"  {i:5d} | {dt:6.2f} | {cx:8.3f} | {cy:8.3f} | {cz:8.3f} | {ax:8.3f} | {ay:8.3f} | {az:8.3f} | {pe:8.3f}")

# Velocity analysis
print("\n--- VELOCITY ANALYSIS (first 15 steps) ---")
print("  Step | dt(s) | vx_cmd | vy_cmd | vz_cmd | vx_act | vy_act | vz_act | speed_cmd | speed_act")
print("  " + "-" * 110)
for i in range(min(15, len(cmd2))):
    cvx, cvy, cvz = float(cmd2[i]['vx']), float(cmd2[i]['vy']), float(cmd2[i]['vz'])
    if i < len(act2):
        avx, avy, avz = float(act2[i]['vx']), float(act2[i]['vy']), float(act2[i]['vz'])
        aspeed = math.sqrt(avx**2 + avy**2 + avz**2)
    else:
        avx, avy, avz = 0, 0, 0
        aspeed = 0
    cspeed = math.sqrt(cvx**2 + cvy**2 + cvz**2)
    dt = float(cmd2[i]['timestamp']) - float(cmd2[0]['timestamp'])
    print(f"  {i:5d} | {dt:5.2f} | {cvx:7.3f} | {cvy:7.3f} | {cvz:7.3f} | {avx:7.3f} | {avy:7.3f} | {avz:7.3f} | {cspeed:9.3f} | {aspeed:9.3f}")

# Check if the planned trajectory itself is diverging (commanded state)
print("\n--- COMMANDED TRAJECTORY RATE OF GROWTH ---")
for i in range(1, min(20, len(cmd2))):
    cx, cy, cz = float(cmd2[i]['x']), float(cmd2[i]['y']), float(cmd2[i]['z'])
    cx0, cy0, cz0 = float(cmd2[0]['x']), float(cmd2[0]['y']), float(cmd2[0]['z'])
    dist = math.sqrt((cx-cx0)**2 + (cy-cy0)**2 + (cz-cz0)**2)
    dt = float(cmd2[i]['timestamp']) - float(cmd2[0]['timestamp'])
    print(f"  Step {i:3d}: dt={dt:5.2f}s, dist from start={dist:.4f}m")

# Acceleration analysis
print("\n--- ACCELERATION ANALYSIS (cmd) ---")
if 'acc_x' in cmd2[0]:
    for i in range(min(15, len(cmd2))):
        ax_a, ay_a, az_a = float(cmd2[i]['acc_x']), float(cmd2[i]['acc_y']), float(cmd2[i]['acc_z'])
        acc_mag = math.sqrt(ax_a**2 + ay_a**2 + az_a**2)
        dt = float(cmd2[i]['timestamp']) - float(cmd2[0]['timestamp'])
        print(f"  Step {i:3d}: dt={dt:5.2f}s, acc=({ax_a:.4f},{ay_a:.4f},{az_a:.4f}), |acc|={acc_mag:.4f}")

# Check all_solves for the per-node trajectory of solve 0
print("\n--- SOLVE 0: FULL PLANNED TRAJECTORY (absolute positions) ---")
solve0_nodes = [s for s in solves2 if s['solve_num'] == '0']
print(f"  Nodes in solve 0: {len(solve0_nodes)}")
for s in solve0_nodes:
    node = s['node']
    t = s['t']
    ax_s = s['abs_x']
    ay_s = s['abs_y']
    az_s = s['abs_z']
    avx = s['abs_vx']
    avy = s['abs_vy']
    avz = s['abs_vz']
    tx = s['tgt_x']
    ty = s['tgt_y']
    tz = s['tgt_z']
    rx = s['rel_x']
    ry = s['rel_y']
    rz = s['rel_z']
    print(f"  node={node:>3s} t={t:>8s} abs=({ax_s:>10s},{ay_s:>10s},{az_s:>10s}) tgt=({tx:>10s},{ty:>10s},{tz:>10s}) rel=({rx:>10s},{ry:>10s},{rz:>10s})")

# Compare with gazebo sim solves if available
print("\n--- CHECKING cf_1 (gazebo) SOLVES ---")
base_gz = "cf_1_tracking_circle_target_open_loop_alipddp_20260327_065630"
if os.path.exists(os.path.join(base_gz, "all_solves.csv")):
    solves_gz = read_csv(os.path.join(base_gz, "all_solves.csv"))
    s0_gz = solves_gz[0]
    print(f"  Gazebo Solve 0: time={s0_gz['solve_time_ms']}ms, iters={s0_gz['solve_iters']}")
    print(f"  Target: x={s0_gz['tgt_x']}, y={s0_gz['tgt_y']}, z={s0_gz['tgt_z']}")
    print(f"  Circle: phi0={s0_gz['circle_phi0']}, t_abs={s0_gz['circle_t_abs']}")
    print(f"  Abs state: x={s0_gz['abs_x']}, y={s0_gz['abs_y']}, z={s0_gz['abs_z']}")
    print(f"  Rel state: x={s0_gz['rel_x']}, y={s0_gz['rel_y']}, z={s0_gz['rel_z']}")
    
    # Gazebo trajectory
    gz_solve0 = [s for s in solves_gz if s['solve_num'] == '0']
    print(f"\n  Gazebo Solve 0 trajectory ({len(gz_solve0)} nodes):")
    for s in gz_solve0[:5]:
        print(f"    node={s['node']} abs=({s['abs_x']},{s['abs_y']},{s['abs_z']}) tgt=({s['tgt_x']},{s['tgt_y']},{s['tgt_z']})")
    if len(gz_solve0) > 5:
        print(f"    ... and {len(gz_solve0)-5} more nodes")
    
    # Compare commanded states
    cmd_gz = read_csv(os.path.join(base_gz, "commanded_state.csv"))
    act_gz = read_csv(os.path.join(base_gz, "actual_state.csv"))
    print(f"\n  Gazebo commanded states: {len(cmd_gz)}")
    print(f"  Gazebo actual states: {len(act_gz)}")
    
    cg0 = cmd_gz[0]
    ag0 = act_gz[0]
    print(f"  Gazebo cmd initial: x={cg0['x']}, y={cg0['y']}, z={cg0['z']}")
    print(f"  Gazebo act initial: x={ag0['x']}, y={ag0['y']}, z={ag0['z']}")
    print(f"  Gazebo cmd initial vel: vx={cg0['vx']}, vy={cg0['vy']}, vz={cg0['vz']}")
    print(f"  Gazebo act initial vel: vx={ag0['vx']}, vy={ag0['vy']}, vz={ag0['vz']}")
    
    # Check the gazebo commanded trajectory range
    max_x = max(float(c['x']) for c in cmd_gz)
    max_y = max(float(c['y']) for c in cmd_gz)
    max_z = max(float(c['z']) for c in cmd_gz)
    min_x = min(float(c['x']) for c in cmd_gz)
    min_y = min(float(c['y']) for c in cmd_gz)
    min_z = min(float(c['z']) for c in cmd_gz)
    print(f"\n  Gazebo cmd range: x=[{min_x:.3f},{max_x:.3f}] y=[{min_y:.3f},{max_y:.3f}] z=[{min_z:.3f},{max_z:.3f}]")
    
    # Compare the hardware version range
    max_x2 = max(float(c['x']) for c in cmd2)
    max_y2 = max(float(c['y']) for c in cmd2)
    max_z2 = max(float(c['z']) for c in cmd2)
    min_x2 = min(float(c['x']) for c in cmd2)
    min_y2 = min(float(c['y']) for c in cmd2)
    min_z2 = min(float(c['z']) for c in cmd2)
    print(f"  Hardware cmd range: x=[{min_x2:.3f},{max_x2:.3f}] y=[{min_y2:.3f},{max_y2:.3f}] z=[{min_z2:.3f},{max_z2:.3f}]")

print("\n\nDone.")
