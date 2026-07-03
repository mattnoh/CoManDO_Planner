#!/usr/bin/env bash
# Apply an OCP profile to a running comando_planner node (ROS 1).
#
# Usage:
#   ./apply_profile.sh [param-namespace] [profile.yaml]
#
# Defaults: namespace /comando_planner (the node's private ~ namespace),
# profile config/profile_hover_sitl.yaml. Loads the profile's params and
# bumps command_seq to trigger the command.
set -euo pipefail

NS="${1:-/comando_planner}"
PROFILE="${2:-$(dirname "$0")/../config/profile_hover_sitl.yaml}"

if [[ ! -f "$PROFILE" ]]; then
  echo "Profile file not found: $PROFILE" >&2
  exit 1
fi

CUR=$(rosparam get "$NS/command_seq" 2>/dev/null || echo 0)
NEXT=$((CUR + 1))

rosparam load "$PROFILE" "$NS"
rosparam set "$NS/command_seq" "$NEXT"

echo "Applied profile: $PROFILE"
echo "Triggered command_seq: $NEXT"
