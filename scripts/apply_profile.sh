#!/usr/bin/env bash
set -euo pipefail

NODE="${1:-/comando_planner}"
PROFILE="${2:-$(dirname "$0")/../config/profile_stateswitch.yaml}"

if [[ ! -f "$PROFILE" ]]; then
  echo "Profile file not found: $PROFILE" >&2
  exit 1
fi

CUR=$(ros2 param get "$NODE" command_seq | awk '{print $NF}')
NEXT=$((CUR + 1))

ros2 param load "$NODE" "$PROFILE"
ros2 param set "$NODE" command_seq "$NEXT"

echo "Applied profile: $PROFILE"
echo "Triggered command_seq: $NEXT"
