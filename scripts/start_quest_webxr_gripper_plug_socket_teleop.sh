#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
export NEWTON_WEBXR_NAME="gripper-plug-socket"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_dexforce_webxr_gripper_plug_socket"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8772}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-gripper-plug.service"
export NEWTON_WEBXR_PEERS="newton-quest-webxr.service:8765 newton-quest-webxr-chair.service:8766 newton-quest-webxr-bag.service:8767 newton-quest-webxr-soft-rigid-bag.service:8768 newton-quest-webxr-tshirt.service:8769 newton-quest-webxr-nut-bolt.service:8770 newton-quest-webxr-nonwoven-bag.service:8771"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-gripper-plug-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-gripper-plug-teleop"
export NEWTON_WEBXR_RELOAD_COMMAND="./scripts/reload_quest_webxr_gripper_plug_socket_teleop.sh"
export NEWTON_WEBXR_RELOAD_SOURCES="${repo_root}/newton/examples/mjvbdv2/_webxr_teleop.py:${repo_root}/newton/examples/mjvbdv2/_webxr_w1_head.py:${repo_root}/newton/examples/mjvbdv2/_webxr_w1_single_hand.py:${repo_root}/newton/examples/mjvbdv2/_webxr_parallel_gripper.py:${repo_root}/newton/examples/mjvbdv2/example_mjvbd_v2_dexforce_realtime_plug_socket.py:${repo_root}/newton/examples/mjvbdv2/example_mjvbd_v2_dexforce_webxr_plug_socket.py:${repo_root}/newton/examples/mjvbdv2/example_mjvbd_v2_dexforce_webxr_gripper_plug_socket.py"
if [[ ! -f "${repo_root}/assets/w1-pikka-gripper/DexforceW1V021_pikka_gripper_simple_visual_collision.urdf" ]]; then
  echo "请先下载模型：uv run scripts/download_quest_w1_gripper.py" >&2
  exit 1
fi
exec "${script_dir}/start_quest_webxr_teleop.sh" "$@"
