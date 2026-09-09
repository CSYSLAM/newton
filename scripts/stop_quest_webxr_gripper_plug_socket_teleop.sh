#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="gripper-plug-socket"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_dexforce_webxr_gripper_plug_socket"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8772}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-gripper-plug.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-gripper-plug-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-gripper-plug-teleop"
exec "${script_dir}/stop_quest_webxr_teleop.sh" "$@"
