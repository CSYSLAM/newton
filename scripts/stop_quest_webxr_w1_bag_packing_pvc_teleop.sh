#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="w1-bag-packing-pvc"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_webxr_w1_bag_packing_pvc"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8777}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-w1-bag-packing-pvc.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-w1-bag-packing-pvc-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-w1-bag-packing-pvc-teleop"
export NEWTON_WEBXR_CLEANUP_PROCESSES="1"
exec "${script_dir}/stop_quest_webxr_teleop.sh" "$@"
