#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="w1-bag-packing"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_webxr_w1_bag_packing"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8773}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-w1-bag-packing.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-w1-bag-packing-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-w1-bag-packing-teleop"
export NEWTON_WEBXR_START_SCRIPT="${script_dir}/start_quest_webxr_w1_bag_packing_teleop.sh"
exec "${script_dir}/reload_quest_webxr_teleop.sh" "$@"
