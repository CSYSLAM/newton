#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="bimanual-fold-tshirt"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_webxr_tshirt_fold"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8769}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-tshirt.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-tshirt-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-tshirt-teleop"
exec "${script_dir}/stop_quest_webxr_teleop.sh" "$@"
