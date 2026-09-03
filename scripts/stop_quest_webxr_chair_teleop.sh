#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="push-chair"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_dexforce_webxr_push_chair"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8766}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-chair.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-chair-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-chair-teleop"
exec "${script_dir}/stop_quest_webxr_teleop.sh" "$@"
