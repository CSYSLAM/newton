#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="push-chair"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_dexforce_webxr_push_chair"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8766}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-chair.service"
export NEWTON_WEBXR_PEERS="newton-quest-webxr.service:8765 newton-quest-webxr-bag.service:8767 newton-quest-webxr-soft-rigid-bag.service:8768 newton-quest-webxr-tshirt.service:8769 newton-quest-webxr-nut-bolt.service:8770"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-chair-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-chair-teleop"
export NEWTON_WEBXR_SDF_CACHE=0
exec "${script_dir}/start_quest_webxr_teleop.sh" "$@"
