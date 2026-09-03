#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="bimanual-nut-bolt"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_webxr_bimanual_nut_bolt"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8770}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-nut-bolt.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-nut-bolt-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-nut-bolt-teleop"
exec "${script_dir}/stop_quest_webxr_teleop.sh" "$@"
