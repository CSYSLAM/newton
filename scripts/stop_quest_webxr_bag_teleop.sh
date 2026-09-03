#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="plastic-inflatable-bag"
export NEWTON_WEBXR_EXAMPLE="vbd_mjvbd_v2_dexforce_webxr_plastic_inflatable_bag_pick_release_final00"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8767}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-bag.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-bag-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-bag-teleop"
exec "${script_dir}/stop_quest_webxr_teleop.sh" "$@"
