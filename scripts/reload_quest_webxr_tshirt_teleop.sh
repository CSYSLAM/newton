#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="bimanual-fold-tshirt"
export NEWTON_WEBXR_EXAMPLE="cloth_mjvbd_v2_dexforce_webxr_bimanual_fold_tshirt_waic_house_final00"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8769}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-tshirt.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-tshirt-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-tshirt-teleop"
export NEWTON_WEBXR_START_SCRIPT="${script_dir}/start_quest_webxr_tshirt_teleop.sh"
exec "${script_dir}/reload_quest_webxr_teleop.sh" "$@"
