#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="soft-rigid-cubes-into-bag"
export NEWTON_WEBXR_EXAMPLE="vbd_mjvbd_v2_dexforce_webxr_soft_then_rigid_cube_into_bag_final00"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8768}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-soft-rigid-bag.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-soft-rigid-bag-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-soft-rigid-bag-teleop"
export NEWTON_WEBXR_START_SCRIPT="${script_dir}/start_quest_webxr_soft_rigid_bag_teleop.sh"
exec "${script_dir}/reload_quest_webxr_teleop.sh" "$@"
