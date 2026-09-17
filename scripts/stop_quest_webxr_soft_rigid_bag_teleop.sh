#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_WEBXR_NAME="soft-rigid-cubes-into-bag"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_webxr_cubes_into_bag"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8768}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-soft-rigid-bag.service"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-soft-rigid-bag-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-soft-rigid-bag-teleop"
exec "${script_dir}/stop_quest_webxr_teleop.sh" "$@"
