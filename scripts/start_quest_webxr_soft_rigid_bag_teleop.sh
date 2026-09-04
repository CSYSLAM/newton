#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
export NEWTON_WEBXR_NAME="soft-rigid-cubes-into-bag"
export NEWTON_WEBXR_EXAMPLE="vbd_mjvbd_v2_dexforce_webxr_soft_then_rigid_cube_into_bag_final00"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8768}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-soft-rigid-bag.service"
export NEWTON_WEBXR_PEERS="newton-quest-webxr.service:8765 newton-quest-webxr-chair.service:8766 newton-quest-webxr-bag.service:8767 newton-quest-webxr-tshirt.service:8769 newton-quest-webxr-nut-bolt.service:8770 newton-quest-webxr-nonwoven-bag.service:8771"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-soft-rigid-bag-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-soft-rigid-bag-teleop"
export NEWTON_WEBXR_SDF_CACHE=0
export NEWTON_WEBXR_RELOAD_COMMAND="./scripts/reload_quest_webxr_soft_rigid_bag_teleop.sh"
export NEWTON_WEBXR_RELOAD_SOURCES="${repo_root}/newton/examples/mjvbdv2/_webxr_teleop.py:${repo_root}/newton/examples/mjvbdv2/_webxr_w1_head.py:${repo_root}/newton/examples/mjvbdv2/example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_final00.py:${repo_root}/newton/examples/mjvbdv2/example_vbd_mjvbd_v2_dexforce_webxr_soft_then_rigid_cube_into_bag_final00.py"
exec "${script_dir}/start_quest_webxr_teleop.sh" "$@"
