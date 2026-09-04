#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
export NEWTON_WEBXR_NAME="bimanual-nut-bolt"
export NEWTON_WEBXR_EXAMPLE="mjvbd_v2_webxr_bimanual_nut_bolt"
export NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8770}"
export NEWTON_WEBXR_UNIT="newton-quest-webxr-nut-bolt.service"
export NEWTON_WEBXR_PEERS="newton-quest-webxr.service:8765 newton-quest-webxr-chair.service:8766 newton-quest-webxr-bag.service:8767 newton-quest-webxr-soft-rigid-bag.service:8768 newton-quest-webxr-tshirt.service:8769 newton-quest-webxr-nonwoven-bag.service:8771"
export NEWTON_WEBXR_RUNTIME_NAME="newton-webxr-nut-bolt-teleop"
export NEWTON_WEBXR_STATE_NAME="newton-webxr-nut-bolt-teleop"
export NEWTON_WEBXR_SDF_CACHE=1
export NEWTON_WEBXR_RELOAD_COMMAND="./scripts/reload_quest_webxr_nut_bolt_teleop.sh"
export NEWTON_WEBXR_RELOAD_SOURCES="${repo_root}/newton/examples/mjvbdv2/_webxr_teleop.py:${repo_root}/newton/examples/mjvbdv2/example_mjvbd_v2_bimanual_nut_bolt.py:${repo_root}/newton/examples/mjvbdv2/example_mjvbd_v2_webxr_bimanual_nut_bolt.py"
exec "${script_dir}/start_quest_webxr_teleop.sh" "$@"
