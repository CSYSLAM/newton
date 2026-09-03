#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
teleop_name="${NEWTON_WEBXR_NAME:-plug-socket}"
runtime_name="${NEWTON_WEBXR_RUNTIME_NAME:-newton-webxr-teleop}"
state_name="${NEWTON_WEBXR_STATE_NAME:-newton-webxr-teleop}"
runtime_root="${XDG_RUNTIME_DIR:-/tmp}/${runtime_name}-${UID}"
state_root="${XDG_STATE_HOME:-${HOME}/.local/state}/${state_name}"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/newton-webxr-teleop"
pid_file="${runtime_root}/demo.pid"
log_path_file="${runtime_root}/demo.log.path"
active_run_file="${state_root}/active-run"
uv_cache_dir="${cache_root}/uv-cache"
sdf_cache_dir="${cache_root}/sdf"
unit_name="${NEWTON_WEBXR_UNIT:-newton-quest-webxr.service}"
example_name="${NEWTON_WEBXR_EXAMPLE:-mjvbd_v2_dexforce_webxr_plug_socket}"
sdf_cache_enabled="${NEWTON_WEBXR_SDF_CACHE:-1}"
read -r -a peer_specs <<< "${NEWTON_WEBXR_PEERS:-newton-quest-webxr-chair.service:8766 newton-quest-webxr-bag.service:8767 newton-quest-webxr-soft-rigid-bag.service:8768 newton-quest-webxr-tshirt.service:8769 newton-quest-webxr-nut-bolt.service:8770}"
host="${NEWTON_WEBXR_HOST:-127.0.0.1}"
port="${NEWTON_WEBXR_PORT:-8765}"
device="${NEWTON_WEBXR_DEVICE:-cuda:0}"
viewer="${NEWTON_WEBXR_VIEWER:-null}"
graph_capture="${NEWTON_WEBXR_GRAPH_CAPTURE:-0}"
browser_launch_id="$(date '+%s%N')"
quest_url_base="http://127.0.0.1:${port}/"
quest_url="${quest_url_base}?launch=${browser_launch_id}"
browser_application_id="${NEWTON_WEBXR_BROWSER_APPLICATION_ID:-org.newton.webxr.teleop}"
reload_command="${NEWTON_WEBXR_RELOAD_COMMAND:-}"
if [[ -v NEWTON_WEBXR_RELOAD_SOURCES ]]; then
  IFS=: read -r -a reload_sources <<< "${NEWTON_WEBXR_RELOAD_SOURCES}"
else
  reload_sources=(
    "${repo_root}/newton/examples/mjvbdv2/_webxr_teleop.py"
    "${repo_root}/newton/examples/mjvbdv2/example_${example_name}.py"
  )
fi

command -v adb >/dev/null || { echo "错误：未找到 adb。" >&2; exit 1; }
command -v curl >/dev/null || { echo "错误：未找到 curl。" >&2; exit 1; }
uv_command="$(command -v uv)" || { echo "错误：未找到 uv。" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "错误：未找到 systemctl。" >&2; exit 1; }
command -v systemd-run >/dev/null || { echo "错误：未找到 systemd-run。" >&2; exit 1; }

mkdir -p "${runtime_root}" "${state_root}" "${uv_cache_dir}" "${sdf_cache_dir}"

is_demo_pid() {
  local candidate="$1"
  [[ "${candidate}" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/${candidate}/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/${candidate}/cmdline" \
    | grep -Fq "newton.examples ${example_name}"
}

unit_state() {
  systemctl --user show --property=ActiveState --value "${unit_name}" 2>/dev/null || true
}

unit_is_alive() {
  local state
  state="$(unit_state)"
  [[ "${state}" == "active" || "${state}" == "activating" ]]
}

unit_pid() {
  systemctl --user show --property=MainPID --value "${unit_name}" 2>/dev/null || true
}

if [[ -s "${active_run_file}" ]] && ! unit_is_alive; then
  echo "错误：检测到上一次 Newton Quest 遥操未正常结束；为避免连续硬锁，已拒绝自动重启。" >&2
  echo "上一次运行信息：" >&2
  sed 's/^/  /' "${active_run_file}" >&2
  echo "请先运行当前场景对应的关闭脚本，确认已安全暂停并清除异常运行标记。" >&2
  exit 1
fi

if unit_is_alive && ! is_demo_pid "$(unit_pid)"; then
  echo "错误：另一个 Quest 遥操场景仍在运行或停泊，拒绝同时创建第二个 CUDA 场景。" >&2
  sed 's/^/  /' "${active_run_file}" >&2 || true
  exit 1
fi

active_peers=()
standby_peers=()
running_peer_count=0
for peer_spec in "${peer_specs[@]}"; do
  peer_unit_name="${peer_spec%%:*}"
  peer_port="${peer_spec##*:}"
  if [[ "${peer_unit_name}" == "${peer_spec}" || ! "${peer_port}" =~ ^[0-9]+$ ]]; then
    echo "错误：NEWTON_WEBXR_PEERS 条目无效：${peer_spec}" >&2
    exit 1
  fi
  peer_state="$(systemctl --user show --property=ActiveState --value "${peer_unit_name}" 2>/dev/null || true)"
  if [[ "${peer_state}" == "active" || "${peer_state}" == "activating" || "${peer_state}" == "deactivating" ]]; then
    peer_health="$(curl --silent --fail --max-time 2 "http://127.0.0.1:${peer_port}/healthz" 2>/dev/null || true)"
    if grep -Eq '"simulationActive"[[:space:]]*:[[:space:]]*true' <<< "${peer_health}"; then
      ((running_peer_count += 1))
      if grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*true' <<< "${peer_health}"; then
        active_peers+=("${peer_unit_name}:${peer_port}")
      elif grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*false' <<< "${peer_health}"; then
        standby_peers+=("${peer_unit_name}:${peer_port}")
      else
        echo "错误：另一个 Quest 场景的控制状态不明；没有改变其 CUDA 状态。" >&2
        exit 1
      fi
    elif grep -Eq '"simulationActive"[[:space:]]*:[[:space:]]*false' <<< "${peer_health}" \
      && grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*false' <<< "${peer_health}"; then
      :
    else
      echo "错误：另一个 Quest 场景不支持安全接棒协议；没有改变其 CUDA 状态。" >&2
      echo "请确认该场景使用当前代码启动，并重新运行对应关闭脚本进入安全待机。" >&2
      exit 1
    fi
  fi
done

if ((running_peer_count > 1)); then
  echo "错误：检测到多个仍提交物理帧的 Quest 场景；为避免增加 GPU 负载，没有自动切换。" >&2
  exit 1
fi

for peer_spec in "${active_peers[@]}"; do
  peer_unit_name="${peer_spec%%:*}"
  peer_port="${peer_spec##*:}"
  if ! curl --silent --fail --max-time 2 -X POST \
    "http://127.0.0.1:${peer_port}/control/standby" >/dev/null 2>&1; then
    echo "错误：无法让当前场景 ${peer_unit_name} 进入安全待机；没有启动新场景。" >&2
    exit 1
  fi
  peer_standing_by=false
  for ((attempt = 0; attempt < 50; attempt += 1)); do
    peer_health="$(curl --silent --fail --max-time 1 "http://127.0.0.1:${peer_port}/healthz" 2>/dev/null || true)"
    if grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*false' <<< "${peer_health}" \
      && grep -Eq '"simulationActive"[[:space:]]*:[[:space:]]*true' <<< "${peer_health}"; then
      peer_standing_by=true
      break
    fi
    sleep 0.1
  done
  if [[ "${peer_standing_by}" != true ]]; then
    echo "错误：当前场景 ${peer_unit_name} 未确认安全待机；没有启动新场景。" >&2
    exit 1
  fi
  curl --silent --fail --max-time 2 -X POST \
    "http://127.0.0.1:${peer_port}/control/exit-immersive" >/dev/null 2>&1 || true
  standby_peers+=("${peer_spec}")
done

gpu_summary="not-requested"
if [[ "${device}" == cuda:* ]]; then
  command -v nvidia-smi >/dev/null || {
    echo "错误：CUDA 模式需要 nvidia-smi，但当前系统找不到该命令。" >&2
    exit 1
  }
  if ! gpu_summary="$(
    nvidia-smi \
      --query-gpu=name,driver_version,pstate,power.limit,temperature.gpu,memory.used \
      --format=csv,noheader 2>&1
  )"; then
    echo "错误：NVIDIA 驱动预检失败，拒绝启动 CUDA 遥操：${gpu_summary}" >&2
    exit 1
  fi
fi

adb start-server >/dev/null
adb_state="$(adb get-state 2>/dev/null || true)"
if [[ "${adb_state}" != "device" ]]; then
  echo "错误：Quest 未授权。请戴上头显允许 USB 调试，然后重新运行本脚本。" >&2
  exit 1
fi
adb reverse "tcp:${port}" "tcp:${port}" >/dev/null

viewer_args=(--viewer "${viewer}")
if [[ "${viewer}" == "null" ]]; then
  # ViewerNull is intentionally finite; use a multi-year frame budget and
  # terminate through the cooperative HTTP shutdown path instead.
  viewer_args+=(--num-frames 2147483647)
else
  echo "警告：NEWTON_WEBXR_VIEWER=${viewer} 会启用本机图形 Viewer，可能触发当前机器的显示驱动卡顿。" >&2
fi

graph_args=(--no-graph-capture)
if [[ "${graph_capture}" == "1" ]]; then
  graph_args=(--graph-capture)
  echo "警告：已显式启用 CUDA graph capture；当前 595.71.05 驱动曾在启动/退出阶段硬锁。" >&2
fi

example_args=()
sdf_cache_label="disabled"
if [[ "${sdf_cache_enabled}" == "1" ]]; then
  example_args+=(--sdf-cache-dir "${sdf_cache_dir}")
  sdf_cache_label="${sdf_cache_dir}"
fi

if ! unit_is_alive; then
  launch_id="$(date '+%Y%m%d-%H%M%S')-$$"
  log_file="${state_root}/${teleop_name}-${launch_id}.log"
  ln -sfn "$(basename -- "${log_file}")" "${state_root}/latest.log"
  printf '%s\n' "${log_file}" > "${log_path_file}"
  rm -f "${pid_file}"
  {
    printf 'launchTime=%s\n' "$(date --iso-8601=seconds)"
    printf 'kernel=%s\n' "$(uname -srvo)"
    printf 'device=%s\n' "${device}"
    printf 'viewer=%s\n' "${viewer}"
    printf 'example=%s\n' "${example_name}"
    printf 'graphCapture=%s\n' "${graph_capture}"
    printf 'sdfCache=%s\n' "${sdf_cache_label}"
    printf 'gpu=%s\n' "${gpu_summary}"
  } > "${log_file}"
  {
    printf 'launchTime=%s\n' "$(date --iso-8601=seconds)"
    printf 'log=%s\n' "${log_file}"
    printf 'device=%s\n' "${device}"
    printf 'example=%s\n' "${example_name}"
    printf 'port=%s\n' "${port}"
    printf 'gpu=%s\n' "${gpu_summary}"
  } > "${active_run_file}"
  systemd_run_args=(
    --user
    --unit="${unit_name%.service}"
    --collect
    --property=Type=exec
    --property=KillSignal=SIGTERM
    --property=TimeoutStopSec=30s
    --property=SendSIGKILL=no
    --property="StandardOutput=append:${log_file}"
    --property="StandardError=append:${log_file}"
    --working-directory="${repo_root}"
    --setenv="UV_CACHE_DIR=${uv_cache_dir}"
  )
  [[ -n "${DISPLAY:-}" ]] && systemd_run_args+=(--setenv="DISPLAY=${DISPLAY}")
  [[ -n "${XAUTHORITY:-}" ]] && systemd_run_args+=(--setenv="XAUTHORITY=${XAUTHORITY}")
  [[ -n "${WAYLAND_DISPLAY:-}" ]] && systemd_run_args+=(--setenv="WAYLAND_DISPLAY=${WAYLAND_DISPLAY}")

  systemd-run "${systemd_run_args[@]}" -- \
    "${uv_command}" run --extra examples python -u -m newton.examples \
    "${example_name}" \
    --device "${device}" \
    "${viewer_args[@]}" \
    "${graph_args[@]}" \
    --webxr-host "${host}" \
    --webxr-port "${port}" \
    "${example_args[@]}" \
    --render-fps 60 \
    "$@" >/dev/null
fi

if [[ ! -v log_file ]]; then
  if [[ -r "${log_path_file}" ]]; then
    read -r log_file < "${log_path_file}"
  else
    log_file="${state_root}/latest.log"
  fi
fi

if unit_is_alive; then
  curl --silent --fail --max-time 2 -X POST \
    "http://127.0.0.1:${port}/control/resume" >/dev/null 2>&1 || true
fi

ready=false
for ((attempt = 0; attempt < 600; attempt += 1)); do
  if curl --silent --fail --max-time 1 "http://127.0.0.1:${port}/healthz" \
    | grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*true'; then
    ready=true
    break
  fi
  if ! unit_is_alive; then
    break
  fi
  sleep 0.2
done

demo_pid="$(unit_pid)"
if [[ "${ready}" != true ]] || ! is_demo_pid "${demo_pid}"; then
  curl --silent --fail --max-time 2 -X POST \
    "http://127.0.0.1:${port}/control/standby" >/dev/null 2>&1 || true
  if ! unit_is_alive; then
    rm -f "${pid_file}"
    rm -f "${active_run_file}"
  fi
  echo "错误：Newton Quest 遥操未就绪；为避免 CUDA 销毁硬锁，没有强制结束仍存活的进程。最近日志：" >&2
  tail -n 40 "${log_file}" >&2 || true
  exit 1
fi
printf '%s\n' "${demo_pid}" > "${pid_file}"
printf 'pid=%s\n' "${demo_pid}" >> "${active_run_file}"

stale_sources=()
for source_path in "${reload_sources[@]}"; do
  if [[ -f "${source_path}" && "${source_path}" -nt "/proc/${demo_pid}" ]]; then
    stale_sources+=("${source_path}")
  fi
done
if ((${#stale_sources[@]} > 0)); then
  echo "警告：当前 Newton PID ${demo_pid} 早于以下 Python 源码，当前进程未加载这些更新：" >&2
  printf '  %s\n' "${stale_sources[@]}" >&2
  echo "安全 stop/start 只会恢复同一 PID。当前驱动环境下不要强制结束 CUDA 进程；" >&2
  if [[ -n "${reload_command}" ]]; then
    echo "要加载新代码，请运行分阶段重载：${reload_command}" >&2
  else
    echo "当前场景没有分阶段重载包装脚本，请正常重启电脑后再启动。" >&2
  fi
fi

handoff_ok=true
for peer_spec in "${standby_peers[@]}"; do
  peer_unit_name="${peer_spec%%:*}"
  peer_port="${peer_spec##*:}"
  if ! curl --silent --fail --max-time 2 -X POST \
    "http://127.0.0.1:${peer_port}/control/park" >/dev/null 2>&1; then
    echo "错误：新场景已就绪，但无法请求旧场景 ${peer_unit_name} 停泊；两个 CUDA 进程暂时保持活动。" >&2
    handoff_ok=false
    continue
  fi
  peer_parked=false
  for ((attempt = 0; attempt < 50; attempt += 1)); do
    peer_health="$(curl --silent --fail --max-time 1 "http://127.0.0.1:${peer_port}/healthz" 2>/dev/null || true)"
    if grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*false' <<< "${peer_health}" \
      && grep -Eq '"simulationActive"[[:space:]]*:[[:space:]]*false' <<< "${peer_health}"; then
      peer_parked=true
      break
    fi
    sleep 0.1
  done
  if [[ "${peer_parked}" != true ]]; then
    echo "错误：新场景已就绪，但旧场景 ${peer_unit_name} 未确认停泊；没有终止任何进程。" >&2
    handoff_ok=false
  fi
done

if [[ "${handoff_ok}" != true ]]; then
  echo "请再次运行当前启动脚本重试安全接棒；不要运行终止命令。" >&2
  exit 1
fi

# Android browsers associate externally opened tabs with this stable application
# ID. Reusing the same ID for every Newton scene makes subsequent launches
# navigate the existing tab instead of allocating another renderer process.
# A unique query makes Android refresh that tab so it cannot retain stale JS.
adb shell am start -a android.intent.action.VIEW -d "${quest_url}" \
  --es com.android.browser.application_id "${browser_application_id}" \
  --ez create_new_tab false >/dev/null
printf 'Newton Quest %s 遥操已启动或恢复（PID %s）。\nQuest URL: %s\n日志: %s\n' \
  "${teleop_name}" "${demo_pid}" "${quest_url_base}" "${log_file}"
if ((${#standby_peers[@]} > 0)); then
  echo "新场景已稳定提交 CUDA 帧，旧场景已自动安全停泊。"
fi
printf '本机 Viewer: %s（默认 null，双眼画面由 Quest WebXR 渲染）\n' "${viewer}"
echo "请在 Quest Browser 中点击“进入沉浸式遥操”。"
