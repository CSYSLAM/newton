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
active_run_file="${state_root}/active-run"
unit_name="${NEWTON_WEBXR_UNIT:-newton-quest-webxr.service}"
example_name="${NEWTON_WEBXR_EXAMPLE:-mjvbd_v2_dexforce_webxr_plug_socket}"
port="${NEWTON_WEBXR_PORT:-8765}"
device="${NEWTON_WEBXR_DEVICE:-cuda:0}"
start_script="${NEWTON_WEBXR_START_SCRIPT:-${script_dir}/start_quest_webxr_teleop.sh}"
phase_delay="${NEWTON_WEBXR_PHASE_DELAY_SECONDS:-2}"
park_delay="${NEWTON_WEBXR_PARK_DELAY_SECONDS:-3}"
restart_delay="${NEWTON_WEBXR_RESTART_DELAY_SECONDS:-5}"
mode_timeout="${NEWTON_WEBXR_MODE_TIMEOUT_SECONDS:-10}"
shutdown_timeout="${NEWTON_WEBXR_SHUTDOWN_TIMEOUT_SECONDS:-120}"
guard_unit="${NEWTON_WEBXR_GUARD_UNIT:-newton-quest-webxr-cuda-guard.service}"
guard_runtime_root="${XDG_RUNTIME_DIR:-/tmp}/newton-webxr-cuda-guard-${UID}"
guard_state_root="${XDG_STATE_HOME:-${HOME}/.local/state}/newton-webxr-cuda-guard"
guard_ready_file="${guard_runtime_root}/ready"
guard_log_file="${guard_state_root}/guard.log"
guard_script="${script_dir}/quest_webxr_cuda_guard.py"
guard_interval="${NEWTON_WEBXR_GUARD_INTERVAL_SECONDS:-0.05}"

is_nonnegative_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

for delay_name in phase_delay park_delay restart_delay; do
  delay_value="${!delay_name}"
  if ! is_nonnegative_number "${delay_value}"; then
    echo "错误：${delay_name} 必须是非负秒数，实际为 ${delay_value}。" >&2
    exit 1
  fi
done
if [[ ! "${mode_timeout}" =~ ^[1-9][0-9]*$ || ! "${shutdown_timeout}" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：模式与关闭超时必须是正整数秒。" >&2
  exit 1
fi

command -v curl >/dev/null || { echo "错误：未找到 curl。" >&2; exit 1; }
command -v flock >/dev/null || { echo "错误：未找到 flock。" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "错误：未找到 systemctl。" >&2; exit 1; }
[[ -x "${start_script}" ]] || { echo "错误：启动脚本不可执行：${start_script}" >&2; exit 1; }

mkdir -p "${runtime_root}" "${state_root}" "${guard_runtime_root}" "${guard_state_root}"
exec 9>"${runtime_root}/reload.lock"
if ! flock -n 9; then
  echo "错误：${teleop_name} 已有另一个分阶段重载正在执行。" >&2
  exit 1
fi

unit_state() {
  systemctl --user show --property=ActiveState --value "${unit_name}" 2>/dev/null || true
}

unit_is_alive() {
  local state
  state="$(unit_state)"
  [[ "${state}" == "active" || "${state}" == "activating" || "${state}" == "deactivating" ]]
}

unit_pid() {
  systemctl --user show --property=MainPID --value "${unit_name}" 2>/dev/null || true
}

unit_load_state() {
  systemctl --user show --property=LoadState --value "${unit_name}" 2>/dev/null || true
}

is_demo_pid() {
  local candidate="$1"
  [[ "${candidate}" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/${candidate}/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/${candidate}/cmdline" \
    | grep -Fq "newton.examples ${example_name}"
}

health_snapshot() {
  curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/healthz" 2>/dev/null || true
}

wait_for_mode() {
  local expected_simulation="$1"
  local attempts=$((mode_timeout * 10))
  local health
  for ((attempt = 0; attempt < attempts; attempt += 1)); do
    health="$(health_snapshot)"
    if grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*false' <<< "${health}" \
      && grep -Eq "\"simulationActive\"[[:space:]]*:[[:space:]]*${expected_simulation}" <<< "${health}"; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

ensure_cuda_guard() {
  [[ "${device}" == cuda:* ]] || return 0
  command -v nvidia-smi >/dev/null || { echo "错误：未找到 nvidia-smi；旧进程保持不变。" >&2; return 1; }
  command -v systemd-run >/dev/null || { echo "错误：未找到 systemd-run；旧进程保持不变。" >&2; return 1; }
  local uv_command
  uv_command="$(command -v uv)" || { echo "错误：未找到 uv；旧进程保持不变。" >&2; return 1; }
  if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >/dev/null 2>&1; then
    echo "错误：NVIDIA 驱动预检失败；旧进程保持不变。" >&2
    return 1
  fi

  local guard_state
  guard_state="$(systemctl --user show --property=ActiveState --value "${guard_unit}" 2>/dev/null || true)"
  if [[ "${guard_state}" != "active" && "${guard_state}" != "activating" ]]; then
    rm -f "${guard_ready_file}"
    local uv_cache_dir="${cache_root}/uv-cache"
    mkdir -p "${uv_cache_dir}"
    systemd-run \
      --user \
      --unit="${guard_unit%.service}" \
      --collect \
      --property=Type=exec \
      --property=KillSignal=SIGTERM \
      --property=TimeoutStopSec=30s \
      --property=SendSIGKILL=no \
      --property="StandardOutput=append:${guard_log_file}" \
      --property="StandardError=append:${guard_log_file}" \
      --working-directory="${repo_root}" \
      --setenv="UV_CACHE_DIR=${uv_cache_dir}" \
      -- \
      "${uv_command}" run --extra examples python -u "${guard_script}" \
      --device "${device}" \
      --ready-file "${guard_ready_file}" \
      --interval-seconds "${guard_interval}" >/dev/null
  fi

  for ((attempt = 0; attempt < 600; attempt += 1)); do
    guard_state="$(systemctl --user show --property=ActiveState --value "${guard_unit}" 2>/dev/null || true)"
    if [[ "${guard_state}" == "active" && -s "${guard_ready_file}" ]] \
      && grep -Fxq "device=${device}" "${guard_ready_file}"; then
      local guard_pid
      guard_pid="$(sed -n 's/^pid=//p' "${guard_ready_file}")"
      if [[ "${guard_pid}" =~ ^[0-9]+$ && -r "/proc/${guard_pid}/cmdline" ]] \
        && tr '\0' ' ' < "/proc/${guard_pid}/cmdline" | grep -Fq "${guard_script}"; then
        return 0
      fi
      echo "错误：CUDA guard ready 文件与运行进程不匹配；旧场景保持不变。" >&2
      return 1
    fi
    if [[ "${guard_state}" != "active" && "${guard_state}" != "activating" ]]; then
      break
    fi
    sleep 0.1
  done
  echo "错误：CUDA reload guard 未就绪；旧场景没有被停泊或关闭。日志：${guard_log_file}" >&2
  return 1
}

demo_pid=""
health=""
if unit_is_alive; then
  demo_pid="$(unit_pid)"
  if ! is_demo_pid "${demo_pid}"; then
    echo "错误：${unit_name} 的 PID ${demo_pid:-unknown} 不是 ${teleop_name}；没有发送控制请求。" >&2
    exit 1
  fi
  health="$(health_snapshot)"
  if [[ -z "${health}" ]]; then
    echo "错误：无法读取旧场景健康状态；没有停泊或关闭 PID ${demo_pid}。" >&2
    exit 1
  fi
fi

echo "[1/8] 检查并建立 CUDA 连续负载保护。"
ensure_cuda_guard

if [[ -n "${demo_pid}" ]]; then
  echo "[2/8] 禁用手柄输入并暂停录制，保持 CUDA 物理帧运行。"
  if ! grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*false' <<< "${health}"; then
    curl --silent --fail --max-time 2 -X POST \
      "http://127.0.0.1:${port}/control/standby" >/dev/null
    if ! wait_for_mode true; then
      echo "错误：旧场景未确认进入 standby；保持当前进程，不继续。" >&2
      exit 1
    fi
  fi

  echo "[3/8] 请求 Quest 退出沉浸模式并等待 ${phase_delay} 秒。"
  curl --silent --fail --max-time 2 -X POST \
    "http://127.0.0.1:${port}/control/exit-immersive" >/dev/null
  sleep "${phase_delay}"

  health="$(health_snapshot)"
  echo "[4/8] 在 CUDA guard 持续提交工作的同时停泊旧场景。"
  if ! grep -Eq '"simulationActive"[[:space:]]*:[[:space:]]*false' <<< "${health}"; then
    curl --silent --fail --max-time 2 -X POST \
      "http://127.0.0.1:${port}/control/park" >/dev/null
    if ! wait_for_mode false; then
      echo "错误：旧场景未确认停泊；CUDA guard 保持运行，没有关闭进程。" >&2
      exit 1
    fi
  fi
  sleep "${park_delay}"

  echo "[5/8] 发送一次协作式 shutdown；不使用 SIGKILL 或强制终止。"
  curl --silent --fail --max-time 2 -X POST \
    "http://127.0.0.1:${port}/control/shutdown" >/dev/null

  echo "[6/8] 等待旧 systemd 单元和 PID ${demo_pid} 完全退出。"
  stopped=false
  for ((attempt = 0; attempt < shutdown_timeout * 10; attempt += 1)); do
    if ! unit_is_alive && ! is_demo_pid "${demo_pid}"; then
      stopped=true
      break
    fi
    sleep 0.1
  done
  if [[ "${stopped}" != true ]]; then
    echo "错误：旧进程在 ${shutdown_timeout} 秒内没有退出；CUDA guard 保持运行，未发送额外信号。" >&2
    echo "不要运行 kill -9。请查看 ${state_root}/latest.log。" >&2
    exit 1
  fi
  released=false
  for ((attempt = 0; attempt < 100; attempt += 1)); do
    load_state="$(unit_load_state)"
    if [[ -z "${load_state}" || "${load_state}" == "not-found" ]]; then
      released=true
      break
    fi
    sleep 0.1
  done
  if [[ "${released}" != true ]]; then
    echo "错误：旧进程已退出，但 transient systemd 单元尚未释放；CUDA guard 保持运行。" >&2
    echo "请稍后重新执行本脚本，不要强制删除单元。" >&2
    exit 1
  fi
else
  echo "[2/8–6/8] 未发现旧 ${teleop_name} systemd 单元，无需终止。"
fi

rm -f "${pid_file}" "${active_run_file}"
echo "[7/8] 旧进程已退出；CUDA guard 保持设备活跃，等待 ${restart_delay} 秒后加载新代码。"
sleep "${restart_delay}"

echo "[8/8] 启动更新后的 ${teleop_name} 场景。"
if ! "${start_script}" "$@"; then
  echo "错误：新场景启动失败。CUDA guard 仍在运行，请修复错误后重新执行本重载脚本。" >&2
  exit 1
fi

echo "分阶段重载完成。CUDA guard 将在本次开发会话中继续运行，以避免最后一个 CUDA 上下文退出。"
echo "Guard 日志：${guard_log_file}"
