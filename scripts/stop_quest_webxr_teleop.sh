#!/usr/bin/env bash

set -euo pipefail

runtime_name="${NEWTON_WEBXR_RUNTIME_NAME:-newton-webxr-teleop}"
state_name="${NEWTON_WEBXR_STATE_NAME:-newton-webxr-teleop}"
runtime_root="${XDG_RUNTIME_DIR:-/tmp}/${runtime_name}-${UID}"
state_root="${XDG_STATE_HOME:-${HOME}/.local/state}/${state_name}"
pid_file="${runtime_root}/demo.pid"
active_run_file="${state_root}/active-run"
port="${NEWTON_WEBXR_PORT:-8765}"
unit_name="${NEWTON_WEBXR_UNIT:-newton-quest-webxr.service}"
teleop_name="${NEWTON_WEBXR_NAME:-plug-socket}"
example_name="${NEWTON_WEBXR_EXAMPLE:-mjvbd_v2_dexforce_webxr_plug_socket}"
terminate_process="${NEWTON_WEBXR_TERMINATE:-0}"

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
  [[ "${state}" == "active" || "${state}" == "activating" || "${state}" == "deactivating" ]]
}

stopped=false
standing_by=false
parked=false
managed=false
if unit_is_alive; then
  managed=true
  demo_pid="$(systemctl --user show --property=MainPID --value "${unit_name}" 2>/dev/null || true)"
  if ! is_demo_pid "${demo_pid}"; then
    echo "错误：运行中的服务不是 ${teleop_name} 场景；没有向它发送控制请求。" >&2
    exit 1
  fi
  if [[ "${terminate_process}" == "1" ]]; then
    echo "警告：正在显式销毁 CUDA 进程；595.71.05 驱动上该路径曾触发整机硬锁。" >&2
    curl --silent --fail --max-time 2 -X POST \
      "http://127.0.0.1:${port}/control/exit-immersive" >/dev/null 2>&1 || true
    sleep 0.8
    curl --silent --fail --max-time 2 -X POST \
      "http://127.0.0.1:${port}/control/shutdown" >/dev/null 2>&1 || \
      systemctl --user kill --signal=SIGTERM "${unit_name}" >/dev/null 2>&1 || true
    for ((attempt = 0; attempt < 300; attempt += 1)); do
      if ! unit_is_alive; then
        stopped=true
        break
      fi
      sleep 0.1
    done
  else
    if curl --silent --fail --max-time 2 -X POST \
      "http://127.0.0.1:${port}/control/standby" >/dev/null 2>&1; then
      for ((attempt = 0; attempt < 50; attempt += 1)); do
        health="$(curl --silent --fail --max-time 1 "http://127.0.0.1:${port}/healthz" 2>/dev/null || true)"
        if grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*false' <<< "${health}" \
          && grep -Eq '"simulationActive"[[:space:]]*:[[:space:]]*true' <<< "${health}"; then
          standing_by=true
          break
        elif grep -Eq '"teleoperationActive"[[:space:]]*:[[:space:]]*false' <<< "${health}" \
          && grep -Eq '"simulationActive"[[:space:]]*:[[:space:]]*false' <<< "${health}"; then
          parked=true
          break
        fi
        sleep 0.1
      done
    fi
  fi
fi

# Compatibility cleanup for a demo launched by an older nohup-based script.
if [[ "${managed}" != true && -r "${pid_file}" && "${terminate_process}" == "1" ]]; then
  read -r demo_pid < "${pid_file}" || true
  if is_demo_pid "${demo_pid:-}"; then
    managed=true
    curl --silent --fail --max-time 2 -X POST \
      "http://127.0.0.1:${port}/control/exit-immersive" >/dev/null 2>&1 || true
    sleep 0.8
    curl --silent --fail --max-time 2 -X POST \
      "http://127.0.0.1:${port}/control/shutdown" >/dev/null 2>&1 || kill -TERM "${demo_pid}"
    for ((attempt = 0; attempt < 300; attempt += 1)); do
      if ! is_demo_pid "${demo_pid}"; then
        stopped=true
        break
      fi
      sleep 0.1
    done
  fi
fi

if [[ "${standing_by}" == true || "${parked}" == true || "${managed}" != true ]]; then
  # Disarm Newton before changing the browser XR session. The default path
  # intentionally leaves both CUDA work and the USB reverse mapping active.
  curl --silent --fail --max-time 2 -X POST \
    "http://127.0.0.1:${port}/control/exit-immersive" >/dev/null 2>&1 || true
fi

if [[ "${stopped}" == true ]]; then
  adb reverse --remove "tcp:${port}" >/dev/null 2>&1 || true
  rm -f "${pid_file}"
  rm -f "${active_run_file}"
  echo "已退出 Quest 沉浸模式并停止 Newton 遥操。"
elif [[ "${standing_by}" == true ]]; then
  echo "已退出 Quest 沉浸模式并让 ${teleop_name} 进入安全待机。"
  echo "手柄输入和录制已暂停；CUDA 物理帧及 ADB 映射保持运行，避免设备电源/连接状态突变。"
  echo "可运行同一启动脚本恢复，或运行另一个场景的启动脚本完成安全接棒。"
  echo "注意：再次启动只会恢复同一 PID；期间修改的 Python 代码不会被该进程重新加载。"
  echo "需要加载更新时，请使用该场景对应的 reload_quest_webxr_*_teleop.sh 分阶段重载脚本。"
elif [[ "${parked}" == true ]]; then
  echo "${teleop_name} 已由另一个活动场景安全接棒并处于停泊状态；未重新启动其物理帧。"
elif [[ "${managed}" == true ]]; then
  echo "错误：已请求退出 Quest，但 Newton 未能进入安全待机或结束；未执行停泊、ADB 移除或强杀。" >&2
  exit 1
else
  rm -f "${pid_file}"
  if [[ -e "${active_run_file}" ]]; then
    rm -f "${active_run_file}"
    echo "已确认无 demo 进程，并清除上一次异常运行标记。"
  else
    echo "已请求退出 Quest 沉浸模式；未发现由一键启动脚本管理的 demo 进程。"
  fi
fi
