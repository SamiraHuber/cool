#!/usr/bin/env bash
set -euo pipefail

PIDFILE="${SLAM_TOOLBOX_WRAPPER_PIDFILE:-/tmp/slam_toolbox_wrapper.pid}"
PARAMS_FILE="${SLAM_TOOLBOX_PARAMS_FILE:-/opt/local/slam_toolbox_params.yaml}"
LOCALIZATION_PARAMS_FILE="${SLAM_TOOLBOX_LOCALIZATION_PARAMS_FILE:-/opt/local/slam_toolbox_localization_params.yaml}"
MODE_FILE="${SLAM_TOOLBOX_MODE_FILE:-/tmp/slam_toolbox_mode.txt}"
ROS_PREFIX='source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && '

restart_requested=0
child_pid=""
child_pgid=""

start_child() {
  local mode executable selected_params
  mode="mapping"
  if [[ -f "${MODE_FILE}" ]]; then
    mode="$(tr -d '[:space:]' < "${MODE_FILE}")"
  fi
  if [[ "${mode}" == "localization" ]]; then
    executable="localization_slam_toolbox_node"
    selected_params="${LOCALIZATION_PARAMS_FILE}"
  else
    executable="async_slam_toolbox_node"
    mode="mapping"
    selected_params="${PARAMS_FILE}"
  fi
  echo "Starting slam_toolbox in ${mode} mode with ${executable} using ${selected_params}"
  setsid /bin/bash -lc "${ROS_PREFIX}exec ros2 run slam_toolbox ${executable} --ros-args --params-file ${selected_params}" &
  child_pid=$!
  child_pgid="${child_pid}"
}

stop_child() {
  if [[ -n "${child_pgid}" ]] && kill -0 "-${child_pgid}" 2>/dev/null; then
    kill "-${child_pgid}" 2>/dev/null || true
    sleep 0.5
    kill -9 "-${child_pgid}" 2>/dev/null || true
  elif [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  if [[ -n "${child_pid}" ]]; then
    wait "${child_pid}" 2>/dev/null || true
  fi
  child_pid=""
  child_pgid=""
}

request_restart() {
  restart_requested=1
  stop_child
}

cleanup() {
  trap - EXIT INT TERM USR1
  stop_child
  rm -f "${PIDFILE}"
}

trap cleanup EXIT INT TERM
trap request_restart USR1

echo "$$" > "${PIDFILE}"

while true; do
  # Recreate pid file if it was deleted (e.g. /tmp cleaned)
  if [[ ! -f "${PIDFILE}" ]]; then
    echo "$$" > "${PIDFILE}"
  fi
  restart_requested=0
  start_child
  wait "${child_pid}" 2>/dev/null || true
  child_pid=""
  if [[ "${restart_requested}" -eq 0 ]]; then
    sleep 1
  fi
done
