#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_LOG="${PROJECT_DIR}/runtime/logs/backend.log"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/runtime/logs"

vinext_lock="${PROJECT_DIR}/.vinext/dev/lock.json"
if [[ -f "${vinext_lock}" ]]; then
  lock_pid="$(jq -r '.pid // empty' "${vinext_lock}" 2>/dev/null || true)"
  if [[ -z "${lock_pid}" ]] || ! kill -0 "${lock_pid}" 2>/dev/null; then
    rm -f "${vinext_lock}"
  fi
fi

backend_pid=""
frontend_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${frontend_pid}" ]] && kill -0 "${frontend_pid}" 2>/dev/null; then
    kill "${frontend_pid}" 2>/dev/null || true
  fi
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    kill "${backend_pid}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

backend/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8765 >>"${BACKEND_LOG}" 2>&1 &
backend_pid=$!

backend_ready=false
for _ in {1..80}; do
  if curl -fsS http://127.0.0.1:8765/api/health >/dev/null 2>&1; then
    backend_ready=true
    break
  fi
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    echo "后端启动失败，请检查 ${BACKEND_LOG}" >&2
    exit 1
  fi
  sleep 0.25
done

if [[ "${backend_ready}" != true ]]; then
  echo "后端健康检查超时，请检查 ${BACKEND_LOG}" >&2
  exit 1
fi

npm run dev -- --host 127.0.0.1 &
frontend_pid=$!

while kill -0 "${backend_pid}" 2>/dev/null && kill -0 "${frontend_pid}" 2>/dev/null; do
  sleep 1
done

echo "工作台进程意外退出，systemd 将按策略重启。" >&2
exit 1
