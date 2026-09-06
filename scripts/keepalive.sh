#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="avatar-studio.service"
LOCK_FILE="/run/user/$(id -u)/avatar-studio.keepalive.lock"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

systemctl --user start "${SERVICE_NAME}"

while true; do
  service_state="$(systemctl --user show "${SERVICE_NAME}" --property=ActiveState --value 2>/dev/null || true)"
  case "${service_state}" in
    active|activating|reloading|deactivating)
      sleep 2
      ;;
    *)
      exit 0
      ;;
  esac
done
