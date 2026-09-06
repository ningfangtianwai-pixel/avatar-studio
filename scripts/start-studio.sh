#!/usr/bin/env bash
set -euo pipefail

systemctl --user start avatar-studio.service

for _ in {1..80}; do
  if curl -fsS http://127.0.0.1:8765/api/health >/dev/null 2>&1 \
    && curl -fsS http://127.0.0.1:3000/ >/dev/null 2>&1; then
    echo "数字人口播工作台已在后台运行：http://localhost:3000"
    echo "关闭当前窗口不会中断生成任务。"
    exit 0
  fi
  sleep 0.25
done

echo "工作台启动超时。请运行：systemctl --user status avatar-studio.service" >&2
exit 1
