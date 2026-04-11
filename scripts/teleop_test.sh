#!/bin/bash
# 遙操作測試（不錄資料）：確認 leader→follower 連動正常
# 用法: bash scripts/teleop_test.sh
# Ctrl+C 停止

LEADER_PORT="/dev/ttyACM0"
FOLLOWER_PORT="/dev/ttyACM1"

echo "=== SO-101 遙操作測試（不錄資料）==="
echo "  Leader:  ${LEADER_PORT}"
echo "  Follower: ${FOLLOWER_PORT}"
echo "  Ctrl+C 停止"
echo ""

# 確認 port 權限
for port in ${FOLLOWER_PORT} ${LEADER_PORT}; do
    if [ ! -w "$port" ]; then
        echo "⚠ ${port} 權限不足，嘗試修復..."
        sudo chmod 666 "$port"
    fi
done

cd "$(dirname "$0")/../Isaac-GR00T"
source .venv/bin/activate
source scripts/activate_thor.sh

python -m lerobot.scripts.lerobot_teleoperate \
  --teleop.type=so101_leader --teleop.port=${LEADER_PORT} --teleop.id=my_awesome_leader_arm \
  --robot.type=so101_follower --robot.port=${FOLLOWER_PORT} --robot.id=my_awesome_follower_arm
