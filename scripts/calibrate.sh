#!/bin/bash
# 校準 SO-101 手臂
# 用法:
#   bash scripts/calibrate.sh follower    # 校準 follower
#   bash scripts/calibrate.sh leader      # 校準 leader
#   bash scripts/calibrate.sh both        # 兩個都校準

LEADER_PORT="/dev/ttyACM0"
FOLLOWER_PORT="/dev/ttyACM1"

TARGET="${1:-both}"

echo "=== SO-101 手臂校準 ==="
echo ""

# 確認 port 權限
for port in ${FOLLOWER_PORT} ${LEADER_PORT}; do
    if [ -e "$port" ] && [ ! -w "$port" ]; then
        echo "⚠ ${port} 權限不足，嘗試修復..."
        sudo chmod 666 "$port"
    fi
done

cd "$(dirname "$0")/../Isaac-GR00T"
source .venv/bin/activate
source scripts/activate_thor.sh

if [ "$TARGET" = "follower" ] || [ "$TARGET" = "both" ]; then
    echo "[Follower] 開始校準 (${FOLLOWER_PORT})..."
    echo "  按照提示擺放手臂位置，然後按 Enter"
    echo ""
    python -m lerobot.scripts.lerobot_calibrate \
      --robot.type=so101_follower \
      --robot.port=${FOLLOWER_PORT} \
      --robot.id=my_awesome_follower_arm
    echo ""
fi

if [ "$TARGET" = "leader" ] || [ "$TARGET" = "both" ]; then
    echo "[Leader] 開始校準 (${LEADER_PORT})..."
    echo "  按照提示擺放手臂位置，然後按 Enter"
    echo ""
    python -m lerobot.scripts.lerobot_calibrate \
      --teleop.type=so101_leader \
      --teleop.port=${LEADER_PORT} \
      --teleop.id=my_awesome_leader_arm
    echo ""
fi

echo "=== 校準完成 ==="
