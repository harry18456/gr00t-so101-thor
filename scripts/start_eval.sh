#!/bin/bash
# 啟動 GR00T 推論 Client（控制 follower 手臂）
# 用法: bash scripts/start_eval.sh [lang_instruction]
# 範例: bash scripts/start_eval.sh "pick up the ball and place it in the cup"
#       bash scripts/start_eval.sh   （使用預設指令）
#
# 前提: Server 必須先啟動（bash scripts/start_server.sh）

LANG_INSTRUCTION="${1:-pick up the ball and place it in the cup}"
FOLLOWER_PORT="/dev/ttyACM1"
WRIST_CAM=2
FRONT_CAM=0

echo "=== GR00T 推論 Client ==="
echo "  Follower: $FOLLOWER_PORT"
echo "  Cameras: wrist=$WRIST_CAM, front=$FRONT_CAM"
echo "  指令: $LANG_INSTRUCTION"
echo "  Ctrl+C 停止"
echo ""

# 確認 port 權限
if [ ! -w "$FOLLOWER_PORT" ]; then
    echo "⚠ ${FOLLOWER_PORT} 權限不足，嘗試修復..."
    sudo chmod 666 "$FOLLOWER_PORT"
fi

# 確認 server 是否在跑
if ! lsof -i :5555 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "❌ Server 未啟動（port 5555 沒有在監聽）"
    echo "   請先在另一個 terminal 執行: bash scripts/start_server.sh"
    exit 1
fi

cd "$(dirname "$0")/../Isaac-GR00T"
source .venv/bin/activate
source scripts/activate_thor.sh

python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{wrist: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: $FRONT_CAM, width: 640, height: 480, fps: 30}}" \
  --policy_host=0.0.0.0 \
  --lang_instruction="$LANG_INSTRUCTION"
