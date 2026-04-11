#!/bin/bash
# 錄製 SO-101 遙操作資料集
# 用法: bash scripts/record_data.sh [dataset_name]
# 例如: bash scripts/record_data.sh so101_pick_cube

set -e

DATASET_NAME="${1:-so101_pick_cube}"
REPO_ID="harry18456/${DATASET_NAME}"

LEADER_PORT="/dev/ttyACM0"
FOLLOWER_PORT="/dev/ttyACM1"
WRIST_CAM_INDEX=0
FRONT_CAM_INDEX=2

echo "=== SO-101 資料錄製 ==="
echo "  Dataset: ${REPO_ID}"
echo "  Leader:  ${LEADER_PORT}"
echo "  Follower: ${FOLLOWER_PORT}"
echo "  Front cam: video${FRONT_CAM_INDEX}"
echo "  Wrist cam: video${WRIST_CAM_INDEX}"
echo ""

# 確認 port 權限
for port in ${FOLLOWER_PORT} ${LEADER_PORT}; do
    if [ ! -w "$port" ]; then
        echo "⚠ ${port} 沒有寫入權限，嘗試修復..."
        sudo chmod 666 "$port"
    fi
done

# 設定 DISPLAY 讓 pynput 鍵盤控制可用（需要 X server）
export DISPLAY="${DISPLAY:-:1}"

cd "$(dirname "$0")/../Isaac-GR00T"
source .venv/bin/activate
source scripts/activate_thor.sh

python -m lerobot.scripts.lerobot_record \
  --teleop.type=so101_leader --teleop.port=${LEADER_PORT} --teleop.id=my_awesome_leader_arm \
  --robot.type=so101_follower --robot.port=${FOLLOWER_PORT} --robot.id=my_awesome_follower_arm \
  --robot.cameras="{wrist: {type: opencv, index_or_path: ${WRIST_CAM_INDEX}, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: ${FRONT_CAM_INDEX}, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${REPO_ID} \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="pick up the ball and place it in the cup"
