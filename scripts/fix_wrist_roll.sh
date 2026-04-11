#!/bin/bash
# 修正 wrist_roll 校準偏移
# 用法: 把兩隻手臂的 wrist_roll 擺到同一個方向，然後跑這個 script
#   bash scripts/fix_wrist_roll.sh

LEADER_PORT="/dev/ttyACM0"
FOLLOWER_PORT="/dev/ttyACM1"

cd "$(dirname "$0")/../Isaac-GR00T"
source .venv/bin/activate
source scripts/activate_thor.sh

echo "=== Wrist Roll 校準修正 ==="
echo ""
echo "請確認兩隻手臂的 wrist_roll 都擺在同一個物理方向"
echo "（例如夾爪線條都朝正前方）"
echo ""
read -p "準備好了按 Enter..." _

python3 -c "
import scservo_sdk as scs
import json

leader_port = '${LEADER_PORT}'
follower_port = '${FOLLOWER_PORT}'
cal_path = '/home/asus/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json'
leader_cal_path = '/home/asus/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json'

# 讀取兩邊 wrist_roll 原始位置
positions = {}
for port, name in [(leader_port, 'leader'), (follower_port, 'follower')]:
    ph = scs.PortHandler(port)
    ph.openPort()
    ph.setBaudRate(1000000)
    pk = scs.PacketHandler(0)
    pos, _, _ = pk.read2ByteTxRx(ph, 5, 56)
    positions[name] = pos
    ph.closePort()

print(f'Leader  wrist_roll 原始位置: {positions[\"leader\"]}')
print(f'Follower wrist_roll 原始位置: {positions[\"follower\"]}')

# 讀取 leader 的 offset
leader_cal = json.load(open(leader_cal_path))
leader_offset = leader_cal['wrist_roll']['homing_offset']
leader_logical = positions['leader'] + leader_offset

# 計算 follower 需要的 offset
new_offset = leader_logical - positions['follower']

print(f'')
print(f'Leader offset: {leader_offset}, logical: {leader_logical}')
print(f'Follower 新 offset: {new_offset} (舊: ', end='')

# 更新 follower 校準檔
cal = json.load(open(cal_path))
old_offset = cal['wrist_roll']['homing_offset']
print(f'{old_offset})')

cal['wrist_roll']['homing_offset'] = new_offset
json.dump(cal, open(cal_path, 'w'), indent=4)
print(f'')
print(f'✓ Follower wrist_roll offset 已更新: {old_offset} → {new_offset}')
print(f'  現在兩邊邏輯位置都是 {leader_logical}')
"

echo ""
echo "修正完成！跑 bash scripts/teleop_test.sh 測試看看"
