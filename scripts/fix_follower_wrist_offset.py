"""
修正 follower wrist_roll 的 homing_offset 為正值。

STS3215 馬達韌體在負 homing_offset 時 PID 會失控（正回饋），
所以必須確保 offset 為正值。

做法：讀取 follower wrist_roll 目前實際 encoder 位置，
計算讓 Present_Position = 2047 的正 offset。
如果計算出來的 offset 是負的，透過加/減 4096 來調整到正值範圍。

用法：
  1. 把 follower 的 wrist_roll 擺到和 leader 大致相同的方向
  2. python3 scripts/fix_follower_wrist_offset.py
  3. bash scripts/teleop_test.sh 測試
"""
import scservo_sdk as scs
import json
import time

FOLLOWER_PORT = "/dev/ttyACM1"
LEADER_PORT = "/dev/ttyACM0"
WRIST_ROLL_ID = 5

leader_cal_path = "/home/asus/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json"
follower_cal_path = "/home/asus/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json"


def decode_sm(val, sign_bit):
    direction = (val >> sign_bit) & 1
    magnitude = val & ((1 << sign_bit) - 1)
    return -magnitude if direction else magnitude


def encode_sm(val, sign_bit):
    if val < 0:
        return (1 << sign_bit) | abs(val)
    return val


def read_reg(pk, ph, motor_id, addr, nbytes=2, retries=3):
    for _ in range(retries):
        try:
            if nbytes == 2:
                val, res, err = pk.read2ByteTxRx(ph, motor_id, addr)
            else:
                val, res, err = pk.read1ByteTxRx(ph, motor_id, addr)
            if res == 0:
                return val
        except:
            pass
        time.sleep(0.1)
    return None


# Step 1: 先把 follower wrist_roll offset 設為 0，讀取真實 encoder 位置
print("=== Step 1: 讀取實際 encoder 位置 ===")

ph = scs.PortHandler(FOLLOWER_PORT)
ph.openPort()
ph.setBaudRate(1000000)
pk = scs.PacketHandler(0)
time.sleep(0.3)

# 關閉 torque
pk.write1ByteTxRx(ph, WRIST_ROLL_ID, 40, 0)
time.sleep(0.1)

# 暫時設 offset 為 0 以讀取真實 encoder
pk.write2ByteTxRx(ph, WRIST_ROLL_ID, 31, 0)
time.sleep(0.1)

# 讀取真實 encoder（offset=0 時 Present=Actual）
raw = read_reg(pk, ph, WRIST_ROLL_ID, 56)
follower_actual = decode_sm(raw, 15)
print(f"Follower wrist_roll 實際 encoder: {follower_actual}")

ph.closePort()

# Step 2: 讀取 leader 的真實 encoder 位置
ph = scs.PortHandler(LEADER_PORT)
ph.openPort()
ph.setBaudRate(1000000)
pk = scs.PacketHandler(0)
time.sleep(0.3)

leader_offset_raw = read_reg(pk, ph, WRIST_ROLL_ID, 31)
leader_offset = decode_sm(leader_offset_raw, 11)
leader_present_raw = read_reg(pk, ph, WRIST_ROLL_ID, 56)
leader_present = decode_sm(leader_present_raw, 15)
leader_actual = leader_present + leader_offset

print(f"Leader wrist_roll: present={leader_present}, offset={leader_offset}, actual={leader_actual}")
ph.closePort()

# Step 3: 計算正確的 follower offset
# 公式: Present_Position = Actual - Homing_Offset
# 我們要 Present ≈ 2047 (midpoint) → offset = Actual - 2047
new_offset = follower_actual - 2047

print(f"\n計算: offset = actual({follower_actual}) - 2047 = {new_offset}")

# 如果 offset 是負的，代表 actual < 2047
# STS3215 韌體無法正常處理負 offset，需要調整
if new_offset < 0:
    # 實際 encoder < 2047，offset 會是負的
    # 解法：不用 2047 作為 midpoint，而是直接用 actual encoder 作為 "零點"
    # 把 offset 設為 0，然後調整 range_min/range_max 以 actual 為中心
    print(f"⚠ offset 為負值 ({new_offset})，STS3215 韌體會導致 PID 失控！")
    print(f"  調整方案：讓 offset 為正值")

    # 方法：offset 必須 > 0 且 offset = actual - target_present
    # 讓 target_present = actual - small_positive_offset
    # 最簡單：offset = 0，present = actual
    # 但這會讓 present range 偏移

    # 更好的方法：如果 actual 很小（<2047），offset 是負的
    # 重新校準前先把手臂轉到 encoder > 2047 的位置
    print(f"\n  follower 實際 encoder = {follower_actual}")
    if follower_actual < 2047:
        print(f"  encoder < 2047，需要把 wrist_roll 轉到另一邊")
        print(f"  大約需要轉 {(2047 - follower_actual) / 4096 * 360:.0f} 度")
        print(f"\n  ===== 快速修復 =====")
        print(f"  直接把 offset 設為 0，range 用 [0, 4095]")
        new_offset = 0
    else:
        new_offset = follower_actual - 2047

print(f"\n最終 offset: {new_offset}")

if new_offset < 0:
    print("ERROR: offset 仍為負值，無法修正。請手動轉動 wrist_roll 到另一側後重試。")
    exit(1)

# Step 4: 寫入新的 offset 和 range
print(f"\n=== Step 4: 寫入修正值 ===")

# 讀取 leader 校準以保持 range 一致
leader_cal = json.load(open(leader_cal_path))
follower_cal = json.load(open(follower_cal_path))

# 計算新的 present position
new_present = follower_actual - new_offset

# 計算以 new_present 為中心的 range
# 用和 leader 相同的 range 寬度
leader_range = leader_cal["wrist_roll"]["range_max"] - leader_cal["wrist_roll"]["range_min"]
new_range_min = max(0, new_present - leader_range // 2)
new_range_max = min(4095, new_present + leader_range // 2)

# 如果 offset 為 0，直接用全範圍（和 leader 一樣的策略）
if new_offset == 0:
    new_range_min = 0
    new_range_max = 4095

print(f"  old offset: {follower_cal['wrist_roll']['homing_offset']}")
print(f"  new offset: {new_offset}")
print(f"  new present (predicted): {new_present}")
print(f"  new range: [{new_range_min}, {new_range_max}]")

# 寫入馬達
ph = scs.PortHandler(FOLLOWER_PORT)
ph.openPort()
ph.setBaudRate(1000000)
pk = scs.PacketHandler(0)
time.sleep(0.3)

pk.write1ByteTxRx(ph, WRIST_ROLL_ID, 40, 0)  # torque off
time.sleep(0.1)

# 寫入新 offset
encoded_offset = encode_sm(new_offset, 11)
pk.write2ByteTxRx(ph, WRIST_ROLL_ID, 31, encoded_offset)
time.sleep(0.1)

# 寫入新 range
pk.write2ByteTxRx(ph, WRIST_ROLL_ID, 9, new_range_min)
time.sleep(0.1)
pk.write2ByteTxRx(ph, WRIST_ROLL_ID, 11, new_range_max)
time.sleep(0.1)

# 讀回確認
for retry in range(3):
    try:
        off_read = read_reg(pk, ph, WRIST_ROLL_ID, 31)
        min_read = read_reg(pk, ph, WRIST_ROLL_ID, 9)
        max_read = read_reg(pk, ph, WRIST_ROLL_ID, 11)
        pos_raw = read_reg(pk, ph, WRIST_ROLL_ID, 56)
        pos = decode_sm(pos_raw, 15) if pos_raw else None
        off_decoded = decode_sm(off_read, 11) if off_read else None
        print(f"\n  馬達確認: offset={off_decoded} (raw={off_read}), limits=[{min_read}, {max_read}], present={pos}")
        break
    except:
        time.sleep(0.2)

ph.closePort()

# Step 5: 更新校準檔
print(f"\n=== Step 5: 更新校準檔 ===")
follower_cal["wrist_roll"]["homing_offset"] = new_offset
follower_cal["wrist_roll"]["range_min"] = new_range_min
follower_cal["wrist_roll"]["range_max"] = new_range_max
json.dump(follower_cal, open(follower_cal_path, "w"), indent=4)
print(f"  已更新: {follower_cal_path}")

# 比較兩邊
print(f"\n=== 比較 ===")
print(f"Leader:   offset={leader_cal['wrist_roll']['homing_offset']}, range=[{leader_cal['wrist_roll']['range_min']}, {leader_cal['wrist_roll']['range_max']}]")
print(f"Follower: offset={new_offset}, range=[{new_range_min}, {new_range_max}]")

# 同時修正 leader 的 range 為 [0, 4095]（和 SO100 一樣的策略）
if leader_cal["wrist_roll"]["range_min"] != 0 or leader_cal["wrist_roll"]["range_max"] != 4095:
    print(f"\n⚠ Leader wrist_roll range 不是 [0, 4095]，一併修正...")
    leader_cal["wrist_roll"]["range_min"] = 0
    leader_cal["wrist_roll"]["range_max"] = 4095
    json.dump(leader_cal, open(leader_cal_path, "w"), indent=4)

    # 也寫入 leader 馬達
    ph = scs.PortHandler(LEADER_PORT)
    ph.openPort()
    ph.setBaudRate(1000000)
    pk = scs.PacketHandler(0)
    time.sleep(0.3)
    pk.write2ByteTxRx(ph, WRIST_ROLL_ID, 9, 0)
    time.sleep(0.1)
    pk.write2ByteTxRx(ph, WRIST_ROLL_ID, 11, 4095)
    time.sleep(0.1)
    ph.closePort()
    print(f"  Leader 已修正為 [0, 4095]")

print(f"\n✓ 修正完成！測試: bash scripts/teleop_test.sh")
