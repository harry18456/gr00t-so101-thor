"""診斷 wrist_roll 問題：讀取兩隻手臂的馬達寄存器和校準檔，分析 teleop 映射"""
import scservo_sdk as scs
import json
import time

LEADER_PORT = "/dev/ttyACM0"
FOLLOWER_PORT = "/dev/ttyACM1"
WRIST_ROLL_ID = 5

# 讀取校準檔
leader_cal_path = "/home/asus/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json"
follower_cal_path = "/home/asus/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json"

leader_cal = json.load(open(leader_cal_path))["wrist_roll"]
follower_cal = json.load(open(follower_cal_path))["wrist_roll"]

print("=== 校準檔 ===")
print(f"Leader:   offset={leader_cal['homing_offset']}, range=[{leader_cal['range_min']}, {leader_cal['range_max']}]")
print(f"Follower: offset={follower_cal['homing_offset']}, range=[{follower_cal['range_min']}, {follower_cal['range_max']}]")
print()

# 讀取馬達硬體寄存器
for port, name, cal in [(LEADER_PORT, "Leader", leader_cal), (FOLLOWER_PORT, "Follower", follower_cal)]:
    ph = scs.PortHandler(port)
    ph.openPort()
    ph.setBaudRate(1000000)
    pk = scs.PacketHandler(0)
    time.sleep(0.3)

    # 讀取各寄存器（帶重試）
    def read_reg(addr, nbytes=2, retries=3):
        for _ in range(retries):
            try:
                if nbytes == 2:
                    val, res, err = pk.read2ByteTxRx(ph, WRIST_ROLL_ID, addr)
                else:
                    val, res, err = pk.read1ByteTxRx(ph, WRIST_ROLL_ID, addr)
                if res == 0:
                    return val
            except:
                pass
            time.sleep(0.1)
        return None

    min_limit = read_reg(9)     # Min_Position_Limit
    max_limit = read_reg(11)    # Max_Position_Limit
    homing_off_raw = read_reg(31)  # Homing_Offset (sign-magnitude, bit 11)
    present_pos_raw = read_reg(56)  # Present_Position (sign-magnitude, bit 15)
    op_mode = read_reg(33, 1)    # Operating_Mode
    torque = read_reg(40, 1)     # Torque_Enable
    goal_pos_raw = read_reg(42)  # Goal_Position (sign-magnitude, bit 15)

    # Decode sign-magnitude
    def decode_sm(val, sign_bit):
        if val is None:
            return None
        direction = (val >> sign_bit) & 1
        magnitude = val & ((1 << sign_bit) - 1)
        return -magnitude if direction else magnitude

    homing_off = decode_sm(homing_off_raw, 11)
    present_pos = decode_sm(present_pos_raw, 15)
    goal_pos = decode_sm(goal_pos_raw, 15)

    print(f"=== {name} 馬達 ID {WRIST_ROLL_ID} 硬體寄存器 ===")
    print(f"  Homing_Offset:      raw={homing_off_raw}, decoded={homing_off}")
    print(f"  Min_Position_Limit: {min_limit}")
    print(f"  Max_Position_Limit: {max_limit}")
    print(f"  Present_Position:   raw={present_pos_raw}, decoded={present_pos}")
    print(f"  Goal_Position:      raw={goal_pos_raw}, decoded={goal_pos}")
    print(f"  Operating_Mode:     {op_mode}")
    print(f"  Torque_Enable:      {torque}")

    # 計算實際 encoder 值
    if present_pos is not None and homing_off is not None:
        actual_encoder = present_pos + homing_off
        print(f"  Actual encoder:     {actual_encoder} (present + offset)")
    print()

    ph.closePort()

# === 模擬 teleop 映射 ===
print("=== Teleop 映射模擬 ===")
print()

# 讀取 leader 的當前位置
ph = scs.PortHandler(LEADER_PORT)
ph.openPort()
ph.setBaudRate(1000000)
pk = scs.PacketHandler(0)
time.sleep(0.3)
lp_raw, _, _ = pk.read2ByteTxRx(ph, WRIST_ROLL_ID, 56)
leader_present = decode_sm(lp_raw, 15)
ph.closePort()

# 讀取 follower 的當前位置
ph = scs.PortHandler(FOLLOWER_PORT)
ph.openPort()
ph.setBaudRate(1000000)
pk = scs.PacketHandler(0)
time.sleep(0.3)
fp_raw, _, _ = pk.read2ByteTxRx(ph, WRIST_ROLL_ID, 56)
follower_present = decode_sm(fp_raw, 15)
ph.closePort()

# 模擬 normalize（leader）
l_min = leader_cal['range_min']
l_max = leader_cal['range_max']
bounded = min(l_max, max(l_min, leader_present))
normalized = ((bounded - l_min) / (l_max - l_min)) * 200 - 100

print(f"Leader Present_Position: {leader_present}")
print(f"Leader normalize: bounded={bounded}, range=[{l_min},{l_max}] → normalized={normalized:.2f}")

# 模擬 unnormalize（follower）
f_min = follower_cal['range_min']
f_max = follower_cal['range_max']
bounded_norm = min(100.0, max(-100.0, normalized))
follower_goal = int(((bounded_norm + 100) / 200) * (f_max - f_min) + f_min)

print(f"Follower unnormalize: normalized={bounded_norm:.2f}, range=[{f_min},{f_max}] → goal={follower_goal}")
print(f"Follower current: {follower_present}")
print(f"Follower jump: {follower_present} → {follower_goal} (delta={follower_goal - follower_present})")
print()

# 邊界情況分析
print("=== 邊界分析 ===")
for leader_pos in [l_min, 2047, l_max]:
    bounded = min(l_max, max(l_min, leader_pos))
    norm = ((bounded - l_min) / (l_max - l_min)) * 200 - 100
    f_goal = int(((norm + 100) / 200) * (f_max - f_min) + f_min)
    print(f"  Leader={leader_pos} → norm={norm:.1f} → Follower goal={f_goal}")
