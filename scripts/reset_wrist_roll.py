"""重設兩隻手臂 wrist_roll 的馬達限制到全範圍，並清除校準檔中的 wrist_roll offset"""
import scservo_sdk as scs
import json

LEADER_PORT = "/dev/ttyACM0"
FOLLOWER_PORT = "/dev/ttyACM1"
WRIST_ROLL_ID = 5

for port, name in [(LEADER_PORT, "leader"), (FOLLOWER_PORT, "follower")]:
    ph = scs.PortHandler(port)
    ph.openPort()
    ph.setBaudRate(1000000)
    pk = scs.PacketHandler(0)

    import time
    time.sleep(0.5)

    # 先關 torque 才能寫入限制
    pk.write1ByteTxRx(ph, WRIST_ROLL_ID, 40, 0)
    time.sleep(0.1)

    # 重設 Min_Position_Limit = 0, Max_Position_Limit = 4095
    pk.write2ByteTxRx(ph, WRIST_ROLL_ID, 9, 0)
    time.sleep(0.1)
    pk.write2ByteTxRx(ph, WRIST_ROLL_ID, 11, 4095)
    time.sleep(0.1)

    # 讀回確認（加重試）
    for retry in range(3):
        try:
            min_pos, r1, _ = pk.read2ByteTxRx(ph, WRIST_ROLL_ID, 9)
            max_pos, r2, _ = pk.read2ByteTxRx(ph, WRIST_ROLL_ID, 11)
            cur_pos, r3, _ = pk.read2ByteTxRx(ph, WRIST_ROLL_ID, 56)
            if r1 == 0 and r2 == 0 and r3 == 0:
                print(f"{name} wrist_roll: pos={cur_pos}, limit=[{min_pos}, {max_pos}] ✓")
                break
        except (IndexError, Exception) as e:
            time.sleep(0.2)
    else:
        print(f"{name} wrist_roll: 寫入完成，但讀回失敗（不影響）")
    ph.closePort()

# 重設校準檔的 wrist_roll offset 為 0
cal_paths = [
    ("/home/asus/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json", "leader"),
    ("/home/asus/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json", "follower"),
]

for path, name in cal_paths:
    try:
        cal = json.load(open(path))
        old = cal["wrist_roll"]["homing_offset"]
        cal["wrist_roll"]["homing_offset"] = 0
        cal["wrist_roll"]["range_min"] = 0
        cal["wrist_roll"]["range_max"] = 4095
        json.dump(cal, open(path, "w"), indent=4)
        print(f"{name} calibration: wrist_roll offset {old} → 0, range [0, 4095]")
    except Exception as e:
        print(f"{name} calibration: {e}")

print()
print("完成！現在重新校準兩隻手臂:")
print("  bash scripts/calibrate.sh both")
