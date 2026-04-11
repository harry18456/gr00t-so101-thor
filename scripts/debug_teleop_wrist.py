"""
Debug teleop for wrist_roll ONLY.
Manually reads leader → normalizes → unnormalizes → writes to follower.
Prints every step so we can see where the mapping breaks.

Usage: python3 scripts/debug_teleop_wrist.py
Press Ctrl+C to stop.
"""
import json
import time
import sys

# 使用 lerobot 的 motors bus 來確保行為和 teleop 一致
sys.path.insert(0, ".")

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

LEADER_PORT = "/dev/ttyACM0"
FOLLOWER_PORT = "/dev/ttyACM1"

# 讀取校準檔
leader_cal_path = "/home/asus/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json"
follower_cal_path = "/home/asus/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json"

leader_cal_data = json.load(open(leader_cal_path))
follower_cal_data = json.load(open(follower_cal_path))

# 建立 leader bus（只有 wrist_roll）
leader_cal = {
    "wrist_roll": MotorCalibration(
        id=leader_cal_data["wrist_roll"]["id"],
        drive_mode=leader_cal_data["wrist_roll"]["drive_mode"],
        homing_offset=leader_cal_data["wrist_roll"]["homing_offset"],
        range_min=leader_cal_data["wrist_roll"]["range_min"],
        range_max=leader_cal_data["wrist_roll"]["range_max"],
    )
}

follower_cal = {
    "wrist_roll": MotorCalibration(
        id=follower_cal_data["wrist_roll"]["id"],
        drive_mode=follower_cal_data["wrist_roll"]["drive_mode"],
        homing_offset=follower_cal_data["wrist_roll"]["homing_offset"],
        range_min=follower_cal_data["wrist_roll"]["range_min"],
        range_max=follower_cal_data["wrist_roll"]["range_max"],
    )
}

print("=== Debug Teleop: wrist_roll only ===")
print(f"Leader cal:   offset={leader_cal['wrist_roll'].homing_offset}, range=[{leader_cal['wrist_roll'].range_min}, {leader_cal['wrist_roll'].range_max}]")
print(f"Follower cal: offset={follower_cal['wrist_roll'].homing_offset}, range=[{follower_cal['wrist_roll'].range_min}, {follower_cal['wrist_roll'].range_max}]")
print()

# 建立 motor bus
leader_bus = FeetechMotorsBus(
    port=LEADER_PORT,
    motors={"wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100)},
    calibration=leader_cal,
)

follower_bus = FeetechMotorsBus(
    port=FOLLOWER_PORT,
    motors={"wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100)},
    calibration=follower_cal,
)

leader_bus.connect()
follower_bus.connect()

# 寫入校準到馬達
leader_bus.write_calibration(leader_cal)
follower_bus.write_calibration(follower_cal)

# 驗證校準寫入
print("--- Verifying calibration written to motors ---")
l_offset_hw = leader_bus.read("Homing_Offset", "wrist_roll", normalize=False)
f_offset_hw = follower_bus.read("Homing_Offset", "wrist_roll", normalize=False)
l_min_hw = leader_bus.read("Min_Position_Limit", "wrist_roll", normalize=False)
l_max_hw = leader_bus.read("Max_Position_Limit", "wrist_roll", normalize=False)
f_min_hw = follower_bus.read("Min_Position_Limit", "wrist_roll", normalize=False)
f_max_hw = follower_bus.read("Max_Position_Limit", "wrist_roll", normalize=False)

print(f"Leader HW:   offset={l_offset_hw}, limits=[{l_min_hw}, {l_max_hw}]")
print(f"Follower HW: offset={f_offset_hw}, limits=[{f_min_hw}, {f_max_hw}]")
print()

# 讀取初始位置
l_raw = leader_bus.read("Present_Position", "wrist_roll", normalize=False)
l_norm = leader_bus.read("Present_Position", "wrist_roll", normalize=True)
f_raw = follower_bus.read("Present_Position", "wrist_roll", normalize=False)
f_norm = follower_bus.read("Present_Position", "wrist_roll", normalize=True)

print(f"Initial positions:")
print(f"  Leader:   raw={l_raw}, normalized={l_norm:.2f}")
print(f"  Follower: raw={f_raw}, normalized={f_norm:.2f}")
print()

# 配置 follower（和 lerobot 一樣）
follower_bus.disable_torque("wrist_roll")
follower_bus.write("Operating_Mode", "wrist_roll", OperatingMode.POSITION.value)
follower_bus.write("P_Coefficient", "wrist_roll", 16)
follower_bus.write("I_Coefficient", "wrist_roll", 0)
follower_bus.write("D_Coefficient", "wrist_roll", 32)

input("按 Enter 啟動 torque 開始測試 (確認手臂安全)...")

follower_bus.enable_torque("wrist_roll")
print("Torque ON! 開始 teleop 迴圈 (Ctrl+C 停止)")
print()

try:
    for i in range(200):  # 最多跑 200 次迭代
        # 1. 讀 leader 位置
        l_raw = leader_bus.read("Present_Position", "wrist_roll", normalize=False)
        l_norm = leader_bus.read("Present_Position", "wrist_roll", normalize=True)

        # 2. 讀 follower 位置
        f_raw = follower_bus.read("Present_Position", "wrist_roll", normalize=False)

        # 3. 寫入目標到 follower（用 normalized 值）
        follower_bus.write("Goal_Position", "wrist_roll", l_norm)

        # 4. 計算 follower 的 unnormalized 目標
        f_min = follower_cal_data["wrist_roll"]["range_min"]
        f_max = follower_cal_data["wrist_roll"]["range_max"]
        bounded = min(100.0, max(-100.0, l_norm))
        f_goal_calc = int(((bounded + 100) / 200) * (f_max - f_min) + f_min)

        if i < 10 or i % 20 == 0:
            print(f"[{i:3d}] leader: raw={l_raw:5d} norm={l_norm:7.2f}  →  follower: pos={f_raw:5d} goal_calc={f_goal_calc:5d}")

        time.sleep(0.05)  # 20Hz

except KeyboardInterrupt:
    print("\n\n停止!")
finally:
    follower_bus.disable_torque("wrist_roll")
    print("Torque OFF")
    leader_bus.disconnect()
    follower_bus.disconnect()
