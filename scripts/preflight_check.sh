#!/bin/bash
# 錄製前預檢腳本：檢查 CH34x driver、手臂馬達、攝影機、port 權限
# 用法: bash scripts/preflight_check.sh

FOLLOWER_PORT="/dev/ttyACM1"
LEADER_PORT="/dev/ttyACM0"
FRONT_CAM_INDEX=0
WRIST_CAM_INDEX=2

PASS=0
FAIL=0

ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

echo "=== 預檢開始 ==="
echo ""

# 1. CH34x kernel module
echo "[1/5] CH34x 驅動"
if lsmod | grep -q ch34; then
    ok "ch34x kernel module 已載入"
else
    fail "ch34x kernel module 未載入"
    echo "    → 修復: cd CH341SER && make && sudo make load"
fi
echo ""

# 2. Serial port 存在
echo "[2/5] 手臂 Serial Port"
for port in ${FOLLOWER_PORT} ${LEADER_PORT}; do
    if [ -e "$port" ]; then
        ok "${port} 存在"
    else
        fail "${port} 不存在，手臂可能未接上"
    fi
done
echo ""

# 3. Port 權限（不足則自動修復）
echo "[3/5] Port 權限"
for port in ${FOLLOWER_PORT} ${LEADER_PORT}; do
    if [ ! -e "$port" ]; then
        continue
    fi
    if [ -w "$port" ]; then
        ok "${port} 可寫入"
    else
        echo "    ⚠ ${port} 權限不足，嘗試 chmod 666..."
        sudo chmod 666 "$port" 2>/dev/null
        if [ -w "$port" ]; then
            ok "${port} 權限已修復"
        else
            fail "${port} 權限修復失敗，請手動: sudo chmod 666 ${port}"
        fi
    fi
done
echo ""

# 4. 攝影機
echo "[4/5] 攝影機"
for dev in /dev/video${FRONT_CAM_INDEX} /dev/video${WRIST_CAM_INDEX}; do
    if [ -e "$dev" ]; then
        ok "${dev} 存在"
    else
        fail "${dev} 不存在"
    fi
done
echo ""

# 5. 馬達掃描（需要 GR00T venv）
echo "[5/5] 馬達掃描"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GROOT_DIR="${SCRIPT_DIR}/../Isaac-GR00T"

if [ -f "${GROOT_DIR}/.venv/bin/activate" ]; then
    source "${GROOT_DIR}/.venv/bin/activate"
    source "${GROOT_DIR}/scripts/activate_thor.sh" 2>/dev/null

    for port_info in "${FOLLOWER_PORT}:follower" "${LEADER_PORT}:leader"; do
        port="${port_info%%:*}"
        name="${port_info##*:}"
        if [ ! -e "$port" ]; then
            fail "${name} (${port}) 不存在，跳過掃描"
            continue
        fi
        result=$(python3 -c "
import scservo_sdk as scs
ph = scs.PortHandler('${port}')
ph.openPort()
ph.setBaudRate(1000000)
pk = scs.PacketHandler(0)
found = []
for i in range(1, 7):
    m, r, e = pk.ping(ph, i)
    if r == 0: found.append(i)
ph.closePort()
print(','.join(map(str, found)))
" 2>/dev/null)
        count=$(echo "$result" | tr ',' '\n' | wc -w)
        if [ "$count" -eq 6 ]; then
            ok "${name} (${port}): 6/6 馬達 [${result}]"
        else
            fail "${name} (${port}): ${count}/6 馬達 [${result}]"
            echo "    → 檢查接線，重新插拔電源後再試"
        fi
    done
else
    fail "找不到 GR00T venv，跳過馬達掃描"
fi
echo ""

# 結果
echo "=== 預檢結果: ${PASS} 通過, ${FAIL} 失敗 ==="
if [ ${FAIL} -eq 0 ]; then
    echo "全部通過，可以開始錄製！"
    echo "  → bash scripts/record_data.sh so101_place_ball"
else
    echo "有 ${FAIL} 項需要修復"
fi
