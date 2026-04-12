#!/bin/bash
# 每次重開機後跑一次: 載入 CH34x driver + 修 serial port 權限
# 用法: bash scripts/post_boot.sh
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 1. CH34x kernel module
if lsmod | grep -q ch34; then
    echo "  = ch34x 已載入"
else
    echo "  → 編譯並載入 ch34x driver"
    cd "$PROJECT_ROOT/CH341SER"
    make
    sudo make load
    cd "$PROJECT_ROOT"
fi

# 2. Serial port 權限
for port in /dev/ttyACM0 /dev/ttyACM1; do
    if [ -e "$port" ]; then
        sudo chmod 666 "$port"
        echo "  ✓ $port 權限設定"
    else
        echo "  ⚠ $port 不存在"
    fi
done

echo "✓ post-boot setup done"
