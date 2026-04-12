#!/bin/bash
# GR00T 環境啟用（idempotent，可重複 source）
# 手動用法:  source scripts/activate_env.sh
# Script 內:  source "$(dirname "$0")/activate_env.sh"
#
# 會做:
#   1. 切到專案根（PROJECT_ROOT）
#   2. 啟用 Isaac-GR00T/.venv（若尚未啟用）
#   3. source activate_thor.sh（若 TRITON_PTXAS_PATH 未設）
#   4. 修復 /dev/ttyACM0,1 權限（若存在但不可寫）

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

# 1. venv
EXPECTED_VENV="$PROJECT_ROOT/Isaac-GR00T/.venv"
if [ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]; then
    if [ -f "$EXPECTED_VENV/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$EXPECTED_VENV/bin/activate"
        echo "  ✓ venv 啟用: $EXPECTED_VENV"
    else
        echo "  ✗ 找不到 venv: $EXPECTED_VENV" >&2
        return 1 2>/dev/null || exit 1
    fi
else
    echo "  = venv 已啟用"
fi

# 2. Thor 環境（CUDA / LD_LIBRARY_PATH / TRITON_PTXAS_PATH）
if [ -z "$TRITON_PTXAS_PATH" ]; then
    THOR_SCRIPT="$PROJECT_ROOT/Isaac-GR00T/scripts/activate_thor.sh"
    if [ -f "$THOR_SCRIPT" ]; then
        # shellcheck disable=SC1091
        source "$THOR_SCRIPT"
    else
        echo "  ✗ 找不到 activate_thor.sh: $THOR_SCRIPT" >&2
    fi
else
    echo "  = Thor 環境已設定"
fi

# 3. Serial port 權限（只修復已存在的）
for port in /dev/ttyACM0 /dev/ttyACM1; do
    if [ -e "$port" ] && [ ! -w "$port" ]; then
        echo "  ⚠ $port 權限不足，sudo chmod 666 $port"
        sudo chmod 666 "$port"
    fi
done
