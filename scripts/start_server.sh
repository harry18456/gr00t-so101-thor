#!/bin/bash
# 啟動 GR00T 推論 Server
# 用法: bash scripts/start_server.sh [checkpoint 編號或路徑]
# 範例: bash scripts/start_server.sh 200          ← checkpoint-200
#       bash scripts/start_server.sh 1000         ← checkpoint-1000
#       bash scripts/start_server.sh              ← 自動用最新 checkpoint

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CKPT_DIR="$PROJECT_DIR/so101-checkpoints"

ARG="${1:-}"

if [ -z "$ARG" ]; then
    # 沒指定，自動找最新
    CHECKPOINT=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "❌ $CKPT_DIR 中沒有 checkpoint"
        exit 1
    fi
elif [[ "$ARG" =~ ^[0-9]+$ ]]; then
    # 只傳數字，當作 checkpoint 編號
    CHECKPOINT="$CKPT_DIR/checkpoint-$ARG"
elif [ -d "$ARG" ]; then
    # 傳的是存在的目錄路徑
    CHECKPOINT="$(cd "$ARG" && pwd)"
else
    # 可能是相對路徑，先轉絕對
    CHECKPOINT="$PROJECT_DIR/$ARG"
    if [ ! -d "$CHECKPOINT" ]; then
        CHECKPOINT="$ARG"
    fi
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Checkpoint 不存在: $CHECKPOINT"
    echo "   可用的 checkpoints:"
    ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | while read d; do echo "     $(basename $d)"; done
    exit 1
fi

echo "=== GR00T 推論 Server ==="
echo "  Checkpoint: $CHECKPOINT"
echo "  Port: 5555"
echo "  Ctrl+C 停止"
echo ""

# 檢查 port 5555 是否被佔用
if lsof -i :5555 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "⚠ Port 5555 已被佔用，嘗試釋放..."
    kill $(lsof -t -i :5555) 2>/dev/null
    sleep 1
fi

cd "$PROJECT_DIR/Isaac-GR00T"
source .venv/bin/activate
source scripts/activate_thor.sh

python gr00t/eval/run_gr00t_server.py \
  --model_path "$CHECKPOINT" \
  --embodiment_tag NEW_EMBODIMENT
