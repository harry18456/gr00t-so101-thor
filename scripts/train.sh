#!/bin/bash
# 微調 GR00T N1.6
# 用法: bash scripts/train.sh [dataset_name] [max_steps]
# 範例: bash scripts/train.sh so101_pick_place_v2 2000
#       bash scripts/train.sh                          ← 預設 so101_pick_place_v2, 2000 步
#
# 建議用 screen 執行：
#   screen -S train
#   bash scripts/train.sh
#   Ctrl+A D 離開，screen -r train 回來看

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATASET="${1:-so101_pick_place_v2}"
MAX_STEPS="${2:-2000}"
DATASET_PATH="$PROJECT_DIR/datasets/$DATASET"

if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ 資料集不存在: $DATASET_PATH"
    echo "   可用的資料集:"
    ls -d "$PROJECT_DIR/datasets"/*_v2 2>/dev/null | while read d; do echo "     $(basename $d)"; done
    exit 1
fi

# 清除舊 checkpoint（可選）
if [ -d "$PROJECT_DIR/so101-checkpoints" ]; then
    echo "⚠ 已存在 checkpoints 目錄: $PROJECT_DIR/so101-checkpoints"
    read -p "  要刪除舊 checkpoints 嗎？(y/N) " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_DIR/so101-checkpoints"
        echo "  已刪除"
    fi
fi

echo "=== GR00T N1.6 微調 ==="
echo "  資料集: $DATASET_PATH"
echo "  訓練步數: $MAX_STEPS"
echo "  Checkpoint 輸出: $PROJECT_DIR/so101-checkpoints"
echo ""

cd "$PROJECT_DIR/Isaac-GR00T"
source .venv/bin/activate
source scripts/activate_thor.sh

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base_model_path ./pretrained/GR00T-N1.6-3B \
  --dataset_path "$DATASET_PATH/" \
  --modality_config_path examples/SO100/so100_config.py \
  --embodiment_tag NEW_EMBODIMENT \
  --num_gpus 1 \
  --output_dir "$PROJECT_DIR/so101-checkpoints" \
  --max_steps "$MAX_STEPS" \
  --save_steps 500 \
  --learning_rate 1e-4 \
  --global_batch_size 32 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader_num_workers 4
