# GR00T N1.6 x LeRobot SO-101 x Jetson AGX Thor

以 NVIDIA Isaac GR00T N1.6 微調 LeRobot SO-101 機械臂，並部署於 Jetson AGX Thor 的完整流程。

**參考來源**: https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/

> **注意**：上述 wiki 是基於 N1.5 撰寫的，本專案已升級至 N1.6。差異說明見下方。

---

## 為什麼從 N1.5 升級到 N1.6

Seeed Studio wiki 教學原本使用 GR00T N1.5，但 Isaac-GR00T `main` branch 已更新至 N1.6。
經比較後決定直接使用 N1.6：

| | N1.5 | N1.6 |
|---|------|------|
| VLM 骨幹 | 舊版 | Cosmos-Reason-2B（更強）|
| DiT 層數 | 16 層 | **32 層（2 倍大）**|
| 動作表示 | 絕對關節角度 | **相對動作（state-relative）**|
| 收斂速度 | 較慢 | 更快（但更容易 overfit）|
| 微調腳本 | `scripts/gr00t_finetune.py` | `gr00t/experiment/launch_finetune.py` |
| 預訓練模型 | `nvidia/GR00T-N1.5-3B` | `nvidia/GR00T-N1.6-3B` |

**資料格式相容**：LeRobot v2.0 格式不變，已收集的資料集不需重錄。
差異在於 N1.6 使用相對動作預測，對應的 config 已在 `examples/SO100/so100_config.py` 更新。

> 若需使用 N1.5，切換至 `n1.5-release` branch：`git checkout n1.5-release`

---

## 目前環境狀態

| 項目 | 狀態 |
|------|------|
| OS | Ubuntu 24.04.3 LTS (kernel 6.8.12-tegra) |
| GPU | NVIDIA Thor |
| Driver | 580.00 |
| CUDA | 13.0 |
| Docker | 29.3.1（已安裝）|
| Python | 3.12.3（系統，Thor 正確版本）|
| uv | 0.11.2（已安裝）|
| Isaac-GR00T | git submodule（main branch = N1.6）|
| GR00T 環境 | 已安裝（Isaac-GR00T/.venv）|
| GR00T-N1.5-3B 模型 | 已下載（Isaac-GR00T/pretrained/GR00T-N1.5-3B，5.1GB）|
| GR00T-N1.6-3B 模型 | 下載中 |
| SO-101 手臂 | 已校準（校準檔在 SO-ARM100/calibration/）|
| USB 攝影機 | 未接上 |

---

## 目錄

1. [硬體需求](#1-硬體需求)
2. [基礎環境安裝](#2-基礎環境安裝)
3. [GR00T 環境安裝（Thor）](#3-gr00t-環境安裝thor)
4. [SO-101 手臂組裝與校正](#4-so-101-手臂組裝與校正)
5. [資料收集](#5-資料收集)
6. [模型微調](#6-模型微調)
7. [Jetson Thor 推論部署](#7-jetson-thor-推論部署)
8. [故障排除](#8-故障排除)

---

## 1. 硬體需求

### Jetson AGX Thor（本機）
- Blackwell GPU 架構，128GB 記憶體
- CUDA 13.0、Driver 580.00 ✓

### SO-101 手臂（已校準）
- 主臂（Leader）+ 從臂（Follower）各一組
- 兩個 USB 攝影機（腕部攝影機 + 桌面攝影機）
- Serial 控制板（需 CH34x 驅動）
- 校準資料：`SO-ARM100/calibration/`

---

## 2. 基礎環境安裝

> OS 已完成安裝（Ubuntu 24.04、CUDA 13.0），從這裡開始。

### 2.1 安裝工具套件

```bash
# 安裝 uv（若尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 安裝系統套件
sudo apt update
sudo apt install -y nvidia-jetpack firefox

# 安裝 jetson-stats（GPU 監控）
uv tool install jetson-stats
```

確認 GPU 狀態：

```bash
jtop
# 或
nvidia-smi
```

### 2.2 安裝 CH34x 驅動（手臂 Serial 連線）

```bash
git clone https://github.com/juliagoda/CH341SER.git
cd CH341SER
make
sudo make load
```

---

## 3. GR00T 環境安裝（Thor）

### 3.1 Clone repo（含 submodule）

```bash
git clone --recurse-submodules https://github.com/harry18456/gr00t-so101-thor.git
cd gr00t-so101-thor
```

### 3.2 方式 A：Bare Metal 安裝（推薦）

`install_deps.sh` 會自動處理 NVPL LAPACK/BLAS、PyTorch 2.10.0、flash-attn 等所有依賴：

```bash
cd Isaac-GR00T
bash scripts/deployment/thor/install_deps.sh
```

安裝完成後，每次開新 shell 需啟動環境：

```bash
source .venv/bin/activate
source scripts/activate_thor.sh
```

確認 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 預期輸出：2.10.0 True
```

### 3.3 方式 B：Docker 安裝（選用）

```bash
cd Isaac-GR00T/docker
bash build.sh --profile=thor
```

Build 完成後產生 `gr00t-thor` image，啟動容器：

```bash
sudo docker run --rm -it \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  --runtime nvidia \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /etc/X11:/etc/X11 \
  --device /dev/nvhost-vic \
  -v /dev:/dev \
  gr00t-thor
```

### 3.4 下載預訓練模型

```bash
cd Isaac-GR00T
source .venv/bin/activate
huggingface-cli download nvidia/GR00T-N1.6-3B --local-dir ./pretrained/GR00T-N1.6-3B
```

---

## 4. SO-101 手臂組裝與校正

> 手臂已在另一台 PC 上完成組裝與校準（2026-04-08），校準檔在 `SO-ARM100/calibration/`。

### 4.1 攝影機配置

接上兩個 USB 攝影機：
- **腕部攝影機**：接至 USB-A 埠
- **桌面攝影機**：接至靠近 QSFP28 的 USB Type-C 埠（需外接 USB hub）

> **關鍵限制**: 兩台攝影機必須接在不同的 USB hub chip，否則 Jetson 無法同時串流。

確認攝影機裝置號：

```bash
ls /dev/video*
# 通常 wrist=index 0, front=index 2
```

---

## 5. 資料收集

在另一台 PC 上用 LeRobot 遙操作錄製，完成後傳到 Thor：

```bash
# 從收集資料的 PC 傳到 Thor
scp -r <pc>:/path/to/dataset ~/gr00t-so101-thor/datasets/
```

資料集需為 LeRobot v2.0 格式。

---

## 6. 模型微調

### 6.1 在 Thor 本機微調（N1.6）

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh

python gr00t/experiment/launch_finetune.py \
  --base_model_path ./pretrained/GR00T-N1.6-3B \
  --dataset_path ../datasets/<your-dataset>/ \
  --modality_config_path examples/SO100/so100_config.py \
  --embodiment_tag NEW_EMBODIMENT \
  --num_gpus 1 \
  --output_dir ./so101-checkpoints \
  --max_steps 10000 \
  --save_steps 1000 \
  --learning_rate 1e-4 \
  --global_batch_size 32 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

### 6.2 在雲端微調（NVIDIA Brev）

1. 登入：https://login.brev.nvidia.com/signin
2. 建立 GPU instance（需 Ampere+，如 RTX A6000 / RTX 4090）
3. 安裝 GR00T 並執行同樣的微調指令（`--base_model_path nvidia/GR00T-N1.6-3B` 會自動從 HuggingFace 下載）
4. 訓練完成後將 checkpoint 傳回 Thor

---

## 7. Jetson Thor 推論部署

### 7.1 啟動環境

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh
```

### 7.2 啟動推論服務（Terminal 1）

```bash
python gr00t/eval/run_gr00t_server.py \
  --model_path ./so101-checkpoints \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path examples/SO100/so100_config.py
```

### 7.3 啟動機械臂 Client（Terminal 2）

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh

python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --policy_host=0.0.0.0 \
  --lang_instruction="<task description>"
```

---

## 8. 故障排除

| 問題 | 解決方法 |
|------|---------|
| `libnvpl_lapack_lp64_gomp.so.0` 找不到 | 執行 `install_deps.sh`，會自動從 NVIDIA CUDA apt repo 安裝 |
| 找不到 Serial 裝置 `/dev/ttyACM0` | 安裝 CH34x 驅動；先接 Serial 板再接攝影機 |
| Type-C hub 無法辨識 | 改接靠近 QSFP28 的 Type-C 埠 |
| 兩台攝影機串流不穩 | 確認接在不同 USB hub chip |
| Docker `--runtime nvidia` 失敗 | 確認 nvidia-container-toolkit 已安裝 |
| VRAM 不足無法訓練 | 減小 `--global_batch_size` |
| torch.compile / Triton 失敗 | 確認已執行 `source scripts/activate_thor.sh` |
| N1.6 過擬合 | 加強 `--color_jitter_params`、增加 `--weight_decay` |

---

## 版本資訊

| 元件 | 版本 |
|------|------|
| JetPack | 7.1 (L4T 38.4) |
| Ubuntu | 24.04.3 |
| Kernel | 6.8.12-tegra |
| CUDA | 13.0 |
| Driver | 580.00 |
| Docker | 29.3.1 |
| Python | 3.12 |
| PyTorch | 2.10.0 |
| flash-attn | 2.8.4 |
| GR00T | N1.6-3B |

---

## 參考連結

- [Seeed Studio Wiki 原文（N1.5）](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1.6 Research Blog](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- [GR00T N1.6 Hugging Face](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [GR00T N1.5 SO-101 微調指南](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [CH341SER 驅動](https://github.com/juliagoda/CH341SER)
- [LeRobot SO-100M Wiki](https://wiki.seeedstudio.com/lerobot_so100m_new/)
