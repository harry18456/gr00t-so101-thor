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
| GR00T-N1.6-3B 模型 | 已下載（Isaac-GR00T/pretrained/GR00T-N1.6-3B）|
| N1.6 推論 Server 測試 | ✓ 模型載入成功（DiT 1.09B params），pretrained 不含 SO100 config 屬正常（需 fine-tune 後才有）|
| N1.6 Fine-tune 測試 | ✓ 用 demo_data/cube_to_bowl_5 跑 5 步成功（train_loss=1.099），checkpoint 含 `new_embodiment` config |
| wandb | 已安裝（0.23.0）|
| so100_config.py | ✓ 已確認符合 SO-101（2 cam、6 joints + gripper、16 步 action horizon）|
| CH34x 驅動 | 已安裝（CH341SER submodule，kernel module 已載入）|
| SO-101 手臂 | 已校準（校準檔在 SO-ARM100/calibration/），尚未接上 |
| USB 攝影機 | 未接上 |

---

## 目錄

1. [硬體需求](#1-硬體需求)
2. [基礎環境安裝](#2-基礎環境安裝)
3. [GR00T 環境安裝（Thor）](#3-gr00t-環境安裝thor)
4. [驗證環境](#4-驗證環境)
5. [SO-101 手臂組裝與校正](#5-so-101-手臂組裝與校正)
6. [資料收集](#6-資料收集)
7. [模型微調](#7-模型微調)
8. [Jetson Thor 推論部署](#8-jetson-thor-推論部署)
9. [故障排除](#9-故障排除)

---

## 1. 硬體需求

### Jetson AGX Thor（本機）
- Blackwell GPU 架構，128GB 記憶體
- CUDA 13.0、Driver 580.00

### SO-101 手臂
- 主臂（Leader）+ 從臂（Follower）各一組
- 兩個 USB 攝影機（腕部攝影機 + 桌面攝影機）
- Serial 控制板（需 CH34x 驅動）

---

## 2. 基礎環境安裝

> 前提：OS 已完成安裝（Ubuntu 24.04、CUDA 13.0、JetPack 7.1）。

### 2.1 Clone 本專案（含所有 submodule）

```bash
git clone --recurse-submodules https://github.com/harry18456/gr00t-so101-thor.git
cd gr00t-so101-thor
```

本專案包含三個 git submodule：

| Submodule | 用途 |
|-----------|------|
| `Isaac-GR00T/` | NVIDIA GR00T N1.6 核心程式庫（fork 自 NVIDIA/Isaac-GR00T）|
| `SO-ARM100/` | SO-101 手臂校準資料與筆記 |
| `CH341SER/` | CH34x USB Serial 驅動原始碼 |

### 2.2 安裝系統工具

```bash
# 安裝 uv（Python 套件管理）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 安裝系統套件
sudo apt update
sudo apt install -y nvidia-jetpack firefox

# 安裝 jetson-stats（GPU 監控）
# 注意：Thor 上的 Ubuntu 24.04 啟用了 PEP 668，不能用 sudo pip3 安裝系統套件
uv tool install jetson-stats
```

確認 GPU 狀態：

```bash
nvidia-smi
# 應顯示 Driver 580.00、CUDA 13.0

jtop
# 即時 GPU/CPU/記憶體監控
```

### 2.3 編譯 CH34x 驅動（手臂 Serial 連線）

> CH341SER 已是本 repo 的 git submodule，clone 時會一起拉下來，不需另外 clone。

```bash
cd CH341SER
make
sudo make load
```

確認驅動已載入：

```bash
lsmod | grep ch34
# 預期輸出：
# ch34x                  16384  0
# usbserial              32768  1 ch34x
```

> **注意**：`sudo make load` 只在本次開機有效。重開機後需重新執行 `sudo make load`，或將 `ch34x` 加入 `/etc/modules-load.d/` 讓它開機自動載入。

---

## 3. GR00T 環境安裝（Thor）

### 3.1 方式 A：Bare Metal 安裝（推薦）

`install_deps.sh` 會自動處理以下所有依賴，不需手動安裝：

- NVPL LAPACK/BLAS（`libnvpl-lapack0`、`libnvpl-blas0`，從 NVIDIA CUDA apt repo 安裝）
- Python 3.12 虛擬環境（`.venv/`）
- PyTorch 2.10.0（從 Jetson AI Lab PyPI 源：`pypi.jetson-ai-lab.io`）
- flash-attn 2.8.4、triton 3.5.0
- torchcodec v0.10.0（從原始碼編譯）

```bash
cd Isaac-GR00T
bash scripts/deployment/thor/install_deps.sh
```

安裝完成後，**每次開新 shell** 都需要啟動環境：

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh
```

`activate_thor.sh` 會設定 `TRITON_PTXAS_PATH`、`CUDA_HOME`、`LD_LIBRARY_PATH` 等環境變數，不執行的話 torch.compile 和 Triton 會失敗。

確認 PyTorch 安裝正確：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 預期輸出：2.10.0 True
```

### 3.2 方式 B：Docker 安裝（選用）

```bash
cd Isaac-GR00T/docker
bash build.sh --profile=thor
```

> **注意**：不要用第三方預建 Docker image（如 `johnnync/isaac-gr00t:*`），許多已過期或不存在。務必用官方 `build.sh` 本地建構。

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

### 3.3 下載預訓練模型

模型檔約 5GB，下載到 `Isaac-GR00T/pretrained/`（已加入 `.gitignore`，不會被 commit）：

```bash
cd Isaac-GR00T
source .venv/bin/activate
huggingface-cli download nvidia/GR00T-N1.6-3B --local-dir ./pretrained/GR00T-N1.6-3B
```

確認下載完成：

```bash
ls pretrained/GR00T-N1.6-3B/
# 應包含：config.json, model-00001-of-00002.safetensors, model-00002-of-00002.safetensors,
#          processor_config.json, statistics.json, embodiment_id.json, ...
```

---

## 4. 驗證環境

在正式收集資料和訓練之前，用內建的 demo 資料集驗證整個 pipeline 是否正常。

### 4.1 驗證 fine-tune pipeline

用 `demo_data/cube_to_bowl_5`（隨 Isaac-GR00T repo 附帶）跑 5 步測試：

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base_model_path ./pretrained/GR00T-N1.6-3B \
  --dataset_path ./demo_data/cube_to_bowl_5 \
  --modality_config_path examples/SO100/so100_config.py \
  --embodiment_tag NEW_EMBODIMENT \
  --num_gpus 1 \
  --output_dir /tmp/so100_finetune_test \
  --max_steps 5 \
  --save_steps 5 \
  --learning_rate 1e-4 \
  --global_batch_size 2 \
  --dataloader_num_workers 2
```

預期結果（約 1-2 分鐘）：

```
Training completed!
{'train_runtime': ~65s, 'train_samples_per_second': 0.154, 'train_loss': ~1.099}
```

Checkpoint 會存在 `/tmp/so100_finetune_test/checkpoint-5/`。

### 4.2 驗證 checkpoint 包含 SO100 config

Fine-tune 後的 checkpoint 應包含 `new_embodiment`（pretrained 模型不含，這是正常的）：

```bash
python -c "import json; d=json.load(open('/tmp/so100_finetune_test/checkpoint-5/processor_config.json')); print(list(d['processor_kwargs']['modality_configs'].keys()))"
# 預期輸出：['behavior_r1_pro', 'gr1', 'robocasa_panda_omron', 'new_embodiment']
```

> **為什麼 pretrained 模型不含 `new_embodiment`？**
>
> Pretrained GR00T-N1.6-3B 的 `processor_config.json` 只包含 NVIDIA 預訓練時使用的 3 個 embodiment（`behavior_r1_pro`、`gr1`、`robocasa_panda_omron`）。
> `new_embodiment` 是在 fine-tune 過程中，透過 `so100_config.py` 的 `register_modality_config()` 註冊，並由 `processor.save_pretrained()` 寫入 checkpoint。
> 因此，用 pretrained 模型直接啟動推論 server 搭配 `--embodiment_tag NEW_EMBODIMENT` 會得到 `KeyError: 'new_embodiment'`，必須使用 fine-tuned checkpoint。

### 4.3 驗證推論 server 載入（用測試 checkpoint）

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh

python gr00t/eval/run_gr00t_server.py \
  --model_path /tmp/so100_finetune_test/checkpoint-5 \
  --embodiment_tag NEW_EMBODIMENT
```

預期輸出：

```
Starting GR00T inference server...
  Embodiment tag: EmbodimentTag.NEW_EMBODIMENT
  ...
```

Server 啟動後按 `Ctrl+C` 結束。

### 4.4 so100_config.py 驗證

`examples/SO100/so100_config.py` 定義了 SO-101 的 modality 配置：

| Modality | 設定 | 說明 |
|----------|------|------|
| video | `front`, `wrist` | 兩個 USB 攝影機 |
| state | `single_arm`, `gripper` | 6 軸手臂 + 夾爪 |
| action | 16 步 horizon, `RELATIVE` + `ABSOLUTE` | 手臂用相對動作、夾爪用絕對動作 |
| language | `annotation.human.task_description` | 自然語言任務描述 |

這與 SO-101 的硬體配置（2 cam、6 joints + 1 gripper）完全吻合，不需修改。

---

## 5. SO-101 手臂組裝與校正

> 手臂已在另一台 PC 上完成組裝與校準（2026-04-08），校準檔在 `SO-ARM100/calibration/`。

校準檔內容（follower 和 leader 各一份）：

```
SO-ARM100/calibration/
├── follower/calibration.json   # 從臂校準資料（6 joints, ID 1-6, STS3215 servo）
└── leader/calibration.json     # 主臂校準資料（6 joints, ID 1-6, STS3215 servo）
```

### 5.1 攝影機配置

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

## 6. 資料收集

在另一台 PC 上用 LeRobot 遙操作錄製（需安裝 LeRobot 0.4.4 + feetech 驅動、Python 3.10），完成後傳到 Thor：

```bash
# 從收集資料的 PC 傳到 Thor
scp -r <pc>:/path/to/dataset ~/gr00t-so101-thor/datasets/
```

> **注意**：資料集放在 `datasets/` 目錄，不要放在 `Isaac-GR00T/` 內（那是 submodule）。

資料集需為 **LeRobot v2.0 格式**（又稱 GR00T-flavored LeRobot v2）。詳細格式說明見 `Isaac-GR00T/getting_started/data_preparation.md`。

---

## 7. 模型微調

### 7.1 在 Thor 本機微調（N1.6）

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base_model_path ./pretrained/GR00T-N1.6-3B \
  --dataset_path ../datasets/<your-dataset>/ \
  --modality_config_path examples/SO100/so100_config.py \
  --embodiment_tag NEW_EMBODIMENT \
  --num_gpus 1 \
  --output_dir ./so101-checkpoints \
  --max_steps 10000 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --learning_rate 1e-4 \
  --global_batch_size 32 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader_num_workers 4 \
  --use_wandb
```

#### 參數說明

| 參數 | 說明 |
|------|------|
| `--base_model_path` | 預訓練模型路徑（本地或 HuggingFace ID）|
| `--dataset_path` | 訓練資料集路徑（LeRobot v2.0 格式）|
| `--modality_config_path` | SO100 modality 配置（定義 camera、state、action 格式）|
| `--embodiment_tag` | 必須用 `NEW_EMBODIMENT`（自定義機器人）|
| `--max_steps` | 訓練步數（建議 5000-10000）|
| `--save_steps` | 每 N 步存一次 checkpoint |
| `--global_batch_size` | 全域 batch size（VRAM 不足時調小）|
| `--color_jitter_params` | 資料增強（N1.6 容易 overfit，建議開啟）|
| `--weight_decay` | 正則化強度（N1.6 建議 1e-5 以上）|
| `--use_wandb` | 啟用 Weights & Biases 訓練監控（需先 `wandb login`）|

#### wandb 設定（選用）

```bash
wandb login
# 輸入你的 API key（從 https://wandb.ai/authorize 取得）
```

### 7.2 在雲端微調（NVIDIA Brev）

1. 登入：https://login.brev.nvidia.com/signin
2. 建立 GPU instance（需 Ampere+，如 RTX A6000 / RTX 4090）
3. 安裝 GR00T 並執行同樣的微調指令（`--base_model_path nvidia/GR00T-N1.6-3B` 會自動從 HuggingFace 下載）
4. 訓練完成後將 checkpoint 傳回 Thor

### 7.3 Open Loop 評估（選用）

Fine-tune 完成後，可用 open loop evaluation 評估模型品質：

```bash
python gr00t/eval/open_loop_eval.py \
  --dataset-path ../datasets/<your-dataset>/ \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path ./so101-checkpoints/checkpoint-<N> \
  --traj-ids 0 \
  --action-horizon 16 \
  --steps 400 \
  --modality-keys single_arm gripper
```

會產生預測動作 vs 真實軌跡的比較圖。

---

## 8. Jetson Thor 推論部署

> **前提**：必須使用 fine-tuned checkpoint，不能用 pretrained 模型（見[驗證環境](#42-驗證-checkpoint-包含-so100-config)說明）。

### 8.1 啟動推論服務（Terminal 1）

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh

python gr00t/eval/run_gr00t_server.py \
  --model_path ./so101-checkpoints/checkpoint-<N> \
  --embodiment_tag NEW_EMBODIMENT
```

> **注意**：`--modality_config_path` 在推論 server 中**不會**傳給 `Gr00tPolicy`，它只對 `ReplayPolicy`（dataset replay 模式）有效。Fine-tuned checkpoint 的 `processor_config.json` 已包含 `new_embodiment` 的 modality config，所以不需要額外指定。

Server 預設在 port 5555 上監聽，透過 ZMQ 與 client 通訊。

### 8.2 啟動機械臂 Client（Terminal 2）

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

推論架構為 server-client 分離：
- `run_gr00t_server.py`：在 GPU 上跑模型推論
- `eval_so100.py`：負責攝影機影像擷取、機械臂控制，透過 ZMQ 將觀測資料送給 server 並接收動作指令

---

## 9. 故障排除

| 問題 | 解決方法 |
|------|---------|
| `libnvpl_lapack_lp64_gomp.so.0` 找不到 | 執行 `install_deps.sh`，會自動從 NVIDIA CUDA apt repo 安裝 NVPL |
| `sudo pip3 install` 報 `externally-managed-environment` | Ubuntu 24.04 啟用 PEP 668，改用 `uv tool install` 或在 venv 內安裝 |
| 找不到 Serial 裝置 `/dev/ttyACM0` | 確認 CH34x 驅動已載入（`lsmod \| grep ch34`）；先接 Serial 板再接攝影機 |
| Type-C hub 無法辨識 | 改接靠近 QSFP28 的 Type-C 埠 |
| 兩台攝影機串流不穩 | 確認接在不同 USB hub chip |
| Docker image pull 失敗 | 不要用第三方 image，用 `bash build.sh --profile=thor` 本地建構 |
| Docker `--runtime nvidia` 失敗 | 確認 nvidia-container-toolkit 已安裝 |
| VRAM 不足無法訓練 | 減小 `--global_batch_size` |
| torch.compile / Triton 失敗 | 確認已執行 `source scripts/activate_thor.sh`（設定 CUDA_HOME 等）|
| N1.6 過擬合 | 加強 `--color_jitter_params`、增加 `--weight_decay` |
| `KeyError: 'new_embodiment'`（推論 server）| 必須用 fine-tuned checkpoint，不能用 pretrained 模型。Pretrained 的 `processor_config.json` 只含 `behavior_r1_pro`、`gr1`、`robocasa_panda_omron`，fine-tune 時 `so100_config.py` 的 config 才會被寫入 checkpoint |

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
| Python | 3.12.3 |
| uv | 0.11.2 |
| PyTorch | 2.10.0 |
| flash-attn | 2.8.4 |
| triton | 3.5.0 |
| wandb | 0.23.0 |
| GR00T | N1.6-3B |

---

## 參考連結

- [Seeed Studio Wiki 原文（N1.5）](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1.6 Research Blog](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- [GR00T N1.6 Hugging Face](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [GR00T N1.5 SO-101 微調指南](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [Isaac-GR00T Fine-tune 新 Embodiment 教學](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md)
- [Isaac-GR00T 資料準備指南](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)
- [CH341SER 驅動](https://github.com/juliagoda/CH341SER)
- [LeRobot SO-100M Wiki](https://wiki.seeedstudio.com/lerobot_so100m_new/)
