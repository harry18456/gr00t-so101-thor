# GR00T N1.5 x LeRobot SO-101 x Jetson AGX Thor

以 NVIDIA Isaac GR00T N1.5 微調 LeRobot SO-101 機械臂，並部署於 Jetson AGX Thor 的完整流程。

**參考來源**: https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/

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
| Isaac-GR00T | 已 clone |
| GR00T 環境 | 已安裝（Isaac-GR00T/.venv）|
| GR00T-N1.5-3B 模型 | 已下載（Isaac-GR00T/pretrained/GR00T-N1.5-3B，5.1GB）|
| SO-101 手臂 | 未接上 |
| USB 攝影機 | 未接上 |

---

## 目錄

1. [硬體需求](#1-硬體需求)
2. [基礎環境安裝](#2-基礎環境安裝)
3. [GR00T 環境安裝（Thor）](#3-gr00t-環境安裝thor)
4. [SO-101 手臂組裝與校正](#4-so-101-手臂組裝與校正)
5. [資料收集](#5-資料收集)
6. [模型微調（雲端 NVIDIA Brev）](#6-模型微調雲端-nvidia-brev)
7. [Jetson Thor 推論部署](#7-jetson-thor-推論部署)
8. [故障排除](#8-故障排除)

---

## 1. 硬體需求

### Jetson AGX Thor（本機）
- Blackwell GPU 架構，128GB 記憶體
- CUDA 13.0、Driver 580.00 ✓

### SO-101 手臂（待接上）
- 主臂（Master）+ 從臂（Follower）各一組
- 兩個 USB 攝影機（腕部攝影機 + 桌面攝影機）
- Serial 控制板（需 CH34x 驅動）

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

Isaac-GR00T 已 clone 至 `./Isaac-GR00T`。Thor 的安裝有兩種方式：

### 方式 A：Bare Metal 安裝（推薦）

直接在本機安裝，`install_deps.sh` 會自動處理：
- NVPL LAPACK/BLAS（PyTorch 依賴的 NVIDIA 線性代數函式庫）
- PyTorch 2.10.0、torchvision、flash-attn 等所有依賴
- 從 Jetson AI Lab PyPI 拉取 aarch64 專用 wheel
- 從原始碼 build torchcodec

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

### 方式 B：Docker 安裝（選用）

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

> **注意**：Docker 容器內環境變數已自動設定，不需要執行 `activate_thor.sh`。

---

## 4. SO-101 手臂組裝與校正

### 4.1 組裝

1. 參照 [LeRobot SO-100M wiki](https://wiki.seeedstudio.com/lerobot_so100m_new/) 設定每個關節的伺服馬達 ID
2. 組裝主臂與從臂
3. 確認 Serial 控制板已透過 USB 連接

確認 Serial 裝置（手臂接上後應出現）：

```bash
ls /dev/ttyACM*
# 預期出現 /dev/ttyACM0
```

### 4.2 校正手臂

> **重要**: 校正前不要接 USB 攝影機，避免裝置衝突。

手動移動每個關節到完整活動範圍，確認主臂與從臂姿態同步。

### 4.3 攝影機配置

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

參照 LeRobot SO-100M wiki 執行遙操作錄製：

```bash
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/so100.yaml \
  --fps 30 \
  --repo-id <your-hf-username>/<dataset-name> \
  --tags so100 tutorial \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 10 \
  --num-episodes 50
```

完成後若要上傳至雲端訓練：

```bash
scp -r ./record_2_cameras/ gr00t-trainer:/home/ubuntu/Datasets
```

---

## 6. 模型微調（雲端 NVIDIA Brev）

> 若要在 Thor 本機微調，跳至步驟 7.3。

### 6.1 建立雲端環境

1. 登入：https://login.brev.nvidia.com/signin
2. 建立 GPU instance
   - 需 Ampere 或更新架構（RTX A6000 / RTX 4090）
   - **V100（Volta）不支援 GR00T 訓練**
   - 預設需要約 25GB VRAM

### 6.2 安裝 GR00T（雲端環境）

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
uv venv --python 3.10
source .venv/bin/activate
uv pip install --upgrade setuptools
uv pip install -e .[base]
uv pip install --no-build-isolation flash-attn==2.7.1.post4
```

### 6.3 執行微調

```bash
python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/so101-table-cleanup/ \
  --num-gpus 1 \
  --output-dir ./so101-checkpoints \
  --max-steps 10000 \
  --data-config so100_dualcam \
  --video-backend torchvision_av
```

VRAM 不足（<25GB）時加上：

```bash
  --no-tune_diffusion_model
```

### 6.4 下載 Checkpoint

訓練完成後透過 Brev notebook 介面打包下載 `so101-checkpoints/`，放回 Thor。

---

## 7. Jetson Thor 推論部署

### 7.1 下載預訓練模型

```bash
# https://huggingface.co/nvidia/GR00T-N1.5-3B/tree/main
mkdir -p ./pretrained
# 使用 huggingface-cli 下載後放至 ./pretrained/GR00T-N1.5-3B/
pip install huggingface-hub
huggingface-cli download nvidia/GR00T-N1.5-3B --local-dir ./pretrained/GR00T-N1.5-3B
```

### 7.2 啟動環境

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh
```

### 7.3 （選用）在 Thor 本機微調

```bash
python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/so101-table-cleanup/ \
  --num-gpus 1 \
  --output-dir ./so101-checkpoints \
  --max-steps 10000 \
  --data-config so100_dualcam \
  --video-backend torchvision_av \
  --base-model-path ./pretrained/GR00T-N1.5-3B
```

### 7.4 啟動推論服務（Terminal 1）

```bash
python scripts/inference_service.py --server \
  --model_path ./so101-checkpoints \
  --embodiment-tag new_embodiment \
  --data-config so100_dualcam \
  --denoising-steps 4
```

### 7.5 啟動機械臂 Client（Terminal 2）

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh

python examples/SO-100/eval_lerobot.py \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --policy_host=0.0.0.0 \
  --lang_instruction="Grab pens and place into pen holder."
```

> **首次執行**會要求手臂關節校正，將每個關節移至完整活動範圍。

---

## 8. 故障排除

| 問題 | 解決方法 |
|------|---------|
| `libnvpl_lapack_lp64_gomp.so.0` 找不到 | 執行 `install_deps.sh`，會自動從 NVIDIA CUDA apt repo 安裝 |
| 找不到 Serial 裝置 `/dev/ttyACM0` | 安裝 CH34x 驅動；先接 Serial 板再接攝影機 |
| Type-C hub 無法辨識 | 改接靠近 QSFP28 的 Type-C 埠 |
| 兩台攝影機串流不穩 | 確認接在不同 USB hub chip |
| 雲端 GPU "not supported" | 換用 Ampere+ GPU；V100 不支援 |
| Docker `--runtime nvidia` 失敗 | 確認 nvidia-container-toolkit 已安裝 |
| VRAM 不足無法訓練 | 加上 `--no-tune_diffusion_model` 旗標 |
| torch.compile / Triton 失敗 | 確認已執行 `source scripts/activate_thor.sh` |

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
| Python（GR00T Thor 用）| 3.12 |
| PyTorch | 2.10.0 |
| flash-attn | 2.8.4 |
| GR00T N1.5 | 3B 參數 |

---

## 參考連結

- [Seeed Studio Wiki 原文](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1.5 Hugging Face](https://huggingface.co/nvidia/GR00T-N1.5-3B/tree/main)
- [GR00T 微調指南](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [CH341SER 驅動](https://github.com/juliagoda/CH341SER)
- [LeRobot SO-100M Wiki](https://wiki.seeedstudio.com/lerobot_so100m_new/)
