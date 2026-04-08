# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Fine-tuning NVIDIA Isaac GR00T N1.5 (Vision-Language-Action model) for the LeRobot SO-101 robotic arm, deployed on Jetson AGX Thor. The workflow is: collect teleoperation data (on a separate PC) → fine-tune GR00T → run inference on Thor to control the follower arm autonomously.

## Repository Structure

```
.
├── Isaac-GR00T/        # git submodule → github.com/NVIDIA/Isaac-GR00T
│   ├── .venv/          # GR00T Python 3.12 environment (created by install_deps.sh)
│   ├── pretrained/     # Downloaded GR00T-N1.5-3B model (5.1GB)
│   ├── scripts/        # Fine-tuning, inference, deployment scripts
│   └── examples/SO100/ # SO-100 arm config
├── SO-ARM100/          # git submodule → github.com/harry18456/SO-ARM100
│   ├── calibration/    # Follower + leader arm calibration JSONs
│   ├── STL/            # 3D printable parts
│   └── NOTE.md         # Motor setup and calibration notes
├── README.md           # Full setup and deployment guide (Traditional Chinese)
└── datasets/           # Training data (to be created, not in Isaac-GR00T/)
```

## Target Machine

Jetson AGX Thor — Ubuntu 24.04.3, CUDA 13.0, Driver 580.00, Python 3.12, JetPack 7.1 (L4T 38.4).

## Key Commands

### Activate GR00T environment (required every new shell)

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh
```

### Fine-tune on Thor

```bash
python scripts/gr00t_finetune.py \
  --dataset-path ../datasets/<dataset-name>/ \
  --num-gpus 1 \
  --output-dir ./so101-checkpoints \
  --max-steps 10000 \
  --data-config so100_dualcam \
  --video-backend torchvision_av \
  --base-model-path ./pretrained/GR00T-N1.5-3B
```

### Run inference (two terminals)

Terminal 1 — model server:
```bash
python scripts/inference_service.py --server \
  --model_path ./so101-checkpoints \
  --embodiment-tag new_embodiment \
  --data-config so100_dualcam \
  --denoising-steps 4
```

Terminal 2 — robot client:
```bash
python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --policy_host=0.0.0.0 \
  --lang_instruction="<task description>"
```

### Rebuild GR00T environment from scratch

```bash
cd Isaac-GR00T
bash scripts/deployment/thor/install_deps.sh
```

### Docker alternative

```bash
cd Isaac-GR00T/docker
bash build.sh --profile=thor
```

## Architecture Notes

- **Two separate environments**: GR00T inference runs in `Isaac-GR00T/.venv` (Python 3.12). Data collection uses LeRobot on a separate PC (Python 3.10, LeRobot 0.4.4 + feetech).
- **Submodules**: Both `Isaac-GR00T` and `SO-ARM100` are git submodules. After cloning this repo, run `git submodule update --init --recursive`.
- **Dataset path**: Store training data in `./datasets/`, not inside `Isaac-GR00T/` (which is a submodule).
- **Calibration data**: Arm calibration files live in `SO-ARM100/calibration/`. Both follower and leader arms are calibrated (6 joints each, STS3215 Feetech servos).
- **Camera constraint**: Two USB cameras must be on different USB hub chips on Thor. Wrist camera on USB-A, front camera on Type-C hub near QSFP28 port.
- **NVPL dependency**: PyTorch on Thor requires `libnvpl-lapack0` and `libnvpl-blas0` from the NVIDIA CUDA apt repo. `install_deps.sh` handles this automatically.
- **Inference is server-client**: `inference_service.py` runs the model on GPU, `eval_so100.py` handles robot control and camera input. They communicate via ZMQ.
