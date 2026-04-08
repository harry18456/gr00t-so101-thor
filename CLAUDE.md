# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Fine-tuning NVIDIA Isaac GR00T N1.6 (Vision-Language-Action model) for the LeRobot SO-101 robotic arm, deployed on Jetson AGX Thor. The workflow is: collect teleoperation data (on a separate PC) → fine-tune GR00T → run inference on Thor to control the follower arm autonomously.

**Note**: The Seeed Studio wiki tutorial targets N1.5, but this project uses N1.6 (main branch). N1.6 uses state-relative action prediction (not absolute), a 2x larger DiT, and different script paths. Existing LeRobot v2.0 datasets are compatible.

## Repository Structure

```
.
├── Isaac-GR00T/        # git submodule → github.com/harry18456/Isaac-GR00T (fork of NVIDIA)
│   ├── .venv/          # GR00T Python 3.12 environment (created by install_deps.sh)
│   ├── pretrained/     # Downloaded models (GR00T-N1.5-3B, GR00T-N1.6-3B)
│   ├── gr00t/          # Core library, training, eval scripts
│   └── examples/SO100/ # SO-100 arm config and finetune script
├── SO-ARM100/          # git submodule → github.com/harry18456/SO-ARM100
│   ├── calibration/    # Follower + leader arm calibration JSONs
│   └── NOTE.md         # Motor setup and calibration notes
├── datasets/           # Training data (to be created, not in Isaac-GR00T/)
└── README.md           # Full setup and deployment guide (Traditional Chinese)
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

### Fine-tune on Thor (N1.6)

```bash
python gr00t/experiment/launch_finetune.py \
  --base_model_path ./pretrained/GR00T-N1.6-3B \
  --dataset_path ../datasets/<dataset-name>/ \
  --modality_config_path examples/SO100/so100_config.py \
  --embodiment_tag NEW_EMBODIMENT \
  --num_gpus 1 \
  --output_dir ./so101-checkpoints \
  --max_steps 10000 \
  --learning_rate 1e-4 \
  --global_batch_size 32
```

### Run inference (two terminals)

Terminal 1 — model server:
```bash
python gr00t/eval/run_gr00t_server.py \
  --model_path ./so101-checkpoints \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path examples/SO100/so100_config.py
```

Terminal 2 — robot client:
```bash
python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
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

- **N1.6 vs N1.5**: N1.6 uses relative action prediction, larger DiT (32 layers), Cosmos-Reason-2B VLM. Fine-tune script is `gr00t/experiment/launch_finetune.py` (not `scripts/gr00t_finetune.py`). For N1.5, checkout `n1.5-release` branch.
- **Two separate environments**: GR00T inference runs in `Isaac-GR00T/.venv` (Python 3.12). Data collection uses LeRobot on a separate PC (Python 3.10, LeRobot 0.4.4 + feetech).
- **Submodules**: Both `Isaac-GR00T` and `SO-ARM100` are git submodules. After cloning this repo, run `git submodule update --init --recursive`.
- **Dataset path**: Store training data in `./datasets/`, not inside `Isaac-GR00T/` (which is a submodule).
- **Calibration data**: Arm calibration files live in `SO-ARM100/calibration/`. Both follower and leader arms are calibrated (6 joints each, STS3215 Feetech servos).
- **Camera constraint**: Two USB cameras must be on different USB hub chips on Thor. Wrist camera on USB-A, front camera on Type-C hub near QSFP28 port.
- **NVPL dependency**: PyTorch on Thor requires `libnvpl-lapack0` and `libnvpl-blas0`. `install_deps.sh` handles this automatically.
- **Inference is server-client**: `run_gr00t_server.py` runs the model on GPU, `eval_so100.py` handles robot control and camera input. They communicate via ZMQ.
