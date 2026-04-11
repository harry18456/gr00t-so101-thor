# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Fine-tuning NVIDIA Isaac GR00T N1.6 (Vision-Language-Action model) for the LeRobot SO-101 robotic arm, deployed on Jetson AGX Thor. The workflow is: collect teleoperation data (on a separate PC) → fine-tune GR00T → run inference on Thor to control the follower arm autonomously.

**Note**: The Seeed Studio wiki tutorial targets N1.5, but this project uses N1.6 (main branch). N1.6 uses state-relative action prediction (not absolute), a 2x larger DiT (32 layers), and different script paths. Existing LeRobot v2.0 datasets are compatible.

## Repository Structure

```
.
├── Isaac-GR00T/        # git submodule → github.com/harry18456/Isaac-GR00T (fork of NVIDIA)
│   ├── .venv/          # GR00T Python 3.12 environment (created by install_deps.sh)
│   ├── pretrained/     # Downloaded models (GR00T-N1.6-3B) — gitignored
│   ├── demo_data/      # Built-in demo datasets (cube_to_bowl_5) for pipeline testing
│   ├── gr00t/          # Core library, training, eval scripts
│   └── examples/SO100/ # SO-100 arm config (so100_config.py) and finetune script
├── SO-ARM100/          # git submodule → github.com/harry18456/SO-ARM100
│   └── calibration/    # Follower + leader arm calibration JSONs (6 joints each)
├── CH341SER/           # git submodule → github.com/juliagoda/CH341SER (USB serial driver)
├── datasets/           # Training data (to be created, not inside Isaac-GR00T/)
└── README.md           # Full setup and deployment guide (Traditional Chinese)
```

## Target Machine

Jetson AGX Thor — Ubuntu 24.04.3, CUDA 13.0, Driver 580.00, Python 3.12.3, JetPack 7.1 (L4T 38.4), PyTorch 2.10.0.

## Key Commands

All commands below assume working directory is `Isaac-GR00T/`.

### Activate GR00T environment (required every new shell)

```bash
source .venv/bin/activate
source scripts/activate_thor.sh
```

### Quick validation (demo fine-tune, 5 steps)

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py --base_model_path ./pretrained/GR00T-N1.6-3B --dataset_path ./demo_data/cube_to_bowl_5 --modality_config_path examples/SO100/so100_config.py --embodiment_tag NEW_EMBODIMENT --num_gpus 1 --output_dir /tmp/so100_finetune_test --max_steps 5 --save_steps 5 --global_batch_size 2 --dataloader_num_workers 2
```

### Fine-tune on Thor (N1.6)

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py --base_model_path ./pretrained/GR00T-N1.6-3B --dataset_path ../datasets/<dataset-name>/ --modality_config_path examples/SO100/so100_config.py --embodiment_tag NEW_EMBODIMENT --num_gpus 1 --output_dir ./so101-checkpoints --max_steps 10000 --save_steps 1000 --learning_rate 1e-4 --global_batch_size 32 --warmup_ratio 0.05 --weight_decay 1e-5 --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader_num_workers 4
```

### Run inference (two terminals, requires fine-tuned checkpoint)

Terminal 1 — model server:
```bash
python gr00t/eval/run_gr00t_server.py --model_path ./so101-checkpoints/checkpoint-<N> --embodiment_tag NEW_EMBODIMENT
```

Terminal 2 — robot client:
```bash
python gr00t/eval/real_robot/SO100/eval_so100.py --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" --policy_host=0.0.0.0 --lang_instruction="<task description>"
```

### Rebuild GR00T environment from scratch

```bash
bash scripts/deployment/thor/install_deps.sh
```

## Architecture Notes

### Embodiment Config Registration Flow (critical)

This is the most important non-obvious architecture detail:

1. **Pretrained model** (`processor_config.json`) only contains NVIDIA's built-in embodiments: `behavior_r1_pro`, `gr1`, `robocasa_panda_omron`. It does NOT contain `new_embodiment`.
2. **During fine-tune**: `launch_finetune.py` imports `so100_config.py` → calls `register_modality_config()` → adds `new_embodiment` to the global `MODALITY_CONFIGS` dict → processor is built with this config → `processor.save_pretrained()` writes it into the checkpoint's `processor_config.json`.
3. **During inference**: `Gr00tPolicy.__init__` loads `processor_config.json` from the checkpoint and looks up `self.embodiment_tag.value` in it. If the checkpoint is pretrained (not fine-tuned), `KeyError: 'new_embodiment'` will occur.

**Consequence**: Inference server MUST use a fine-tuned checkpoint, never the pretrained model, when using `--embodiment_tag NEW_EMBODIMENT`.

### Inference Server

- `--modality_config_path` in `run_gr00t_server.py` is **only** used for `ReplayPolicy` (when `--dataset_path` is given). It is NOT passed to `Gr00tPolicy`. The modality config comes from the checkpoint's `processor_config.json` instead.
- Server-client communicate via ZMQ on port 5555.

### lerobot Installation (critical — easy to break the environment)

`eval_so100.py` requires `lerobot` (0.4.1) and `draccus`, but they are NOT in the GR00T pyproject.toml. They have their own pyproject.toml at `gr00t/eval/real_robot/SO100/pyproject.toml`.

**NEVER install lerobot with dependencies** — `uv pip install lerobot` will pull PyTorch 2.7.1+cpu from PyPI, overwriting the Jetson-specific 2.10.0+CUDA. It also replaces the source-built torchcodec with a prebuilt version that lacks FFmpeg 7 support.

Correct install:
```bash
uv pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
uv pip install --no-deps draccus
uv pip install mergedeep "pyyaml-include<2" typing-inspect pyserial deepdiff orderly-set
uv pip install feetech-servo-sdk rerun-sdk
uv pip install numpy==1.26.4  # rerun-sdk may upgrade numpy, pin it back
```

If the environment is already broken, see README Section 9.1 for full recovery steps (uv sync → rebuild torchcodec from source → reinstall lerobot with --no-deps).

### Other Notes

- **N1.6 vs N1.5**: Fine-tune script is `gr00t/experiment/launch_finetune.py` (NOT `scripts/gr00t_finetune.py`). For N1.5, checkout `n1.5-release` branch.
- **Two separate environments**: GR00T inference runs in `Isaac-GR00T/.venv` (Python 3.12). Data collection uses LeRobot on a separate PC (Python 3.10).
- **Arm port mapping on Thor**: leader=`/dev/ttyACM1`, follower=`/dev/ttyACM0`. After reboot, run `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1`.
- **Arm calibration on Thor**: Stored in `~/.cache/huggingface/lerobot/calibration/`. Calibration also exists in `SO-ARM100/calibration/` (from separate PC).
- **Submodules**: `Isaac-GR00T`, `SO-ARM100`, and `CH341SER` are git submodules. After cloning, run `git submodule update --init --recursive`.
- **Dataset path**: Store training data in `./datasets/`, not inside `Isaac-GR00T/` (which is a submodule).
- **Camera constraint**: Two USB cameras must be on different USB hub chips on Thor.
- **NVPL dependency**: PyTorch on Thor requires `libnvpl-lapack0` and `libnvpl-blas0`. `install_deps.sh` handles this via the NVIDIA CUDA apt repo for `ubuntu2404/sbsa`.
- **PEP 668**: Thor's Ubuntu 24.04 blocks `sudo pip3`. Use `uv tool install` for system-wide tools (e.g., `jetson-stats`).
- **CH34x driver**: Must `make && sudo make load` in `CH341SER/` after each reboot (or add to `/etc/modules-load.d/`).
- **N1.6 overfits faster**: Use `--color_jitter_params` and `--weight_decay` to regularize.
- **torchcodec**: Must be built from source against system FFmpeg 7. The prebuilt wheel from PyPI/Jetson AI Lab doesn't work. `install_deps.sh` handles this.
