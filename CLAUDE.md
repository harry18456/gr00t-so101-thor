# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Fine-tuning NVIDIA Isaac GR00T N1.6 (Vision-Language-Action model) for the LeRobot SO-101 robotic arm, deployed on Jetson AGX Thor. The workflow is: collect teleoperation data → fine-tune GR00T → run inference on Thor to control the follower arm autonomously.

**Note**: The Seeed Studio wiki tutorial targets N1.5, but this project uses N1.6 (main branch). N1.6 uses state-relative action prediction (not absolute), a 2x larger DiT (32 layers), and different script paths.

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
├── CH341SER/           # git submodule → github.com/juliagoda/CH341SER (USB serial driver)
├── scripts/            # All operational scripts (see below)
├── datasets/           # Training data (gitignored, not inside Isaac-GR00T/)
├── so101-checkpoints/  # Fine-tuned model checkpoints (gitignored, ~22GB each)
├── camera_check/       # Camera angle reference images
└── README.md           # Full setup and deployment guide (Traditional Chinese)
```

## Target Machine

Jetson AGX Thor T5000 — Ubuntu 24.04.3, CUDA 13.0, Driver 580.00, Python 3.12.3, JetPack 7.1 (L4T 38.4), PyTorch 2.10.0, 128GB LPDDR5X, 2560 CUDA Cores, ~20 SM.

## Scripts (Primary Interface)

Most operations use scripts from the project root directory.

**Standard boot sequence** (after each reboot):
```bash
bash scripts/post_boot.sh            # Load ch34x driver + fix /dev/ttyACM* perms (once per boot)
source scripts/activate_env.sh       # Activate venv + Thor env (idempotent, each new shell)
bash scripts/preflight_check.sh      # Verify driver, motors, cameras, ports
```

`activate_env.sh` is idempotent and can be safely sourced from both the user's shell and other scripts. Prefer it over manually `cd Isaac-GR00T && source .venv/bin/activate && source scripts/activate_thor.sh`.

**Daily workflow:**
```bash
bash scripts/check_cameras.sh                    # Verify camera angles (saves to camera_check/)
bash scripts/teleop_test.sh                      # Test leader→follower without recording
bash scripts/record_data.sh [dataset_name]       # Record teleoperation data (default 50 episodes)
python3 scripts/convert_v3_to_v2.py <in> <out>   # Convert LeRobot v3.0 → v2.1 for GR00T
bash scripts/train.sh [dataset] [max_steps]      # Fine-tune GR00T (default: so101_pick_place_v2, 2000 steps)
bash scripts/start_server.sh [checkpoint]        # Start inference server (accepts number: 2000)
bash scripts/start_eval.sh [lang_instruction]    # Start eval client to control follower
```

**Server vs source**: use `bash scripts/start_server.sh` (not `source`) so the server runs in a subshell and Ctrl+C doesn't kill your current shell.

**Calibration & repair:**
```bash
bash scripts/calibrate.sh [both|leader|follower]
python3 scripts/fix_follower_wrist_offset.py      # Fix negative homing_offset PID runaway
python3 scripts/diagnose_wrist_roll.py            # Read wrist_roll hardware registers
python3 scripts/debug_teleop_wrist.py             # Single-axis wrist_roll teleop debug
```

## Key Commands (Manual)

When not using scripts, working directory is `Isaac-GR00T/`:

```bash
# Activate environment (required every new shell)
source .venv/bin/activate
source scripts/activate_thor.sh

# Quick validation (demo fine-tune, 5 steps)
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py --base_model_path ./pretrained/GR00T-N1.6-3B --dataset_path ./demo_data/cube_to_bowl_5 --modality_config_path examples/SO100/so100_config.py --embodiment_tag NEW_EMBODIMENT --num_gpus 1 --output_dir /tmp/so100_finetune_test --max_steps 5 --save_steps 5 --global_batch_size 2 --dataloader_num_workers 2

# Rebuild environment from scratch
bash scripts/deployment/thor/install_deps.sh
```

## Architecture Notes

### GR00T N1.6 Model Architecture (3B params)

```
GR00T N1.6
├── Backbone: Eagle3-VL 2B (frozen during fine-tune by default)
│   ├── Vision: SigLIP-2 (image → visual tokens)
│   └── Language: Qwen2 (task description → language tokens)
└── Action Head: DiT 32-layer + encoders/decoders (1.09B, main training target)
    └── Flow matching diffusion: noise → denoise → action trajectory
```

- Fine-tune trains the **DiT action head** + projectors. Backbone (SigLIP-2 + Qwen2) is frozen by default (`tune_llm=False`, `tune_visual=False`).
- Action prediction is **state-relative** (N1.6), not absolute positions.
- Action horizon: 16 steps. Max supported: 50.

### Embodiment Config Registration Flow (critical)

1. **Pretrained model** (`processor_config.json`) does NOT contain `new_embodiment`.
2. **During fine-tune**: `launch_finetune.py` imports `so100_config.py` → `register_modality_config()` → adds `new_embodiment` → saved into checkpoint's `processor_config.json`.
3. **During inference**: `Gr00tPolicy` loads config from checkpoint. If using pretrained (not fine-tuned), `KeyError: 'new_embodiment'`.

**Consequence**: Inference server MUST use a fine-tuned checkpoint, never the pretrained model.

### Inference Server

- `--modality_config_path` in `run_gr00t_server.py` is only for `ReplayPolicy`. `Gr00tPolicy` gets config from the checkpoint's `processor_config.json`.
- Server-client communicate via ZMQ on port 5555.

### lerobot Installation (critical — easy to break the environment)

**NEVER install lerobot with dependencies** — `uv pip install lerobot` will pull PyTorch 2.7.1+cpu from PyPI, overwriting the Jetson-specific 2.10.0+CUDA.

Correct install:
```bash
uv pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
uv pip install --no-deps draccus
uv pip install mergedeep "pyyaml-include<2" typing-inspect pyserial deepdiff orderly-set
uv pip install feetech-servo-sdk rerun-sdk
uv pip install numpy==1.26.4  # rerun-sdk may upgrade numpy, pin it back
```

If broken, see README Section 9.1 for full recovery.

## Hardware Mapping

- **Serial ports**: `/dev/ttyACM0` = leader, `/dev/ttyACM1` = follower. May swap after USB re-plug; verify with `scripts/preflight_check.sh`. After reboot: `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1`.
- **Cameras**: `video0` = wrist cam (pointing down at table), `video2` = front cam (viewing arm from front). Must be on different USB hub chips.
- **Arm calibration**: `~/.cache/huggingface/lerobot/calibration/`.

## Critical Pitfalls

- **Wrist roll (motor ID 5)**: STS3215 firmware has a PID runaway bug with negative `homing_offset`. After calibration, verify `wrist_roll.homing_offset >= 0`. If negative, run `python3 scripts/fix_follower_wrist_offset.py`. See README 9.2.
- **Dataset format**: lerobot 0.4.1 records v3.0 format, GR00T requires v2.1. Must run `convert_v3_to_v2.py`. Key differences: v3.0 has one parquet/mp4 for all episodes; v2.1 needs per-episode files + `episodes.jsonl` + `tasks.jsonl` + `modality.json`.
- **Recording args**: Use `--dataset.push_to_hub=false` (NOT `--dataset.local_files_only=true` which was removed). Set `DISPLAY=:1` for pynput on Thor.
- **N1.6 overfits faster**: Use `--color_jitter_params` and `--weight_decay` to regularize.
- **CH34x driver**: Must `make && sudo make load` in `CH341SER/` after each reboot.
- **torchcodec**: Must be built from source against system FFmpeg 7. `install_deps.sh` handles this.
- **N1.6 fine-tune script**: `gr00t/experiment/launch_finetune.py` (NOT `scripts/gr00t_finetune.py` which is N1.5).
- **Submodules**: Run `git submodule update --init --recursive` after cloning.
- **PEP 668**: Thor's Ubuntu 24.04 blocks `sudo pip3`. Use `uv tool install` for system-wide tools.
- **lerobot default**: `--dataset.num_episodes` defaults to 50, recording stops automatically.
- **Language instruction must match `tasks.jsonl` exactly**: N1.6 is highly sensitive to the language input. If `start_eval.sh` is run with a different task string than what's in `datasets/<name>/meta/tasks.jsonl`, the model will produce plausible-but-random motion. Always `cat tasks.jsonl` before eval.
- **Camera indices must be consistent between `record_data.sh` and `start_eval.sh`**: Both must use the same `WRIST_CAM`/`FRONT_CAM` indices so training and inference feed images into the same channels. Mismatch causes visually-correct-looking motion that totally fails the task. Current convention: `wrist=0, front=2`.
- **Bash scripts sourced vs executed**: scripts using `$(dirname "$0")` break when sourced (because `$0` is the shell name, not the script path). Use `${BASH_SOURCE[0]}` instead to make scripts work both ways.
