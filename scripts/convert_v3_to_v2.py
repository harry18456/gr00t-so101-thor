"""
Convert LeRobot v3.0 dataset to v2.1 format for GR00T fine-tuning.

v3.0: all episodes in one parquet/mp4, meta in parquet
v2.1: per-episode parquet/mp4, meta in jsonl

Usage:
  python3 scripts/convert_v3_to_v2.py <input_dir> <output_dir>
  python3 scripts/convert_v3_to_v2.py ../datasets/so101_pick_place ../datasets/so101_pick_place_v2
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
    sys.exit(1)

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

print(f"Converting {input_dir} → {output_dir}")
print()

# Read v3.0 metadata
info = json.load(open(input_dir / "meta" / "info.json"))
episodes_df = pq.read_table(input_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
tasks_df = pq.read_table(input_dir / "meta" / "tasks.parquet").to_pandas()

total_episodes = info["total_episodes"]
total_frames = info["total_frames"]
fps = info["fps"]

print(f"Episodes: {total_episodes}, Frames: {total_frames}, FPS: {fps}")

# Read all data
data_df = pq.read_table(input_dir / "data" / "chunk-000" / "file-000.parquet").to_pandas()
print(f"Data columns: {list(data_df.columns)}")
print()

# Create output directory structure
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "meta").mkdir(exist_ok=True)
(output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

# Detect video keys
video_keys = [k for k in info["features"] if info["features"][k].get("dtype") == "video"]
print(f"Video keys: {video_keys}")

for vk in video_keys:
    (output_dir / "videos" / "chunk-000" / vk).mkdir(parents=True, exist_ok=True)

# 1. Split data into per-episode parquet files
print("\n=== Splitting data into per-episode parquets ===")
for ep_idx in range(total_episodes):
    ep_data = data_df[data_df["episode_index"] == ep_idx].copy()
    ep_path = output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    ep_data.to_parquet(ep_path, index=False)
    print(f"  Episode {ep_idx}: {len(ep_data)} frames → {ep_path.name}")

# 2. Split videos into per-episode mp4 files
print("\n=== Splitting videos ===")
for vk in video_keys:
    input_video = input_dir / "videos" / vk / "chunk-000" / "file-000.mp4"
    if not input_video.exists():
        print(f"  ⚠ Video not found: {input_video}")
        continue

    for ep_idx in range(total_episodes):
        ep_info = episodes_df.iloc[ep_idx]
        ep_length = int(ep_info["length"])

        # Calculate start/end timestamps from frame indices
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        start_frame = int(ep_data["frame_index"].min())
        # timestamp within the concatenated video
        # In v3.0, the video is concatenated, so we need cumulative frame offsets
        cumulative_offset = 0
        for i in range(ep_idx):
            cumulative_offset += int(episodes_df.iloc[i]["length"])

        start_time = cumulative_offset / fps
        duration = ep_length / fps

        output_video = output_dir / "videos" / "chunk-000" / vk / f"episode_{ep_idx:06d}.mp4"

        # Use ffmpeg to extract exact number of frames (re-encode to ensure frame accuracy)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start_time:.4f}",
            "-i", str(input_video),
            "-frames:v", str(ep_length),
            "-c:v", "libsvtav1", "-crf", "30", "-preset", "8",
            "-pix_fmt", "yuv420p",
            str(output_video),
        ]
        subprocess.run(cmd, check=True)
        print(f"  {vk} ep{ep_idx}: {start_time:.2f}s, {ep_length} frames → {output_video.name}")

# 3. Create episodes.jsonl
print("\n=== Creating meta files ===")
with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
    for ep_idx in range(total_episodes):
        ep_info = episodes_df.iloc[ep_idx]
        tasks = list(ep_info["tasks"]) if isinstance(ep_info["tasks"], list) else [str(ep_info["tasks"])]
        line = {"episode_index": int(ep_info["episode_index"]), "tasks": tasks, "length": int(ep_info["length"])}
        f.write(json.dumps(line) + "\n")
print("  episodes.jsonl ✓")

# 4. Create tasks.jsonl
with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
    # tasks_df has task text as index and task_index as column
    tasks_df_reset = tasks_df.reset_index()
    for _, row in tasks_df_reset.iterrows():
        # The index column has the task text, task_index column has the index
        task_text = row.iloc[0]  # first column (the task text, was the index)
        task_idx = int(row["task_index"])
        f.write(json.dumps({"task_index": task_idx, "task": task_text}) + "\n")
print("  tasks.jsonl ✓")

# 5. Create modality.json (same as demo_data format)
modality = {
    "state": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6},
    },
    "action": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6},
    },
    "video": {},
    "annotation": {
        "human.task_description": {"original_key": "task_index"}
    },
}

for vk in video_keys:
    short_key = vk.replace("observation.images.", "")
    modality["video"][short_key] = {"original_key": vk}

json.dump(modality, open(output_dir / "meta" / "modality.json", "w"), indent=4)
print("  modality.json ✓")

# 6. Create info.json (v2.1 format)
new_info = info.copy()
new_info["codebase_version"] = "v2.1"
new_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
new_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
# Remove v3-specific fields
new_info.pop("data_files_size_in_mb", None)
new_info.pop("video_files_size_in_mb", None)
new_info["total_chunks"] = 0
new_info["total_videos"] = total_episodes * len(video_keys)
json.dump(new_info, open(output_dir / "meta" / "info.json", "w"), indent=4)
print("  info.json ✓")

# 7. Copy stats.json
if (input_dir / "meta" / "stats.json").exists():
    shutil.copy2(input_dir / "meta" / "stats.json", output_dir / "meta" / "stats.json")
    print("  stats.json ✓")

print(f"\n✓ 轉換完成！輸出: {output_dir}")
print(f"  訓練指令: --dataset_path {output_dir}")
