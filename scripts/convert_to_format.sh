#!/bin/bash
read -rp "Scene to Process (id): " scene_id
export SCENE_DIR="$(pwd)/dataset/${scene_id}"
uv run python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

scene = Path(os.environ["SCENE_DIR"]).resolve()
source_path = scene / "transforms_undistorted.json"
image_dir = scene / "resized_undistorted_images"
output_path = scene / "cameras.json"

source = json.loads(source_path.read_text())

K = [
    [float(source["fl_x"]), 0.0, float(source["cx"])],
    [0.0, float(source["fl_y"]), float(source["cy"])],
    [0.0, 0.0, 1.0],
]

frames = []

def add_frames(items, split):
    for item in items:
        image_name = Path(item["file_path"]).name
        image_path = image_dir / image_name

        if not image_path.is_file():
            print(f"Skipping missing {split} image: {image_path}")
            continue

        # ScanNet++ Nerfstudio -> ScanNet++ COLMAP/OpenCV convention.
        c2w = np.asarray(item["transform_matrix"], dtype=np.float64).copy()
        c2w[2, :] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[:3, 1:3] *= -1

        frames.append({
            "image": str(image_path),
            "K": K,
            "transform": c2w.tolist(),
            "split": split,
        })

add_frames(source.get("frames", []), "train")
add_frames(source.get("test_frames", []), "test")

if not frames:
    raise RuntimeError("No images from the transform file were found")

if not any(frame["split"] == "train" for frame in frames):
    raise RuntimeError("No training images were found")

manifest = {
    "pose_convention": "camera_to_world",
    "frames": frames,
}

output_path.write_text(json.dumps(manifest, indent=2) + "\n")

train_count = sum(frame["split"] == "train" for frame in frames)
test_count = sum(frame["split"] == "test" for frame in frames)

print(f"Wrote: {output_path}")
print(f"Training frames: {train_count}")
print(f"Test frames: {test_count}")
PY