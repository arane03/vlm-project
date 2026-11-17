from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Frame:
    image_path: Path
    width: int
    height: int
    K: np.ndarray
    world_to_camera: np.ndarray
    split: str = "train"


def _matrix(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix")
    return result


def load_manifest(path: str | Path) -> list[Frame]:
    manifest_path = Path(path).expanduser().resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get("pose_convention", "world_to_camera") not in {"world_to_camera", "camera_to_world"}:
        raise ValueError("pose_convention must be 'world_to_camera' or 'camera_to_world'")
    convention = raw.get("pose_convention", "world_to_camera")
    global_K = raw.get("K")
    frames: list[Frame] = []
    for index, item in enumerate(raw.get("frames", [])):
        image_path = (manifest_path.parent / item["image"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"frame {index}: image not found: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        K = _matrix(item.get("K", global_K), (3, 3), f"frame {index} K")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[2, 2] - 1) > 1e-8:
            raise ValueError(f"frame {index}: invalid pinhole intrinsics")
        transform = _matrix(item["transform"], (4, 4), f"frame {index} transform")
        w2c = np.linalg.inv(transform) if convention == "camera_to_world" else transform
        if not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"frame {index}: transform must be rigid homogeneous")
        rotation = w2c[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < 0.999:
            raise ValueError(f"frame {index}: rotation is not in SO(3)")
        split = item.get("split", "train")
        if split not in {"train", "test"}:
            raise ValueError(f"frame {index}: split must be train or test")
        frames.append(Frame(image_path, width, height, K, w2c, split))
    if not frames:
        raise ValueError("manifest contains no frames")
    if not any(frame.split == "train" for frame in frames):
        raise ValueError("manifest must contain at least one training frame")
    return frames


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return COLMAP's scalar-first quaternion for a 3x3 rotation."""
    trace = np.trace(rotation)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s,
                      (rotation[0, 2] - rotation[2, 0]) / s,
                      (rotation[1, 0] - rotation[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2
        q = np.empty(4)
        q[i + 1] = 0.25 * s
        q[0] = (rotation[k, j] - rotation[j, k]) / s
        q[j + 1] = (rotation[j, i] + rotation[i, j]) / s
        q[k + 1] = (rotation[k, i] + rotation[i, k]) / s
    return q / np.linalg.norm(q)

