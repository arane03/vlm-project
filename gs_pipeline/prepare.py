from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .cameras import Frame, load_manifest, rotation_to_quaternion


def _seed_points(frames: list[Frame], count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    centers = np.stack([np.linalg.inv(f.world_to_camera)[:3, 3] for f in frames])
    forwards = np.stack([np.linalg.inv(f.world_to_camera)[:3, 2] for f in frames])
    center = np.median(centers + forwards, axis=0)
    baseline = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
    radius = max(float(np.percentile(baseline, 90)), 0.5)
    rng = np.random.default_rng(seed)
    points = center + rng.normal(size=(count, 3)) * radius
    colors = rng.integers(32, 224, size=(count, 3), dtype=np.uint8)
    return points, colors


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = ("ply\nformat ascii 1.0\n" f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("w") as output:
        output.write(header)
        for xyz, rgb in zip(points, colors):
            output.write(f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} 0 0 0 {rgb[0]} {rgb[1]} {rgb[2]}\n")


def prepare_dataset(manifest: str | Path, output: str | Path, seed_points: int = 100_000,
                    seed: int = 0, overwrite: bool = False) -> Path:
    frames = load_manifest(manifest)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output is not empty: {output} (use --overwrite)")
    images_dir = output / "images"
    sparse_dir = output / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    camera_ids: dict[tuple, int] = {}
    camera_lines, image_lines, test_names = [], [], []
    used_names: set[str] = set()
    for index, frame in enumerate(frames, 1):
        key = (frame.width, frame.height, *frame.K[[0, 1], [0, 1]], frame.K[0, 2], frame.K[1, 2])
        if key not in camera_ids:
            camera_ids[key] = len(camera_ids) + 1
            cid = camera_ids[key]
            camera_lines.append(f"{cid} PINHOLE {frame.width} {frame.height} {frame.K[0,0]:.12g} "
                                f"{frame.K[1,1]:.12g} {frame.K[0,2]:.12g} {frame.K[1,2]:.12g}")
        cid = camera_ids[key]
        name = f"{index:06d}{frame.image_path.suffix.lower()}"
        if name in used_names:
            raise RuntimeError(f"duplicate generated image name: {name}")
        used_names.add(name)
        shutil.copy2(frame.image_path, images_dir / name)
        q = rotation_to_quaternion(frame.world_to_camera[:3, :3])
        t = frame.world_to_camera[:3, 3]
        image_lines.extend([f"{index} {' '.join(f'{v:.12g}' for v in q)} "
                            f"{' '.join(f'{v:.12g}' for v in t)} {cid} {name}", ""])
        if frame.split == "test":
            test_names.append(name)

    (sparse_dir / "cameras.txt").write_text("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + "\n".join(camera_lines) + "\n")
    (sparse_dir / "images.txt").write_text("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + "\n".join(image_lines) + "\n")
    (sparse_dir / "points3D.txt").write_text("# Empty: initialization is provided by points3D.ply\n")
    if test_names:
        (sparse_dir / "test.txt").write_text("\n".join(test_names) + "\n")
    points, colors = _seed_points([f for f in frames if f.split == "train"], seed_points, seed)
    _write_ply(sparse_dir / "points3D.ply", points, colors)
    return output

