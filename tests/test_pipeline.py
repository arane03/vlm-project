import json
from pathlib import Path

import numpy as np
from PIL import Image

from gs_pipeline.cameras import load_manifest, rotation_to_quaternion
from gs_pipeline.prepare import prepare_dataset


def test_rotation_to_quaternion_identity():
    np.testing.assert_allclose(rotation_to_quaternion(np.eye(3)), [1, 0, 0, 0])


def test_prepare_dataset(tmp_path: Path):
    Image.new("RGB", (8, 6), (20, 40, 60)).save(tmp_path / "frame.png")
    manifest = {"pose_convention": "camera_to_world", "K": [[10, 0, 4], [0, 11, 3], [0, 0, 1]],
                "frames": [{"image": "frame.png", "transform": np.eye(4).tolist(), "split": "train"}]}
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(manifest))
    assert load_manifest(path)[0].width == 8
    output = prepare_dataset(path, tmp_path / "dataset", seed_points=10)
    assert (output / "images/000001.png").is_file()
    assert "PINHOLE 8 6 10 11 4 3" in (output / "sparse/0/cameras.txt").read_text()
    assert (output / "sparse/0/points3D.ply").read_text().splitlines()[2] == "element vertex 10"

