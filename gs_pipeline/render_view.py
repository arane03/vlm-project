from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _latest_iteration(model: Path) -> int:
    values = []
    for path in (model / "point_cloud").glob("iteration_*"):
        try:
            values.append(int(path.name.removeprefix("iteration_")))
        except ValueError:
            pass
    if not values:
        raise FileNotFoundError(f"no trained point cloud found under {model}")
    return max(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained GraphDECO model from an arbitrary camera")
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", required=True, help="JSON with width, height, K, transform, pose_convention")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--antialiasing", action="store_true")
    args = parser.parse_args()

    # These imports intentionally resolve from the official GraphDECO checkout.
    from gaussian_renderer import GaussianModel, render
    from scene.cameras import Camera

    model = Path(args.model).resolve()
    iteration = _latest_iteration(model) if args.iteration == -1 else args.iteration
    ply = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError(ply)
    raw = json.loads(Path(args.camera).read_text())
    width, height = int(raw["width"]), int(raw["height"])
    K = np.asarray(raw["K"], dtype=np.float64)
    transform = np.asarray(raw["transform"], dtype=np.float64)
    if K.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("K and transform must be 3x3 and 4x4")
    if raw.get("pose_convention", "world_to_camera") == "camera_to_world":
        transform = np.linalg.inv(transform)
    elif raw.get("pose_convention", "world_to_camera") != "world_to_camera":
        raise ValueError("unknown pose_convention")

    fovx = 2 * math.atan(width / (2 * K[0, 0]))
    fovy = 2 * math.atan(height / (2 * K[1, 1]))
    blank = Image.new("RGB", (width, height))
    camera = Camera((width, height), 0, transform[:3, :3].T, transform[:3, 3], fovx, fovy,
                    None, blank, None, "novel", 0)
    # GraphDECO's stock camera assumes the principal point is centered. Add the
    # standard off-axis terms while retaining its CUDA/rasterizer conventions.
    projection = camera.projection_matrix.clone()
    projection[2, 0] = 2.0 * float(K[0, 2]) / width - 1.0
    projection[2, 1] = 2.0 * float(K[1, 2]) / height - 1.0
    camera.projection_matrix = projection
    camera.full_proj_transform = camera.world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)

    gaussians = GaussianModel(3)
    gaussians.load_ply(str(ply), False)
    pipeline = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                                  debug=False, antialiasing=args.antialiasing)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    with torch.no_grad():
        image = render(camera, gaussians, pipeline, background)["render"].clamp(0, 1)
    pixels = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output)
    print(output)


if __name__ == "__main__":
    main()

