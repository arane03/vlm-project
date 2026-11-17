# End-to-end GraphDECO 3D Gaussian Splatting

This project turns images plus calibrated pinhole cameras into a trained 3D Gaussian Splatting model and renders it from dataset or arbitrary camera views. Training and rasterization use the official [GraphDECO implementation](https://github.com/graphdeco-inria/gaussian-splatting), vendored under `third_party/gaussian-splatting`.

## Requirements

- Linux with an NVIDIA CUDA GPU (GraphDECO's rasterizer is CUDA-only)
- A CUDA-compatible PyTorch installation and CUDA compiler
- Python 3.10+

The repository uses `uv`. The basic CPU-side environment is enough to validate
manifests and prepare datasets:

```bash
uv sync
uv run gs-pipeline --help
```

For training and rendering, use Linux with an NVIDIA GPU and first install a
CUDA/PyTorch combination supported by GraphDECO. Then build its CUDA extensions:

```bash
bash scripts/bootstrap.sh
```

The checkout is pinned and its local compatibility changes are recorded in `THIRD_PARTY.md`.

## 1. Describe the calibrated sequence

Create `cameras.json`; see `examples/cameras.example.json`. Paths are relative to the JSON. `K` can be global or overridden per frame. Every transform is a rigid 4x4 matrix. Set `pose_convention` to `world_to_camera` or `camera_to_world`. Coordinates follow COLMAP/OpenCV: camera X right, Y down, Z forward. Mark optional held-out frames with `"split": "test"`.

## 2. Prepare and train

```bash
gs-pipeline prepare --manifest /data/scene/cameras.json --output /data/scene/graphdeco
gs-pipeline train --data /data/scene/graphdeco --output /models/scene --iterations 30000 --eval
```

The preparer validates images, intrinsics, rotations, and splits; writes a COLMAP text model; and creates a deterministic 100k-point seed cloud. If an SfM point cloud is available, replacing `sparse/0/points3D.ply` before training usually improves convergence. Extra GraphDECO flags can follow `--`, for example `gs-pipeline train ... -- --resolution 2`.

## 3. Render

Render held-out or training cameras:

```bash
gs-pipeline render --model /models/scene --split test
```

Render any calibrated view using `examples/view.example.json` as the schema:

```bash
gs-pipeline render-view --model /models/scene --camera /data/new_view.json --output /data/render.png
```

The novel-view path supports per-axis focal lengths and off-center principal points. Use `--white-background` if the model was trained with that GraphDECO option.

## Outputs

The trained splats are stored at `MODEL/point_cloud/iteration_N/point_cloud.ply`. Dataset renders are written below `MODEL/{train,test}/ours_N/renders`; arbitrary renders go to the requested output path.

GraphDECO is research software with its own non-commercial license; review `third_party/gaussian-splatting/LICENSE.md` before use.
