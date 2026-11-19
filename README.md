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

## 4. Prepare semantic distillation targets

After RGB training, cache SAM2 masks and normalized CLIP embeddings for the
prepared training images. This command records the trained PLY as the frozen
geometry snapshot and never invokes the RGB optimizer:

```bash
gs-pipeline semantic-prepare \
  --model /models/scene \
  --data /data/scene/graphdeco \
  --output /data/scene/semantic-cache \
  --iteration 30000 \
  --views 50 \
  --view-seed 0
```

Model weights are downloaded through Hugging Face on the first run. The command
is resumable: records that already exist are skipped when the saved manifest
matches. Each compressed `.npz` record contains packed boolean masks, boxes,
SAM confidence scores, and float16 unit-normalized CLIP image embeddings.
The default benchmark uniformly samples 50 distinct training images before
running SAM or CLIP; use `--views 0` to preprocess every training image.
By default CLIP receives the literal black-background product `M_i * I`, as in
3DitScene; `--clip-mask-mode crop` is available for crop-based experiments.

Compress the cached CLIP vectors into a rasterizer-friendly latent space with
a small VAE:

```bash
gs-pipeline semantic-train-vae \
  --cache /data/scene/semantic-cache \
  --output /data/scene/semantic-vae \
  --latent-dim 3 \
  --epochs 50
```

The current objective is intentionally reconstruction MSE only; it does not add
a KL-divergence term. The saved `clip_vae.pt` exposes the deterministic encoder
mean for compressing both image and text CLIP features. Deterministic compressed
image features are also written below `semantic-vae/latents/`, with filenames
matching the source cache records.

The semantic training objective is available as
`gs_pipeline.semantic_loss.embedding_distillation_loss`. Given a rasterized
embedding image `E`, masks `M`, and one compressed CLIP target per mask, its
default reduction computes the literal 3DitScene objective
`sum_i ||(E - g(M_i * I)) * M_i||^2`. Use `reduction="mask_mean"` to give
different-sized masks equal weight during optimization.

Train one three-channel semantic vector per frozen Gaussian on a reproducible
uniform sample of 50 distinct training views:

```bash
gs-pipeline semantic-train \
  --model /models/scene \
  --data /data/scene/graphdeco \
  --cache /data/scene/semantic-cache \
  --vae /data/scene/semantic-vae/clip_vae.pt \
  --output /models/scene-semantics \
  --iteration 30000 \
  --views 50 \
  --view-seed 0 \
  --epochs 10
```

The sampled filenames are saved in `sampled_views.json`. The trainer loads the
RGB PLY, freezes geometry, opacity, and SH appearance, disables topology
changes, and optimizes only the semantic tensor. Its default half-resolution
renders and mask chunking reduce memory use without changing the sampled view
set. The latest semantic checkpoint is `semantic_features.pt`; false-color
embedding previews are saved once per epoch.

The novel-view path supports per-axis focal lengths and off-center principal points. Use `--white-background` if the model was trained with that GraphDECO option.

## Outputs

The trained splats are stored at `MODEL/point_cloud/iteration_N/point_cloud.ply`. Dataset renders are written below `MODEL/{train,test}/ours_N/renders`; arbitrary renders go to the requested output path.

GraphDECO is research software with its own non-commercial license; review `third_party/gaussian-splatting/LICENSE.md` before use.
