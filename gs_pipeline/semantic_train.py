from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .semantic_loss import embedding_distillation_loss
from .semantic_prepare import unpack_masks


T = TypeVar("T")


def uniform_sample_views(views: Sequence[T], count: int, seed: int) -> list[T]:
    """Uniformly sample distinct views without replacement."""
    if count <= 0:
        raise ValueError("view count must be positive")
    if count > len(views):
        raise ValueError(f"requested {count} views, but only {len(views)} are available")
    return random.Random(seed).sample(list(views), count)


@dataclass(frozen=True)
class SemanticTrainingConfig:
    model: str
    data: str
    cache: str
    vae: str
    rgb_iteration: int
    view_count: int
    view_seed: int
    sampled_views: list[str]
    latent_dim: int
    gaussian_count: int
    epochs: int
    learning_rate: float
    resolution: int
    mask_batch_size: int
    loss_reduction: str


def _latest_iteration(model: Path) -> int:
    values = []
    for path in (model / "point_cloud").glob("iteration_*/point_cloud.ply"):
        try:
            values.append(int(path.parent.name.removeprefix("iteration_")))
        except ValueError:
            continue
    if not values:
        raise FileNotFoundError(f"no trained point cloud found below {model / 'point_cloud'}")
    return max(values)


def freeze_gaussians(gaussians: Any) -> None:
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        getattr(gaussians, name).requires_grad_(False)


def render_semantic_features(camera: Any, gaussians: Any, features: torch.Tensor) -> torch.Tensor:
    """Rasterize one three-channel feature vector per frozen Gaussian."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    if features.ndim != 2 or features.shape != (gaussians.get_xyz.shape[0], 3):
        raise ValueError("the stock rasterizer requires semantic features with shape [N,3]")
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    rendered, _, _ = rasterizer(
        means3D=gaussians.get_xyz,
        means2D=torch.zeros_like(gaussians.get_xyz),
        colors_precomp=features,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        cov3D_precomp=None,
    )
    # Semantic values are intentionally not exposed, clamped, or color corrected.
    return rendered


def load_view_targets(
    camera: Any,
    cache: Path,
    latent_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    record_path = cache / "records" / f"{camera.image_name}.npz"
    latent_path = latent_root / f"{camera.image_name}.npy"
    if not record_path.is_file():
        raise FileNotFoundError(f"missing semantic cache record: {record_path}")
    if not latent_path.is_file():
        raise FileNotFoundError(f"missing compressed targets: {latent_path}")
    with np.load(record_path, allow_pickle=False) as record:
        masks = unpack_masks(record)
    targets = np.asarray(np.load(latent_path, allow_pickle=False), dtype=np.float32)
    if len(masks) != len(targets):
        raise ValueError(f"mask/target count mismatch for {camera.image_name}")
    return masks, targets


def _masked_loss_in_chunks(
    rendered: torch.Tensor,
    masks: np.ndarray,
    targets: np.ndarray,
    mask_batch_size: int,
    reduction: str,
) -> torch.Tensor:
    if len(masks) == 0:
        return rendered.sum() * 0.0
    total = rendered.sum() * 0.0
    valid_masks = 0
    for start in range(0, len(masks), mask_batch_size):
        mask_tensor = torch.from_numpy(masks[start:start + mask_batch_size]).to(
            device=rendered.device, dtype=torch.float32
        )
        mask_tensor = F.interpolate(
            mask_tensor[:, None], size=rendered.shape[-2:], mode="nearest"
        )[:, 0]
        target_tensor = torch.from_numpy(targets[start:start + mask_batch_size]).to(
            device=rendered.device, dtype=torch.float32
        )
        chunk_loss = embedding_distillation_loss(
            rendered, mask_tensor, target_tensor, reduction=reduction
        )
        if reduction == "mask_mean":
            count = int((mask_tensor.sum(dim=(1, 2)) > 0).sum().item())
            total = total + chunk_loss * count
            valid_masks += count
        else:
            total = total + chunk_loss
    if reduction == "mask_mean":
        return total / max(valid_masks, 1)
    return total


def _save_preview(path: Path, rendered: torch.Tensor) -> None:
    image = rendered.detach().float().cpu()
    low = image.flatten(1).quantile(0.01, dim=1)[:, None, None]
    high = image.flatten(1).quantile(0.99, dim=1)[:, None, None]
    image = ((image - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    pixels = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def train_semantics(args: argparse.Namespace) -> Path:
    # Imports resolve from the vendored GraphDECO checkout supplied via PYTHONPATH.
    from scene import GaussianModel, Scene

    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    cache_path = Path(args.cache).expanduser().resolve()
    vae_path = Path(args.vae).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    iteration = _latest_iteration(model_path) if args.iteration < 0 else args.iteration

    vae_checkpoint = torch.load(vae_path, map_location="cpu", weights_only=True)
    latent_dim = int(vae_checkpoint["latent_dim"])
    if latent_dim != 3:
        raise ValueError(
            f"this trainer uses the stock three-channel rasterizer; VAE latent_dim is {latent_dim}, expected 3"
        )
    latent_root = vae_path.parent / "latents"

    scene_args = argparse.Namespace(
        source_path=str(data_path),
        model_path=str(model_path),
        images="images",
        depths="",
        resolution=args.resolution,
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=True,
        sh_degree=3,
    )
    gaussians = GaussianModel(scene_args.sh_degree)
    scene = Scene(scene_args, gaussians, load_iteration=iteration, shuffle=False)
    freeze_gaussians(gaussians)
    cameras = scene.getTrainCameras()
    eligible = [
        camera for camera in cameras
        if (cache_path / "records" / f"{camera.image_name}.npz").is_file()
        and (latent_root / f"{camera.image_name}.npy").is_file()
    ]
    sampled = uniform_sample_views(eligible, args.views, args.view_seed)

    # Fail before optimization if any sampled view lacks a target.
    for camera in sampled:
        load_view_targets(camera, cache_path, latent_root)

    semantic = nn.Parameter(
        torch.randn(gaussians.get_xyz.shape[0], latent_dim, device="cuda") * args.init_std
    )
    optimizer = torch.optim.Adam([semantic], lr=args.learning_rate)
    config = SemanticTrainingConfig(
        model=str(model_path),
        data=str(data_path),
        cache=str(cache_path),
        vae=str(vae_path),
        rgb_iteration=iteration,
        view_count=args.views,
        view_seed=args.view_seed,
        sampled_views=[camera.image_name for camera in sampled],
        latent_dim=latent_dim,
        gaussian_count=int(semantic.shape[0]),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        resolution=args.resolution,
        mask_batch_size=args.mask_batch_size,
        loss_reduction=args.loss_reduction,
    )
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    (output / "sampled_views.json").write_text(
        json.dumps(config.sampled_views, indent=2) + "\n"
    )

    history: list[float] = []
    shuffle_rng = random.Random(args.view_seed + 1)
    for epoch in range(1, args.epochs + 1):
        order = list(sampled)
        shuffle_rng.shuffle(order)
        epoch_loss = 0.0
        for step, camera in enumerate(order, 1):
            masks, targets = load_view_targets(camera, cache_path, latent_root)
            rendered = render_semantic_features(camera, gaussians, semantic)
            loss = _masked_loss_in_chunks(
                rendered, masks, targets, args.mask_batch_size, args.loss_reduction
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step == 1 or step % 10 == 0:
                print(
                    f"epoch {epoch:03d}/{args.epochs} view {step:02d}/{len(order)} "
                    f"loss={loss.item():.8f}"
                )
        epoch_loss /= len(order)
        history.append(epoch_loss)
        print(f"epoch {epoch:03d}: mean_loss={epoch_loss:.8f}")

        checkpoint = {
            "semantic_features": semantic.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "history": history,
        }
        torch.save(checkpoint, output / "semantic_features.pt")
        if args.preview_every > 0 and epoch % args.preview_every == 0:
            with torch.no_grad():
                preview = render_semantic_features(sampled[0], gaussians, semantic)
            _save_preview(output / "previews" / f"epoch_{epoch:03d}.png", preview)

    return output / "semantic_features.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train frozen-geometry Gaussian semantic embeddings")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--views", type=int, default=50)
    parser.add_argument("--view-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--init-std", type=float, default=0.01)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--mask-batch-size", type=int, default=8)
    parser.add_argument("--loss-reduction", choices=["sum", "mask_mean"], default="mask_mean")
    parser.add_argument("--preview-every", type=int, default=1)
    return parser


def main() -> None:
    checkpoint = train_semantics(build_parser().parse_args())
    print(checkpoint)


if __name__ == "__main__":
    main()
