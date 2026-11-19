from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData


FROZEN_GAUSSIAN_FIELDS = [
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
]


@dataclass(frozen=True)
class SemanticPrepareConfig:
    model: str
    data: str
    iteration: int
    point_cloud: str
    gaussian_count: int
    sam_model: str
    clip_model: str
    split: str
    min_mask_area: int
    max_mask_fraction: float
    pred_iou_threshold: float
    points_per_batch: int
    duplicate_iou_threshold: float
    max_masks: int
    crop_padding: float
    clip_mask_mode: str
    frozen_fields: list[str]
    view_count: int
    view_seed: int
    sampled_images: list[str]


def find_iteration(model: str | Path, iteration: int) -> tuple[int, Path]:
    model = Path(model).expanduser().resolve()
    point_cloud_root = model / "point_cloud"
    if iteration < 0:
        candidates: list[tuple[int, Path]] = []
        for path in point_cloud_root.glob("iteration_*/point_cloud.ply"):
            try:
                value = int(path.parent.name.removeprefix("iteration_"))
            except ValueError:
                continue
            candidates.append((value, path))
        if not candidates:
            raise FileNotFoundError(f"no trained point cloud found below {point_cloud_root}")
        return max(candidates, key=lambda item: item[0])

    path = point_cloud_root / f"iteration_{iteration}" / "point_cloud.ply"
    if not path.is_file():
        raise FileNotFoundError(f"trained point cloud not found: {path}")
    return iteration, path


def dataset_images(data: str | Path, split: str) -> list[Path]:
    data = Path(data).expanduser().resolve()
    images_dir = data / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"prepared image directory not found: {images_dir}")

    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    test_file = data / "sparse" / "0" / "test.txt"
    test_names = set(test_file.read_text().splitlines()) if test_file.is_file() else set()
    if split == "train":
        images = [path for path in images if path.name not in test_names]
    elif split == "test":
        images = [path for path in images if path.name in test_names]
    if not images:
        raise ValueError(f"no {split} images found in {images_dir}")
    return images


def _as_mask_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=bool)
    if result.ndim == 2:
        result = result[None]
    if result.ndim != 3:
        raise ValueError(f"SAM returned masks with unexpected shape {result.shape}")
    return result


def _as_score_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def filter_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int,
    max_fraction: float,
    duplicate_iou: float,
    max_masks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = masks.shape[-2:]
    image_area = height * width
    order = np.argsort(-scores)
    kept_masks: list[np.ndarray] = []
    kept_scores: list[float] = []
    boxes: list[list[int]] = []

    for index in order:
        mask = masks[index]
        area = int(mask.sum())
        if area < min_area or area > max_fraction * image_area:
            continue
        duplicate = False
        for accepted in kept_masks:
            intersection = np.logical_and(mask, accepted).sum()
            union = area + accepted.sum() - intersection
            if union and intersection / union >= duplicate_iou:
                duplicate = True
                break
        if duplicate:
            continue
        ys, xs = np.nonzero(mask)
        kept_masks.append(mask)
        kept_scores.append(float(scores[index]))
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
        if len(kept_masks) >= max_masks:
            break

    if not kept_masks:
        return (
            np.empty((0, height, width), dtype=bool),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 4), dtype=np.int32),
        )
    return (
        np.stack(kept_masks),
        np.asarray(kept_scores, dtype=np.float32),
        np.asarray(boxes, dtype=np.int32),
    )


def masked_crops(
    image: Image.Image,
    masks: np.ndarray,
    boxes: np.ndarray,
    padding: float,
    mode: str = "full",
) -> list[Image.Image]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    crops: list[Image.Image] = []
    for mask, (x0, y0, x1, y1) in zip(masks, boxes):
        pad_x = int(math.ceil((x1 - x0) * padding))
        pad_y = int(math.ceil((y1 - y0) * padding))
        x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
        y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
        masked = np.full_like(rgb, 127)
        if mode == "full":
            masked.fill(0)
        masked[mask] = rgb[mask]
        if mode == "full":
            crops.append(Image.fromarray(masked))
        elif mode == "crop":
            crops.append(Image.fromarray(masked[y0:y1, x0:x1]))
        else:
            raise ValueError(f"unknown CLIP mask mode: {mode}")
    return crops


def encode_crops(model: Any, processor: Any, crops: list[Image.Image], device: torch.device,
                 batch_size: int) -> np.ndarray:
    batches: list[torch.Tensor] = []
    for start in range(0, len(crops), batch_size):
        inputs = processor(images=crops[start:start + batch_size], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        batches.append(features.cpu())
    if not batches:
        dimension = int(getattr(model.config, "projection_dim", 0))
        return np.empty((0, dimension), dtype=np.float16)
    return torch.cat(batches).numpy().astype(np.float16)


def _batched(values: Iterable[Path], size: int) -> Iterable[list[Path]]:
    batch: list[Path] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_semantics(
    *,
    model_path: str,
    data_path: str,
    output_path: str,
    iteration: int = -1,
    split: str = "train",
    sam_model: str = "facebook/sam2-hiera-base-plus",
    clip_model: str = "openai/clip-vit-base-patch32",
    device_name: str = "cuda",
    points_per_batch: int = 64,
    image_batch_size: int = 1,
    clip_batch_size: int = 32,
    min_mask_area: int = 256,
    max_mask_fraction: float = 0.95,
    pred_iou_threshold: float = 0.88,
    duplicate_iou_threshold: float = 0.90,
    max_masks: int = 64,
    crop_padding: float = 0.10,
    clip_mask_mode: str = "full",
    views: int = 50,
    view_seed: int = 0,
) -> Path:
    try:
        from transformers import AutoProcessor, CLIPModel, pipeline
    except ImportError as error:
        raise RuntimeError("semantic preparation requires transformers and torchvision") from error

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    resolved_iteration, ply_path = find_iteration(model_path, iteration)
    gaussian_count = int(PlyData.read(ply_path)["vertex"].count)
    images = dataset_images(data_path, split)
    if views > 0:
        if views > len(images):
            raise ValueError(f"requested {views} views, but only {len(images)} {split} images exist")
        images = random.Random(view_seed).sample(images, views)
        images.sort()
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "records"
    records_dir.mkdir(exist_ok=True)

    config = SemanticPrepareConfig(
        model=str(Path(model_path).expanduser().resolve()),
        data=str(Path(data_path).expanduser().resolve()),
        iteration=resolved_iteration,
        point_cloud=str(ply_path),
        gaussian_count=gaussian_count,
        sam_model=sam_model,
        clip_model=clip_model,
        split=split,
        min_mask_area=min_mask_area,
        max_mask_fraction=max_mask_fraction,
        pred_iou_threshold=pred_iou_threshold,
        points_per_batch=points_per_batch,
        duplicate_iou_threshold=duplicate_iou_threshold,
        max_masks=max_masks,
        crop_padding=crop_padding,
        clip_mask_mode=clip_mask_mode,
        frozen_fields=FROZEN_GAUSSIAN_FIELDS,
        view_count=len(images),
        view_seed=view_seed,
        sampled_images=[path.name for path in images],
    )
    config_path = output / "manifest.json"
    serialized_config = json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    if config_path.is_file() and config_path.read_text() != serialized_config:
        raise RuntimeError(f"existing semantic cache has different settings: {config_path}")
    config_path.write_text(serialized_config)

    pending = [path for path in images if not (records_dir / f"{path.name}.npz").is_file()]
    if not pending:
        print(f"Semantic cache is already complete: {output}")
        return output

    pipeline_device = device.index if device.type == "cuda" else -1
    if pipeline_device is None and device.type == "cuda":
        pipeline_device = 0
    mask_generator = pipeline("mask-generation", model=sam_model, device=pipeline_device)
    clip_processor = AutoProcessor.from_pretrained(clip_model)
    clip = CLIPModel.from_pretrained(clip_model).to(device).eval()

    completed = len(images) - len(pending)
    for image_paths in _batched(pending, image_batch_size):
        # The Transformers SAM pipeline currently batches prompts rather than images,
        # so images are deliberately processed one at a time within this outer batch.
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            result = mask_generator(
                image,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_threshold,
            )
            masks = _as_mask_array(result["masks"])
            scores = _as_score_array(result["scores"])
            masks, scores, boxes = filter_masks(
                masks,
                scores,
                min_mask_area,
                max_mask_fraction,
                duplicate_iou_threshold,
                max_masks,
            )
            crops = masked_crops(image, masks, boxes, crop_padding, clip_mask_mode)
            embeddings = encode_crops(clip, clip_processor, crops, device, clip_batch_size)
            packed_masks = np.packbits(masks, axis=-1)
            destination = records_dir / f"{image_path.name}.npz"
            np.savez_compressed(
                destination,
                image_name=np.asarray(image_path.name),
                image_size=np.asarray([image.height, image.width], dtype=np.int32),
                masks_packed=packed_masks,
                mask_width=np.asarray(image.width, dtype=np.int32),
                boxes=boxes,
                scores=scores,
                embeddings=embeddings,
            )
            completed += 1
            print(f"[{completed}/{len(images)}] {image_path.name}: {len(masks)} masks")

    return output


def unpack_masks(record: Any) -> np.ndarray:
    """Recover boolean [N,H,W] masks from one cached .npz record."""
    width = int(record["mask_width"])
    return np.unpackbits(record["masks_packed"], axis=-1, count=width).astype(bool)
