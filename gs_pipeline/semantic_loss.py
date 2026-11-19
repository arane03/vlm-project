from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


LossReduction = Literal["sum", "mean", "mask_mean", "none"]


def embedding_distillation_loss(
    rendered_embedding: torch.Tensor,
    masks: torch.Tensor,
    target_embeddings: torch.Tensor,
    *,
    reduction: LossReduction = "sum",
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """3DitScene-style masked embedding distillation loss.

    Computes, for every mask i,

        || (E - g(M_i * I)) * M_i ||_2^2

    where ``rendered_embedding`` is E and ``target_embeddings`` contains the
    (optionally VAE-compressed) output of g for each masked image. Inputs may be
    unbatched or batched:

      rendered_embedding: [C,H,W] or [B,C,H,W]
      masks:              [M,H,W] or [B,M,H,W]
      target_embeddings:  [M,C] or [B,M,C]

    ``sum`` is the literal loss in the paper. ``mask_mean`` first averages each
    mask over its valid pixels and channels, then gives every non-empty mask
    equal weight; this is often easier to optimize when mask areas vary widely.
    """
    unbatched = rendered_embedding.ndim == 3
    if unbatched:
        rendered_embedding = rendered_embedding.unsqueeze(0)
        masks = masks.unsqueeze(0)
        target_embeddings = target_embeddings.unsqueeze(0)

    if rendered_embedding.ndim != 4:
        raise ValueError("rendered_embedding must have shape [C,H,W] or [B,C,H,W]")
    if masks.ndim != 4:
        raise ValueError("masks must have shape [M,H,W] or [B,M,H,W]")
    if target_embeddings.ndim != 3:
        raise ValueError("target_embeddings must have shape [M,C] or [B,M,C]")

    batch, channels, height, width = rendered_embedding.shape
    if masks.shape[0] != batch or masks.shape[2:] != (height, width):
        raise ValueError("mask batch/spatial dimensions must match the rendered embedding")
    if target_embeddings.shape[:2] != masks.shape[:2]:
        raise ValueError("one target embedding is required for every mask")
    if target_embeddings.shape[2] != channels:
        raise ValueError(
            f"target dimension {target_embeddings.shape[2]} does not match rendered channels {channels}"
        )

    rendered_embedding = rendered_embedding.float()
    target_embeddings = target_embeddings.to(
        device=rendered_embedding.device, dtype=rendered_embedding.dtype
    )
    masks = masks.to(device=rendered_embedding.device, dtype=rendered_embedding.dtype)

    if normalize:
        rendered_embedding = F.normalize(rendered_embedding, dim=1, eps=eps)
        target_embeddings = F.normalize(target_embeddings, dim=-1, eps=eps)

    # Expand ||(E-t)M||^2 algebraically instead of materializing a potentially
    # enormous [B,M,C,H,W] residual at full image resolution.
    weights = masks.square()
    areas = weights.sum(dim=(2, 3))
    rendered_squared = torch.einsum(
        "bmhw,bchw->bm", weights, rendered_embedding.square()
    )
    cross = torch.einsum(
        "bmhw,bchw,bmc->bm", weights, rendered_embedding, target_embeddings
    )
    target_squared = areas * target_embeddings.square().sum(dim=-1)
    per_mask = rendered_squared - 2.0 * cross + target_squared

    if reduction == "none":
        return per_mask[0] if unbatched else per_mask
    if reduction == "sum":
        return per_mask.sum()
    if reduction == "mean":
        return per_mask.sum() / (batch * masks.shape[1] * channels * height * width)
    if reduction == "mask_mean":
        valid = areas > 0
        if not valid.any():
            # Preserve a differentiable zero tied to the rendered tensor.
            return rendered_embedding.sum() * 0.0
        normalized = per_mask / (areas * channels).clamp_min(eps)
        return normalized[valid].mean()
    raise ValueError(f"unknown reduction: {reduction}")
