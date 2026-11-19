from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ClipVAE(nn.Module):
    """Small VAE for compressing unit-normalized CLIP vectors.

    Training currently uses reconstruction loss only. The mean is used as the
    stable semantic code at inference time; sampling is used while training.
    """

    def __init__(self, input_dim: int, latent_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.to_mu = nn.Linear(64, latent_dim)
        self.to_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def distribution(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(F.normalize(features.float(), dim=-1))
        return self.to_mu(hidden), self.to_logvar(hidden).clamp(-10.0, 10.0)

    def encode(self, features: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Encode image or text CLIP vectors using the deterministic mean."""
        mu, _ = self.distribution(features)
        return F.normalize(mu, dim=-1) if normalize else mu

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        reconstruction = self.decoder(latent)
        return F.normalize(reconstruction, dim=-1) if normalize else reconstruction

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.distribution(features)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar, latent


@dataclass(frozen=True)
class VAETrainingConfig:
    cache: str
    input_dim: int
    latent_dim: int
    embedding_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    objective: str = "mse_reconstruction_only"


def load_cached_embeddings(cache_path: str | Path) -> np.ndarray:
    records_dir = Path(cache_path).expanduser().resolve() / "records"
    record_paths = sorted(records_dir.glob("*.npz"))
    if not record_paths:
        raise FileNotFoundError(f"no semantic records found in {records_dir}")

    arrays: list[np.ndarray] = []
    dimension: int | None = None
    for path in record_paths:
        with np.load(path, allow_pickle=False) as record:
            values = np.asarray(record["embeddings"], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"embeddings in {path} must be a matrix, got {values.shape}")
        if values.shape[0] == 0:
            continue
        if dimension is None:
            dimension = values.shape[1]
        elif values.shape[1] != dimension:
            raise ValueError(f"inconsistent embedding dimension in {path}")
        arrays.append(values)
    if not arrays:
        raise ValueError(f"semantic cache contains no embeddings: {records_dir}")
    return np.concatenate(arrays, axis=0)


def load_vae(path: str | Path, device: str | torch.device = "cpu") -> ClipVAE:
    checkpoint = torch.load(Path(path).expanduser(), map_location=device, weights_only=True)
    model = ClipVAE(checkpoint["input_dim"], checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def train_vae(
    *,
    cache_path: str,
    output_path: str,
    latent_dim: int = 3,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device_name: str = "cuda",
    seed: int = 0,
) -> Path:
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    embeddings = load_cached_embeddings(cache_path)
    features = F.normalize(torch.from_numpy(embeddings), dim=-1)
    input_dim = int(features.shape[1])
    loader = DataLoader(
        TensorDataset(features),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model = ClipVAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = VAETrainingConfig(
        cache=str(Path(cache_path).expanduser().resolve()),
        input_dim=input_dim,
        latent_dim=latent_dim,
        embedding_count=int(features.shape[0]),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")

    history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            reconstruction, _, _, _ = model(batch)
            # Deliberately no KL-divergence term: reconstruction only.
            loss = F.mse_loss(reconstruction, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch.shape[0]
            sample_count += batch.shape[0]
        epoch_loss = loss_sum / sample_count
        history.append(epoch_loss)
        print(f"epoch {epoch:03d}/{epochs}: reconstruction_mse={epoch_loss:.8f}")

    checkpoint_path = output / "clip_vae.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "config": asdict(config),
            "history": history,
        },
        checkpoint_path,
    )
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    # Materialize deterministic latent means for the cached image features.
    latent_dir = output / "latents"
    latent_dir.mkdir(exist_ok=True)
    model.eval()
    for record_path in sorted((Path(cache_path).expanduser().resolve() / "records").glob("*.npz")):
        with np.load(record_path, allow_pickle=False) as record:
            values = np.asarray(record["embeddings"], dtype=np.float32)
        latent_batches: list[torch.Tensor] = []
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(device)
            with torch.inference_mode():
                latent_batches.append(model.encode(batch).cpu())
        if latent_batches:
            latent = torch.cat(latent_batches).numpy().astype(np.float16)
        else:
            latent = np.empty((0, latent_dim), dtype=np.float16)
        np.save(latent_dir / f"{record_path.stem}.npy", latent)

    return checkpoint_path
