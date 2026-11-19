import json
from pathlib import Path

import numpy as np
from PIL import Image

from gs_pipeline.cameras import load_manifest, rotation_to_quaternion
from gs_pipeline.prepare import prepare_dataset
from gs_pipeline.semantic_prepare import dataset_images, filter_masks, find_iteration, masked_crops, unpack_masks
from gs_pipeline.semantic_vae import ClipVAE, load_cached_embeddings, load_vae, train_vae
from gs_pipeline.semantic_loss import embedding_distillation_loss
from gs_pipeline.semantic_train import uniform_sample_views


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


def test_dataset_images_respects_split(tmp_path: Path):
    images = tmp_path / "images"
    sparse = tmp_path / "sparse" / "0"
    images.mkdir()
    sparse.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (2, 2)).save(images / name)
    (sparse / "test.txt").write_text("b.jpg\n")
    assert [path.name for path in dataset_images(tmp_path, "train")] == ["a.jpg"]
    assert [path.name for path in dataset_images(tmp_path, "test")] == ["b.jpg"]
    assert [path.name for path in dataset_images(tmp_path, "all")] == ["a.jpg", "b.jpg"]


def test_filter_and_pack_masks():
    masks = np.zeros((3, 8, 10), dtype=bool)
    masks[0, 1:5, 1:5] = True
    masks[1, 1:5, 1:5] = True  # Lower-scoring duplicate.
    masks[2, 0, 0] = True      # Too small.
    kept, scores, boxes = filter_masks(masks, np.array([0.9, 0.8, 0.99]), 4, 0.95, 0.9, 10)
    assert kept.shape == (1, 8, 10)
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_array_equal(boxes, [[1, 1, 5, 5]])
    record = {"mask_width": np.array(10), "masks_packed": np.packbits(kept, axis=-1)}
    np.testing.assert_array_equal(unpack_masks(record), kept)


def test_full_mask_input_is_literal_image_product():
    image = Image.new("RGB", (2, 2), (10, 20, 30))
    masks = np.array([[[True, False], [False, False]]])
    boxes = np.array([[0, 0, 1, 1]])
    result = np.asarray(masked_crops(image, masks, boxes, 0.0, "full")[0])
    np.testing.assert_array_equal(result[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(result[0, 1], [0, 0, 0])


def test_find_latest_iteration(tmp_path: Path):
    from plyfile import PlyElement, PlyData

    vertex = np.array([(0.0, 0.0, 0.0)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for iteration in (100, 300):
        directory = tmp_path / "point_cloud" / f"iteration_{iteration}"
        directory.mkdir(parents=True)
        PlyData([PlyElement.describe(vertex, "vertex")]).write(directory / "point_cloud.ply")
    value, path = find_iteration(tmp_path, -1)
    assert value == 300
    assert path.parent.name == "iteration_300"


def test_clip_vae_shapes():
    import torch

    model = ClipVAE(input_dim=12, latent_dim=3)
    reconstruction, mu, logvar, latent = model(torch.randn(5, 12))
    assert reconstruction.shape == (5, 12)
    assert mu.shape == logvar.shape == latent.shape == (5, 3)
    assert model.encode(torch.randn(5, 12)).shape == (5, 3)


def test_train_and_load_vae(tmp_path: Path):
    records = tmp_path / "cache" / "records"
    records.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.savez_compressed(records / "a.npz", embeddings=rng.normal(size=(8, 12)).astype(np.float16))
    np.savez_compressed(records / "b.npz", embeddings=rng.normal(size=(7, 12)).astype(np.float16))
    assert load_cached_embeddings(tmp_path / "cache").shape == (15, 12)
    checkpoint = train_vae(
        cache_path=str(tmp_path / "cache"),
        output_path=str(tmp_path / "vae"),
        latent_dim=3,
        epochs=1,
        batch_size=5,
        device_name="cpu",
    )
    assert checkpoint.is_file()
    assert np.load(tmp_path / "vae" / "latents" / "a.npy").shape == (8, 3)
    assert load_vae(checkpoint).encode(__import__("torch").randn(2, 12)).shape == (2, 3)


def test_embedding_distillation_loss_matches_formula():
    import torch

    rendered = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True)
    masks = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    targets = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
    # Mask 0: (1-0)^2 + (3-1)^2 = 5
    # Mask 1: (2-2)^2 + (4-1)^2 = 9
    loss = embedding_distillation_loss(rendered, masks, targets)
    assert loss.item() == 14.0
    loss.backward()
    assert rendered.grad is not None
    per_mask = embedding_distillation_loss(rendered.detach(), masks, targets, reduction="none")
    torch.testing.assert_close(per_mask, torch.tensor([5.0, 9.0]))


def test_embedding_distillation_mask_mean_ignores_empty_masks():
    import torch

    rendered = torch.zeros(2, 2, 2)
    masks = torch.stack((torch.ones(2, 2), torch.zeros(2, 2)))
    targets = torch.ones(2, 2)
    loss = embedding_distillation_loss(rendered, masks, targets, reduction="mask_mean")
    assert loss.item() == 1.0


def test_uniform_view_sample_is_distinct_and_reproducible():
    views = list(range(100))
    first = uniform_sample_views(views, 50, seed=7)
    second = uniform_sample_views(views, 50, seed=7)
    assert first == second
    assert len(first) == len(set(first)) == 50
