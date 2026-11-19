from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .prepare import prepare_dataset
from .semantic_prepare import prepare_semantics
from .semantic_vae import train_vae


ROOT = Path(__file__).resolve().parents[1]
GRAPHDECO = ROOT / "third_party" / "gaussian-splatting"


def _run(script: str, args: list[str]) -> None:
    if not (GRAPHDECO / script).is_file():
        raise SystemExit("GraphDECO source missing; run scripts/bootstrap.sh")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(GRAPHDECO) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(GRAPHDECO / script), *args], check=True, env=env, cwd=GRAPHDECO)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gs-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="convert cameras.json into a GraphDECO dataset")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed-points", type=int, default=100_000)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--overwrite", action="store_true")
    train = commands.add_parser("train", help="train a GraphDECO model")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--iterations", type=int, default=30_000)
    train.add_argument("--eval", action="store_true")
    train.add_argument("extra", nargs=argparse.REMAINDER)
    render = commands.add_parser("render", help="render dataset train/test cameras")
    render.add_argument("--model", required=True)
    render.add_argument("--iteration", type=int, default=-1)
    render.add_argument("--split", choices=["train", "test", "all"], default="test")
    render.add_argument("extra", nargs=argparse.REMAINDER)
    view = commands.add_parser("render-view", help="render one arbitrary camera")
    view.add_argument("--model", required=True)
    view.add_argument("--camera", required=True)
    view.add_argument("--output", required=True)
    view.add_argument("--iteration", type=int, default=-1)
    view.add_argument("--white-background", action="store_true")
    view.add_argument("--antialiasing", action="store_true")
    semantic = commands.add_parser(
        "semantic-prepare",
        help="freeze a trained snapshot and cache SAM masks plus CLIP embeddings",
    )
    semantic.add_argument("--model", required=True)
    semantic.add_argument("--data", required=True)
    semantic.add_argument("--output", required=True)
    semantic.add_argument("--iteration", type=int, default=-1)
    semantic.add_argument("--split", choices=["train", "test", "all"], default="train")
    semantic.add_argument("--sam-model", default="facebook/sam2-hiera-base-plus")
    semantic.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    semantic.add_argument("--device", default="cuda")
    semantic.add_argument("--points-per-batch", type=int, default=64)
    semantic.add_argument("--clip-batch-size", type=int, default=32)
    semantic.add_argument("--min-mask-area", type=int, default=256)
    semantic.add_argument("--max-mask-fraction", type=float, default=0.95)
    semantic.add_argument("--pred-iou-threshold", type=float, default=0.88)
    semantic.add_argument("--duplicate-iou-threshold", type=float, default=0.90)
    semantic.add_argument("--max-masks", type=int, default=64)
    semantic.add_argument("--crop-padding", type=float, default=0.10)
    semantic.add_argument("--clip-mask-mode", choices=["full", "crop"], default="full")
    semantic.add_argument("--views", type=int, default=50, help="uniformly sampled views; use 0 for all")
    semantic.add_argument("--view-seed", type=int, default=0)
    vae = commands.add_parser(
        "semantic-train-vae",
        help="compress cached CLIP features with a reconstruction-only VAE",
    )
    vae.add_argument("--cache", required=True)
    vae.add_argument("--output", required=True)
    vae.add_argument("--latent-dim", type=int, default=3)
    vae.add_argument("--epochs", type=int, default=50)
    vae.add_argument("--batch-size", type=int, default=256)
    vae.add_argument("--learning-rate", type=float, default=1e-3)
    vae.add_argument("--device", default="cuda")
    vae.add_argument("--seed", type=int, default=0)
    sem_train = commands.add_parser(
        "semantic-train",
        help="train three-channel Gaussian semantic vectors on uniformly sampled views",
    )
    sem_train.add_argument("--model", required=True)
    sem_train.add_argument("--data", required=True)
    sem_train.add_argument("--cache", required=True)
    sem_train.add_argument("--vae", required=True)
    sem_train.add_argument("--output", required=True)
    sem_train.add_argument("--iteration", type=int, default=-1)
    sem_train.add_argument("--views", type=int, default=50)
    sem_train.add_argument("--view-seed", type=int, default=0)
    sem_train.add_argument("--epochs", type=int, default=10)
    sem_train.add_argument("--learning-rate", type=float, default=1e-3)
    sem_train.add_argument("--init-std", type=float, default=0.01)
    sem_train.add_argument("--resolution", type=int, default=2)
    sem_train.add_argument("--mask-batch-size", type=int, default=8)
    sem_train.add_argument("--loss-reduction", choices=["sum", "mask_mean"], default="mask_mean")
    sem_train.add_argument("--preview-every", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_dataset(args.manifest, args.output, args.seed_points, args.seed, args.overwrite)
        print(result)
    elif args.command == "train":
        call = ["-s", str(Path(args.data).resolve()), "-m", str(Path(args.output).resolve()),
                "--iterations", str(args.iterations)]
        if args.eval:
            call.append("--eval")
        _run("train.py", call + args.extra)
    elif args.command == "render":
        call = ["-m", str(Path(args.model).resolve()), "--iteration", str(args.iteration)]
        if args.split == "test": call.append("--skip_train")
        if args.split == "train": call.append("--skip_test")
        _run("render.py", call + args.extra)
    elif args.command == "render-view":
        call = ["--model", str(Path(args.model).resolve()), "--camera", str(Path(args.camera).resolve()),
                "--output", str(Path(args.output).resolve()), "--iteration", str(args.iteration)]
        if args.white_background: call.append("--white-background")
        if args.antialiasing: call.append("--antialiasing")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(GRAPHDECO) + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-m", "gs_pipeline.render_view", *call], check=True, env=env)
    elif args.command == "semantic-prepare":
        result = prepare_semantics(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            iteration=args.iteration,
            split=args.split,
            sam_model=args.sam_model,
            clip_model=args.clip_model,
            device_name=args.device,
            points_per_batch=args.points_per_batch,
            clip_batch_size=args.clip_batch_size,
            min_mask_area=args.min_mask_area,
            max_mask_fraction=args.max_mask_fraction,
            pred_iou_threshold=args.pred_iou_threshold,
            duplicate_iou_threshold=args.duplicate_iou_threshold,
            max_masks=args.max_masks,
            crop_padding=args.crop_padding,
            clip_mask_mode=args.clip_mask_mode,
            views=args.views,
            view_seed=args.view_seed,
        )
        print(result)
    elif args.command == "semantic-train-vae":
        result = train_vae(
            cache_path=args.cache,
            output_path=args.output,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
            seed=args.seed,
        )
        print(result)
    else:
        call = [
            "--model", str(Path(args.model).resolve()),
            "--data", str(Path(args.data).resolve()),
            "--cache", str(Path(args.cache).resolve()),
            "--vae", str(Path(args.vae).resolve()),
            "--output", str(Path(args.output).resolve()),
            "--iteration", str(args.iteration),
            "--views", str(args.views),
            "--view-seed", str(args.view_seed),
            "--epochs", str(args.epochs),
            "--learning-rate", str(args.learning_rate),
            "--init-std", str(args.init_std),
            "--resolution", str(args.resolution),
            "--mask-batch-size", str(args.mask_batch_size),
            "--loss-reduction", args.loss_reduction,
            "--preview-every", str(args.preview_every),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(GRAPHDECO) + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-m", "gs_pipeline.semantic_train", *call], check=True, env=env)


if __name__ == "__main__":
    main()
