from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .prepare import prepare_dataset


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
    else:
        call = ["--model", str(Path(args.model).resolve()), "--camera", str(Path(args.camera).resolve()),
                "--output", str(Path(args.output).resolve()), "--iteration", str(args.iteration)]
        if args.white_background: call.append("--white-background")
        if args.antialiasing: call.append("--antialiasing")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(GRAPHDECO) + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-m", "gs_pipeline.render_view", *call], check=True, env=env)


if __name__ == "__main__":
    main()
