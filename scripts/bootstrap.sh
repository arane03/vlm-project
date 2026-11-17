#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "$ROOT/third_party/gaussian-splatting/.git" ]]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "$ROOT/third_party/gaussian-splatting"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc is required to build the GraphDECO CUDA extensions" >&2
  echo "The basic environment is still usable via: uv sync" >&2
  exit 1
fi
python -m pip install -e "$ROOT/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
python -m pip install -e "$ROOT/third_party/gaussian-splatting/submodules/simple-knn"
python -m pip install -e "$ROOT/third_party/gaussian-splatting/submodules/fused-ssim"
python -m pip install -e "$ROOT"
