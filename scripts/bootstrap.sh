#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -d "$ROOT/third_party/gaussian-splatting/.git" ]]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "$ROOT/third_party/gaussian-splatting"
else
  git -C "$ROOT/third_party/gaussian-splatting" submodule update --init --recursive
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc is required to build the GraphDECO CUDA extensions" >&2
  echo "The basic environment is still usable via: uv sync" >&2
  exit 1
fi

UV_PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$UV_PYTHON" ]]; then
  echo "Creating the uv environment at $ROOT/.venv"
  uv sync --project "$ROOT"
fi

if ! "$UV_PYTHON" - <<'PY'
import sys

try:
    import torch
except ImportError:
    print("error: PyTorch is not installed in the uv environment", file=sys.stderr)
    raise SystemExit(1)

if torch.version.cuda is None:
    print("error: the uv environment contains a CPU-only PyTorch build", file=sys.stderr)
    print("Install a CUDA-enabled PyTorch build before running bootstrap.sh.", file=sys.stderr)
    raise SystemExit(1)

print(f"Using Python: {sys.executable}")
print(f"Using PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
PY
then
  exit 1
fi

install_extension() {
  local source_path="$1"
  echo "Installing GraphDECO extension: $(basename "$source_path")"
  uv pip install \
    --python "$UV_PYTHON" \
    --no-build-isolation \
    --reinstall \
    "$source_path"
}

install_extension "$ROOT/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
install_extension "$ROOT/third_party/gaussian-splatting/submodules/simple-knn"
install_extension "$ROOT/third_party/gaussian-splatting/submodules/fused-ssim"

# Install only this project's editable package. Dependencies are managed by the
# existing uv environment so its CUDA-enabled PyTorch build is not replaced.
uv pip install \
  --python "$UV_PYTHON" \
  --no-deps \
  --editable "$ROOT"

"$UV_PYTHON" - <<'PY'
import diff_gaussian_rasterization
import fused_ssim
import simple_knn

print("GraphDECO CUDA extensions imported successfully.")
PY
