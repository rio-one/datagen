#!/usr/bin/env bash
# Source this file only in the shell used for this tool.
export DATAGEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DATAGEN_ENV="/home/seed/miniconda3/envs/datagen"
export CONDARC="$DATAGEN_ROOT/config/condarc"
export CONDA_PKGS_DIRS="$DATAGEN_ROOT/.cache/conda"
export CONDA_ENVS_PATH="/home/seed/miniconda3/envs"
export PIP_CACHE_DIR="$DATAGEN_ROOT/.cache/pip"
export HF_HOME="$DATAGEN_ROOT/.cache/huggingface"
export TORCH_HOME="$DATAGEN_ROOT/.cache/torch"
export XDG_CACHE_HOME="$DATAGEN_ROOT/.cache/xdg"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$DATAGEN_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
export PIP_CONSTRAINT="$DATAGEN_ROOT/constraints.txt"
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_HOME="$DATAGEN_ENV"
export PATH="$DATAGEN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$DATAGEN_ENV/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$DATAGEN_ROOT/.cache/matplotlib"
export PYTHONPATH="$DATAGEN_ROOT/src"
