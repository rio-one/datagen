#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$DATAGEN_ROOT"/{config,logs,repos,models,work,datasets,.cache}
printf 'envs_dirs:\n  - /home/seed/miniconda3/envs\npkgs_dirs:\n  - %s/.cache/conda\nchannel_priority: flexible\n' "$DATAGEN_ROOT" > "$CONDARC"
CONDA=/home/seed/miniconda3/bin/conda
if [[ ! -x "$DATAGEN_ENV/bin/python" ]]; then
  "$CONDA" create -y --prefix "$DATAGEN_ENV" --override-channels -c conda-forge python=3.10 pip
fi
if [[ ! -x "$DATAGEN_ENV/bin/nvcc" ]]; then
  "$CONDA" install -y --prefix "$DATAGEN_ENV" --override-channels -c nvidia -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-version=12.8
fi
python -m pip install --retries 0 --timeout 30 --no-deps --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 xformers==0.0.31
python - <<'PY'
from importlib.metadata import requires
from pathlib import Path
import os
req = sorted(set(r for p in ('torch', 'torchvision', 'torchaudio', 'xformers') for r in requires(p)))
Path(os.environ['DATAGEN_ROOT'], 'config', 'torch-runtime-requirements.txt').write_text('\n'.join(req) + '\n')
PY
INDEX="${DATAGEN_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
python -m pip install --retries 0 --timeout 30 --index-url "$INDEX" -r "$DATAGEN_ROOT/config/torch-runtime-requirements.txt"
python -m pip install --retries 0 --timeout 30 --index-url "$INDEX" 'numpy<2' 'opencv-python-headless<4.12' pillow scipy tqdm pyyaml 'pycolmap==3.12.6' 'timm==0.6.13' hydra-core iopath huggingface_hub einops matplotlib imageio scikit-image pytest
if [[ ! -d "$DATAGEN_ROOT/repos/sam2/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/sam2.git "$DATAGEN_ROOT/repos/sam2"
  git -C "$DATAGEN_ROOT/repos/sam2" sparse-checkout init --cone
  git -C "$DATAGEN_ROOT/repos/sam2" sparse-checkout set sam2
fi
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e "$DATAGEN_ROOT/repos/sam2"
if [[ ! -d "$DATAGEN_ROOT/repos/omnidata/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/EPFL-VILAB/omnidata.git "$DATAGEN_ROOT/repos/omnidata"
  git -C "$DATAGEN_ROOT/repos/omnidata" sparse-checkout init --cone
  git -C "$DATAGEN_ROOT/repos/omnidata" sparse-checkout set omnidata_tools/torch/modules/midas
fi
python -m pip freeze > "$DATAGEN_ROOT/config/installed-packages.txt"
printf '\nCORE_INSTALL_COMPLETE\n'
