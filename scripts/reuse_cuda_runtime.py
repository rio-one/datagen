"""Copy exact-version runtime distributions into an isolated environment. Never link or modify source."""
from importlib.metadata import distributions
from pathlib import Path
import json
import os
import shutil

root = Path('/home/seed/wangruiyao/datagen')
source_env = Path('/home/seed/miniconda3/envs/dprecon')
source_site = source_env / 'lib/python3.10/site-packages'
dest_env = Path('/home/seed/miniconda3/envs/datagen')
dest_site = dest_env / 'lib/python3.10/site-packages'
requirements = {
    'nvidia-cublas-cu12': '12.8.3.14', 'nvidia-cuda-cupti-cu12': '12.8.57',
    'nvidia-cuda-nvrtc-cu12': '12.8.61', 'nvidia-cuda-runtime-cu12': '12.8.57',
    'nvidia-cudnn-cu12': '9.7.1.26', 'nvidia-cufft-cu12': '11.3.3.41',
    'nvidia-cufile-cu12': '1.13.0.11', 'nvidia-curand-cu12': '10.3.9.55',
    'nvidia-cusolver-cu12': '11.7.2.55', 'nvidia-cusparse-cu12': '12.5.7.53',
    'nvidia-cusparselt-cu12': '0.6.3', 'nvidia-nccl-cu12': '2.26.2',
    'nvidia-nvjitlink-cu12': '12.8.61', 'nvidia-nvtx-cu12': '12.8.55',
    'triton': '3.3.1'
}
found = {d.metadata['Name'].lower(): d for d in distributions(path=[str(source_site)])}
log = []
for name, version in requirements.items():
    dist = found[name]
    if dist.version != version:
        raise RuntimeError(f'Version mismatch: {name} expected {version}, actual {dist.version}')
    count, size = 0, 0
    for record in dist.files:
        if '__pycache__' in record.parts or record.suffix == '.pyc':
            continue
        source = dist.locate_file(record).resolve()
        destination = (dest_site / record).resolve()
        if not source.is_relative_to(source_env) or not destination.is_relative_to(dest_env):
            raise RuntimeError(f'Path outside explicit environments: {record}')
        if not source.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        shutil.copymode(source, destination)
        count += 1
        size += destination.stat().st_size
    log.append(dict(package=name, version=version, files=count, bytes=size, method='independent copyfile; no symlinks/hardlinks; source read only'))
    print(name, version, count, flush=True)
(root/'config/runtime-copy-provenance.json').write_text(json.dumps(dict(source_environment=str(source_env), destination_environment=str(dest_env), packages=log), indent=2)+'\n')
