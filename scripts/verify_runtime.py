"""One small GPU correctness check. Does not benchmark or traverse files."""
import json
import subprocess
from pathlib import Path
import os

import torch
import torchvision
import torchaudio
import xformers

assert torch.__version__.split('+')[0] == '2.7.1'
assert torchvision.__version__.split('+')[0] == '0.22.1'
assert torchaudio.__version__.split('+')[0] == '2.7.1'
assert xformers.__version__ == '0.0.31'
assert torch.version.cuda == '12.8'
assert torch.cuda.get_device_capability(0) == (12, 0)
assert 'sm_120' in torch.cuda.get_arch_list()
a = torch.randn(128, 128, device='cuda')
b = a @ a.T
assert torch.isfinite(b).all().item()
from torchvision.ops import nms
boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], device='cuda')
assert nms(boxes, torch.tensor([1.,0.5], device='cuda'), 0.5).numel() == 1
xf = {}
for mode in ('default', 'cutlass'):
    code = "import torch,xformers.ops; q=torch.randn(1,64,4,32,device='cuda',dtype=torch.float16); "
    code += "y=xformers.ops.memory_efficient_attention(q,q,q" + (",op=xformers.ops.MemoryEfficientAttentionCutlassOp" if mode == 'cutlass' else '') + "); torch.cuda.synchronize(); assert torch.isfinite(y).all().item(); print('PASS')"
    # Native kernels can exit the interpreter; keep this optional check in a child process.
    proc = subprocess.run([os.sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    xf[mode] = dict(works=proc.returncode == 0, returncode=proc.returncode, output=(proc.stdout+proc.stderr)[-1800:])
sdpa_q = torch.randn(1, 4, 64, 32, device='cuda', dtype=torch.float16)
assert torch.isfinite(torch.nn.functional.scaled_dot_product_attention(sdpa_q,sdpa_q,sdpa_q)).all().item()
result = dict(python=os.sys.version, torch=torch.__version__, torchvision=torchvision.__version__, torchaudio=torchaudio.__version__, xformers=xformers.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0), architectures=torch.cuda.get_arch_list(), gpu_matmul=True, torchvision_gpu_nms=True, pytorch_sdpa=True, xformers_forward=xf, nvcc=subprocess.check_output(['nvcc','--version'], text=True))
Path(os.environ['DATAGEN_ROOT'], 'config', 'runtime-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
