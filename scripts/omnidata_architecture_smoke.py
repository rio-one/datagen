"""Architecture-only check without downloaded weights. Outputs are NOT dataset priors."""
import sys
import os
from pathlib import Path
import torch
import timm

sys.path.insert(0,str(Path(os.environ['DATAGEN_ROOT'])/'repos/omnidata/omnidata_tools/torch'))
original = timm.create_model
def offline(*args, **kwargs):
    kwargs['pretrained'] = False
    return original(*args, **kwargs)
timm.create_model = offline
from modules.midas.dpt_depth import DPTDepthModel
for task, channels in [('depth',1),('normal',3)]:
    model = DPTDepthModel(backbone='vitb_rn50_384',num_channels=channels).cuda().eval()
    with torch.inference_mode():
        y=model(torch.zeros(1,3,384,384,device='cuda'))
    assert y.shape == ((1,384,384) if channels == 1 else (1,3,384,384)), y.shape
    print(task,'random-initialized architecture GPU forward PASS',tuple(y.shape),flush=True)
    del model,y
    torch.cuda.empty_cache()

