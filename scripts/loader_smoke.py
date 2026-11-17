"""Read existing baseline loaders without writing to baseline repositories."""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
parser=argparse.ArgumentParser()
parser.add_argument('--kind',choices=['dp','obj'],required=True)
parser.add_argument('--data-root',required=True)
args=parser.parse_args()
code=Path('/home/seed/wangruiyao')/('DP-Recon' if args.kind=='dp' else 'objectsdf_plus')/'code'
sys.path.insert(0,str(code))
from datasets.scene_dataset import SceneDatasetDN_segs
kwargs=dict(data_dir='custom',img_res=[16,16],scan_id=1,center_crop_type='no_crop',use_mask=True,num_views=-1)
if args.kind=='dp':
    kwargs['data_root_dir']=args.data_root
dataset=SceneDatasetDN_segs(**kwargs)
assert len(dataset)==10
item=dataset[0]
assert len(item)==3
assert len(dataset.label_mapping)==2
assert np.isfinite(dataset.pose_all[0].numpy()).all()
print(args.kind,'actual loader PASS: 10 images / 1 foreground instance / CPU dataset read')
