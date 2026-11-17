"""Fallback for slow partial-clone lazy fetch: retrieve just the eight pinned inference source files."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import urllib.request

root = Path('/home/seed/wangruiyao/datagen')
repo = root/'repos/omnidata'
commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
prefix = 'omnidata_tools/torch/modules/midas/'
files = ['__init__.py','base_model.py','blocks.py','dpt_depth.py','midas_net.py','midas_net_custom.py','transforms.py','vit.py']

def fetch(name):
    path = repo/prefix/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        return dict(file=prefix+name,status='present')
    url = f'https://raw.githubusercontent.com/EPFL-VILAB/omnidata/{commit}/{prefix}{name}'
    try:
        with urllib.request.urlopen(url,timeout=30) as response:
            data=response.read()
        path.write_bytes(data)
        return dict(file=prefix+name,status='retrieved_at_pinned_commit',bytes=len(data))
    except Exception as exc:
        return dict(file=prefix+name,status='manual_source_required',error=str(exc),url=url)

with ThreadPoolExecutor(max_workers=4) as pool:
    result=list(pool.map(fetch,files))
(root/'config/omnidata-source.json').write_text(json.dumps(dict(commit=commit,files=result),indent=2)+'\n')
print(json.dumps(result,indent=2))
