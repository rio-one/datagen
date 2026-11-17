"""Create an explicitly synthetic, non-planar textured scene for one COLMAP/SAM interface smoke test."""
import os
from pathlib import Path
import cv2
import numpy as np

root = Path(os.environ['DATAGEN_ROOT']) / 'work' / 'synthetic-smoke'
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(123)
textures = []
for seed in range(2):
    texture = rng.integers(0,256,(512,512,3),dtype=np.uint8)
    texture = cv2.GaussianBlur(texture,(5,5),0)
    for i in range(100):
        x,y = rng.integers(10,500,2)
        cv2.circle(texture,(int(x),int(y)),int(rng.integers(3,15)),tuple(int(v) for v in rng.integers(0,256,3)),2)
    textures.append(texture)
K = np.array([[320.,0,192],[0,320,192],[0,0,1]])
planes = [np.array([[-3,-2,4],[3,-2,4],[3,2,4],[-3,2,4]],float),np.array([[-.7,-.6,2.5],[.7,-.6,2.5],[.7,.6,2.5],[-.7,.6,2.5]],float)]
corners = np.array([[0,0],[511,0],[511,511],[0,511]],np.float32)
writer = cv2.VideoWriter(str(root/'synthetic.avi'), cv2.VideoWriter_fourcc(*'MJPG'),10,(384,384))
for i in range(60):
    C = np.array([-.45+.9*i/59,0,0])
    frame = np.full((384,384,3),128,np.uint8)
    for texture, plane in zip(textures,planes):
        points = (K @ (plane-C).T).T
        xy = (points[:,:2]/points[:,2:]).astype(np.float32)
        H = cv2.getPerspectiveTransform(corners,xy)
        warped = cv2.warpPerspective(texture,H,(384,384))
        mask = cv2.warpPerspective(np.ones((512,512),np.uint8),H,(384,384),flags=cv2.INTER_NEAREST)>0
        frame[mask] = warped[mask]
    writer.write(frame)
writer.release()
print(root/'synthetic.avi')
