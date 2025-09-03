import sys
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(root))
from src.utils.dwpose_util import draw_pose_select_v2

# 1) Config
W = H = 768
pose_folder = root / "assets/halfbody_demo" / "pose" / "01"

# 2) List frames (natural sort: 0,1,2,...10)
nsort = lambda xs: sorted(xs, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])
files = nsort([f for f in os.listdir(pose_folder) if f.endswith(".npy")])
print(f"{len(files)} frames in {pose_folder}")

# 3) Peek one dict
sample = np.load(pose_folder / files[0], allow_pickle=True).item()
print("keys:", list(sample))
print("draw_pose_params:", sample.get("draw_pose_params"))


# 4) Render helper (exactly like infer.py does, then paste into full canvas)
def render_full(pose_dict, W=768, H=768, ref_w=800):
    imh, imw, rb, re, cb, ce = pose_dict["draw_pose_params"]
    im_chw = draw_pose_select_v2(pose_dict, imh, imw, ref_w)  # CHW uint8
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[rb:re, cb:ce] = im_chw.transpose(1, 2, 0)  # HWC
    return canvas


# 5) Show two frames
fig, ax = plt.subplots(1, 2, figsize=(8, 5))
for k, i in enumerate([0, min(5, len(files) - 1)]):
    pose_i = np.load(pose_folder / files[i], allow_pickle=True).item()
    ax[k].imshow(render_full(pose_i, W, H));
    ax[k].set_title(f"frame {i}");
    ax[k].axis("off")
plt.show()

# 6) Build tensor [1, 3, T, H, W] for the pipeline
poses = []
for name in files:
    pose = np.load(pose_folder / name, allow_pickle=True).item()
    img = render_full(pose, W, H)  # HWC uint8
    poses.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.)
poses_tensor = torch.stack(poses, dim=1).unsqueeze(0)

print("poses_tensor shape:", tuple(poses_tensor.shape))  # -> (1, 3, T, H, W)
