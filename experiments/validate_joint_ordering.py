import os, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

from src.utils.dwpose_util import draw_bodypose_white, draw_handpose, draw_facepose

# pick one frame
npy_path = Path("../assets/halfbody_demo/pose/01/100.npy")
pose = np.load(npy_path, allow_pickle=True).item()
H, W = 768, 768

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ------------ BODY (OpenPose-18) ------------
ax = axes[0]; ax.set_title("Body (OpenPose-18)")
canvas_body = np.full((H, W, 3), 255, dtype=np.uint8)

b = pose["bodies"]
canvas_body = draw_bodypose_white(canvas_body, b["candidate"], b["subset"], b["score"])

# (optional) label the 18 joints to verify ordering
if len(b["subset"]) > 0:
    idxs = b["subset"][0].astype(int)
    for j, idx in enumerate(idxs):
        if idx == -1:
            continue
        x, y = b["candidate"][idx]
        ax.text(x*W + 3, y*H, str(j), color="black", fontsize=8)

ax.imshow(canvas_body[..., ::-1])  # BGR->RGB for matplotlib
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

# ------------ LEFT HAND (21) ------------
ax = axes[1]; ax.set_title("Left hand (21)")
canvas_l = np.full((H, W, 3), 255, dtype=np.uint8)
if pose["hands"].shape[0] >= 1:
    # draw_handpose expects a LIST of hands and a LIST of scores
    canvas_l = draw_handpose(canvas_l, [pose["hands"][0]], [pose["hands_score"][0]])
    # label points
    for j, (x, y) in enumerate(pose["hands"][0]):
        ax.text(x*W + 3, y*H, str(j), color="black", fontsize=8)
ax.imshow(canvas_l[..., ::-1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

# ------------ RIGHT HAND (21) ------------
ax = axes[2]; ax.set_title("Right hand (21)")
canvas_r = np.full((H, W, 3), 255, dtype=np.uint8)
if pose["hands"].shape[0] >= 2:
    canvas_r = draw_handpose(canvas_r, [pose["hands"][1]], [pose["hands_score"][1]])
    for j, (x, y) in enumerate(pose["hands"][1]):
        ax.text(x*W + 3, y*H, str(j), color="black", fontsize=8)
ax.imshow(canvas_r[..., ::-1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

plt.tight_layout(); plt.show()
