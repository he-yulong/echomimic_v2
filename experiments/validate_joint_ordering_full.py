# Visualize a frame with skeleton lines (body + hands + face)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# repo utils (make sure repo root is on sys.path if you're in /experiments)
from src.utils.dwpose_util import draw_bodypose_white, draw_handpose, draw_facepose

# ---- pick a frame ----
npy_path = Path("../assets/halfbody_demo/pose/01/100.npy")  # change if needed
pose = np.load(npy_path, allow_pickle=True).item()

# ---- canvas size (pixels) ----
H, W = 768, 768

# white background helps lines pop
canvas = np.full((H, W, 3), 255, dtype=np.uint8)

# ---- draw body (OpenPose-18 ordering), hands (21), and face (68) ----
if "bodies" in pose and pose["bodies"] is not None:
    b = pose["bodies"]
    canvas = draw_bodypose_white(canvas, b["candidate"], b["subset"], b["score"])

if "hands" in pose and pose["hands"] is not None:
    canvas = draw_handpose(canvas, pose["hands"], pose["hands_score"])

if "faces" in pose and pose["faces"] is not None:
    canvas = draw_facepose(canvas, pose["faces"], pose["faces_score"])

# OpenCV draws in BGR; matplotlib wants RGB → flip channels for display
plt.figure(figsize=(6,6))
plt.imshow(canvas[..., ::-1])
plt.title("Skeleton overlay (Body + Hands + Face)")
plt.axis("off")
plt.show()
