# experiments/construct_npy.py
from pathlib import Path
import cv2, numpy as np
from src.models.dwpose.dwpose_detector import DWposeDetector

root = Path(__file__).resolve().parents[1]  # repo root
weights_dir = root / "pretrained_weights" / "DWpose"
model_det = weights_dir / "yolox_l.onnx"  # detector
model_pose = weights_dir / "dw-ll_ucoco_384.onnx"  # pose head

assert model_det.exists(), f"Missing: {model_det}"
assert model_pose.exists(), f"Missing: {model_pose}"

det = DWposeDetector(str(model_det), str(model_pose), device='cuda')

video = root / "assets" / "halfbody_demo" / "video" / "sample.mp4"
out_dir = root / "outputs" / "pose" / "sample"
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))
i = 0
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    # H, W = frame_bgr.shape[:2]
    TARGET_H = TARGET_W = 768  # desired pose canvas size
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    pose = det(frame_rgb)  # dict with bodies/hands/faces(+scores)
    pose["draw_pose_params"] = [TARGET_H, TARGET_W, 0, TARGET_H, 0, TARGET_W]  # (imh, imw, rb, re, cb, ce) -> full frame

    np.save(out_dir / f"{i}.npy", pose, allow_pickle=True)
    i += 1

cap.release()
print(f"Saved {i} frames to: {out_dir}")
