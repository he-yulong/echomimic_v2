import os
from pathlib import Path
import numpy as np
import torch

from .wholebody import Wholebody

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Resolve weights relative to this file ---
REPO_ROOT = Path(__file__).resolve().parents[3]  # .../echomimic_v2
WEIGHTS_DIR = REPO_ROOT / "pretrained_weights" / "DWPose"

DET_ONNX = os.environ.get("YOLOX_ONNX", str(WEIGHTS_DIR / "yolox_l.onnx"))
POSE_ONNX = os.environ.get("DWPOSE_ONNX", str(WEIGHTS_DIR / "dw-ll_ucoco_384.onnx"))


class DWposeDetector:
    """
    A pose detect method for image-like data.

    Parameters:
        model_det: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/yolox_l.onnx
        model_pose: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx
        device: (str) 'cpu' or 'cuda:{device_id}'
    """

    def __init__(self, model_det, model_pose, device='cuda'):
        self.args = model_det, model_pose, device

    def release_memory(self):
        if hasattr(self, 'pose_estimation'):
            del self.pose_estimation
            import gc;
            gc.collect()

    def __call__(self, oriImg):
        if not hasattr(self, 'pose_estimation'):
            self.pose_estimation = Wholebody(*self.args)

        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, score = self.pose_estimation(oriImg)
            nums, _, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            subset = score[:, :18].copy()
            for i in range(len(subset)):
                for j in range(len(subset[i])):
                    if subset[i][j] > 0.3:
                        subset[i][j] = int(18 * i + j)
                    else:
                        subset[i][j] = -1

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            faces_score = score[:, 24:92]
            hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

            bodies = dict(candidate=body, subset=subset, score=score[:, :18])
            pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

            return pose


dwpose_detector = DWposeDetector(
    model_det=DET_ONNX,
    model_pose=POSE_ONNX,
    device=device)
print("dwpose_detector init ok", device, "\n  det:", DET_ONNX, "\n  pose:", POSE_ONNX)
