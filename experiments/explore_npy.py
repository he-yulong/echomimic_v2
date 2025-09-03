from pathlib import Path
import numpy as np

pose_folder = Path.cwd().parent / "assets/halfbody_demo" / "pose" / "01" / "100.npy"
pose_folder = Path.cwd().parent / "outputs" / "pose" / "01" / "100.npy"
sample = np.load(pose_folder, allow_pickle=True).item()
print("keys:", list(sample))