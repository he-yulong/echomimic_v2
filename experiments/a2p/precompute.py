# experiments/a2p/precompute.py
import os, argparse, pathlib, re, numpy as np, torch, torchaudio
from tqdm import tqdm

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def load_pose_files(pose_dir: str):
    files = [f for f in os.listdir(pose_dir) if f.endswith(".npy")]
    files.sort(key=natural_key)
    return [os.path.join(pose_dir, f) for f in files]

def hands_from_pose_dict(p: dict):
    H = np.zeros((2,21,2), np.float32)
    S = np.zeros((2,21),   np.float32)
    h  = p.get("hands", None);  hs = p.get("hands_score", None)
    if h is None or hs is None: return H, S
    h  = np.asarray(h);  hs = np.asarray(hs)
    if h.ndim == 3 and h.shape[-1] >= 2: h = h[..., :2]
    if   h.shape == (42,2): h = h.reshape(2,21,2)
    elif h.shape == (21,2): h = np.stack([h, np.zeros_like(h)], 0)
    elif h.ndim == 3 and h.shape[1] == 21 and h.shape[2] == 2:
        if h.shape[0] > 2: h = h[:2]
        if h.shape[0] < 2: h = np.concatenate([h, np.zeros((2-h.shape[0],21,2),np.float32)], 0)
    else: h = H
    if   hs.ndim == 1 and hs.size == 42: hs = hs.reshape(2,21)
    elif hs.ndim == 2 and hs.shape[1] == 21:
        if hs.shape[0] > 2: hs = hs[:2]
        if hs.shape[0] < 2: hs = np.concatenate([hs, np.zeros((2-hs.shape[0],21),np.float32)], 0)
    else: hs = S
    h  = np.nan_to_num(h,  nan=0.0).astype(np.float32)
    hs = np.nan_to_num(hs, nan=0.0).astype(np.float32)
    return h, hs

def read_lists(paths):
    pairs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                a, d = ln.split(",")
                pairs.append((a, d))
    # unify by pose-dir stem (robust to weird audio file names)
    from collections import OrderedDict
    od = OrderedDict()
    for a, d in pairs:
        stem = pathlib.Path(d).name  # <STEM>
        od[stem] = (a, d)
    return list(od.items())  # [(stem, (audio, posedir)), ...]

def main():
    ap = argparse.ArgumentParser("Precompute mels and pose packs to disk")
    ap.add_argument("--lists", nargs="+", required=True, help="CSV(s): audio,pose_dir")
    ap.add_argument("--out_root", default="EMTD_dataset/features", help="Output root")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    mel_dir  = out_root / "mels"
    pose_dir = out_root / "poses"
    mel_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    # torchaudio transforms
    mel_tx  = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sr, n_fft=1024,
        hop_length=round(args.sr/args.fps),
        n_mels=80, f_min=50, f_max=7600, power=2.0, center=False
    )
    amp2db  = torchaudio.transforms.AmplitudeToDB()

    items = read_lists(args.lists)
    print(f"Unique clips: {len(items)}")

    for stem, (audio, pdir) in tqdm(items, ncols=100):
        mel_out  = mel_dir  / f"{stem}.pt"
        pose_out = pose_dir / f"{stem}.npz"

        # -------- mel
        if args.overwrite or not mel_out.exists():
            wav, sr = torchaudio.load(audio)
            if sr != args.sr:
                wav = torchaudio.functional.resample(wav, sr, args.sr)
            wav = wav.mean(0, keepdim=True)  # mono
            mel = amp2db(mel_tx(wav)).squeeze(0).transpose(0,1).contiguous()  # [Tm,80]
            torch.save({"mel": mel}, mel_out)

        # -------- pose pack
        if args.overwrite or not pose_out.exists():
            files = load_pose_files(pdir)
            xy, cf = [], []
            for f in files:
                d = np.load(f, allow_pickle=True).item()
                h, hs = hands_from_pose_dict(d)
                xy.append(h); cf.append(hs)
            xy = np.stack(xy, 0).astype(np.float32)   # [Tf,2,21,2]
            cf = np.stack(cf, 0).astype(np.float32)   # [Tf,2,21]
            np.savez_compressed(pose_out, xy=xy, cf=cf)

    print(f"Done.\nMels  → {mel_dir}\nPoses → {pose_dir}")

if __name__ == "__main__":
    main()
