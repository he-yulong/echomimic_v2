# echomimic_v2/experiments/a2p/data_simple.py
import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Literal
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import torch
import pathlib
from collections import OrderedDict


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_pose_dicts(pose_dir: str) -> List[str]:
    # Lists all *.npy files in pose_dir.
    # Sorts them using natural_key so frames go in time order (0.npy, 1.npy, 2.npy, ...).
    # Returns their full paths.
    files = [f for f in os.listdir(pose_dir) if f.endswith(".npy")]
    files.sort(key=natural_key)
    return [os.path.join(pose_dir, f) for f in files]


def hands_from_pose_dict(p: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return coerced (2,21,2) hands & (2,21) scores in float32."""
    H = np.zeros((2, 21, 2), np.float32)
    S = np.zeros((2, 21), np.float32)

    h = p.get("hands", None)
    hs = p.get("hands_score", None)
    if h is None or hs is None:
        return H, S

    h = np.asarray(h)
    hs = np.asarray(hs)

    # keep x,y only
    if h.ndim == 3 and h.shape[-1] >= 2:
        h = h[..., :2]

    # normalize to (2,21,2)
    if h.shape == (42, 2):
        h = h.reshape(2, 21, 2)
    elif h.shape == (21, 2):
        h = np.stack([h, np.zeros_like(h)], 0)
    elif h.ndim == 3 and h.shape[1] == 21 and h.shape[2] == 2:
        if h.shape[0] > 2:   h = h[:2]
        if h.shape[0] < 2:   h = np.concatenate([h, np.zeros((2 - h.shape[0], 21, 2), np.float32)], 0)
    else:
        h = H

    # scores → (2,21)
    if hs.ndim == 1 and hs.size == 42:
        hs = hs.reshape(2, 21)
    elif hs.ndim == 2 and hs.shape[1] == 21:
        if hs.shape[0] > 2: hs = hs[:2]
        if hs.shape[0] < 2: hs = np.concatenate([hs, np.zeros((2 - hs.shape[0], 21), np.float32)], 0)
    else:
        hs = S

    h = np.nan_to_num(h, nan=0.0).astype(np.float32)
    hs = np.nan_to_num(hs, nan=0.0).astype(np.float32)
    return h, hs


@dataclass
class A2PConfig:
    fps: int = 24
    sample_rate: int = 16000
    win_T: int = 12
    hop_T: int = 3
    heat_H: int = 256
    heat_W: int = 256


def gaussian_splats(points_xy01: np.ndarray, conf: np.ndarray, H: int, W: int, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    heat = np.zeros((H, W), np.float32)
    for (x01, y01), c in zip(points_xy01, conf):
        if c <= 0: continue
        x, y = x01 * (W - 1), y01 * (H - 1)
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
        heat = np.maximum(heat, (c * g).astype(np.float32))
    return np.clip(heat, 0, 1)


class A2PDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], cfg: A2PConfig,
                 mel_root: str | None = "EMTD_dataset/features/mels",
                 pose_root: str | None = "EMTD_dataset/features/poses"):
        self.pairs = pairs
        self.cfg = cfg
        self.items = []
        for audio, pdir in pairs:
            T = len(load_pose_dicts(pdir))  # how many frames
            # T - win_T + 1 is the number of valid start indices (the last valid start is T - win_T).
            for t0 in range(0, max(0, T - cfg.win_T + 1), cfg.hop_T):
                if t0 < 0:
                    raise RuntimeError('bad t0!')
                self.items.append((audio, pdir, t0))
        # Step between consecutive analysis frames. It’s chosen so you get ≈1 mel frame per video frame.
        # TODO: round(cfg.sample_rate / cfg.fps) -> maybe fps=25 is better
        # Typical shapes:
        # Input wav: [channels, num_samples] (here we pass [1, N]).
        # Output mel_power: [channels, n_mels, Tm] → with your config: [1, 80, Tm], where Tm ≈ num_samples / hop_length.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=1024,
            hop_length=round(cfg.sample_rate / cfg.fps),
            # hop_length=math.ceil(cfg.sample_rate / cfg.fps),
            n_mels=80, f_min=50, f_max=7600, power=2.0, center=False
        )
        # torchaudio.transforms.Spectrogram
        # self.amplog = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
        self.amplog = torchaudio.transforms.AmplitudeToDB()
        # NEW: effective mel FPS (how many mel frames per second)
        self.mel_hop = int(self.mel.hop_length)
        self.mel_fps = self.cfg.sample_rate / self.mel_hop  # float
        self._mel_cache = OrderedDict()  # key: stem -> Tensor [Tm,80]
        self._pose_cache = OrderedDict()  # key: stem -> (xy [Tf,2,21,2], cf [Tf,2,21])

    def __len__(self) -> int:
        return len(self.items)

    def _pose_idx_to_mel_idx(self, t0: int, Tm: int) -> int:
        # Pose frame index -> time (sec)
        t_sec = t0 / self.cfg.fps
        # Time -> mel frame index (round to nearest)
        start = int(round(t_sec * self.mel_fps))
        # Clamp into valid windowable range
        start = max(0, min(start, max(0, Tm - self.cfg.win_T)))
        return start

    def _stem(self, pdir: str):
        return pathlib.Path(pdir).name

    def _mel_for(self, wav_path: str, pdir: str):
        stem = self._stem(pdir)
        m = self._mel_cache.get(stem)
        if m is not None:
            return m
        audio_data, sr = torchaudio.load(wav_path)
        if sr != self.cfg.sample_rate:
            raise RuntimeWarning(f"{sr} != self.cfg.sample_rate")
        audio_data = audio_data.mean(0, keepdim=True)  # 2 channels -> 1 channel
        power = self.mel(audio_data)  # [1, N] -> [1, n_mels, Tm] → typically [1, 80, Tm]
        # Converts power to decibels, clamped so silence is around −80 dB instead of -inf.
        m = self.amplog(power)  # [1, 80, Tm] -> [1, 80, Tm]
        # Removes the channel dim (size 1).
        m = m.squeeze(0)  # [1, 80, Tm] -> [80, Tm]
        m = m.transpose(0, 1)  # [80, Tm] -> [Tm, 80]
        # After transpose, memory is non-contiguous.
        # This makes a contiguous copy so later .view/slicing ops are fast and safe.
        m = m.contiguous()
        # m = self.amplog(self.mel(wav)).squeeze(0).transpose(0, 1).contiguous()
        # m = torch.nan_to_num(m, neginf=-80.0, posinf=0.0)
        # Now check dB mels for non-finite values
        finite = torch.isfinite(m)
        if not finite.all():
            neginf = torch.isinf(m) & (m < 0)
            posinf = torch.isinf(m) & (m > 0)
            nans = torch.isnan(m)
            # which time frames (columns) are bad
            bad_t_mask = (~finite).any(dim=1) if m.dim() == 3 else (~finite).any(dim=0)  # handle [1,80,T] vs [80,T]
            # normalize to [T] no matter the layout
            if m.dim() == 3:  # [1, 80, T]
                T = m.shape[-1]
                bad_t = (~finite).any(dim=1).any(dim=1)  # -> [1, T] -> [T]
                bad_idx = torch.nonzero(bad_t.squeeze(0)).squeeze(1).tolist()[:20]
            else:  # [T, 80]
                T = m.shape[0]
                bad_t = (~finite).any(dim=1)  # -> [T]
                bad_idx = torch.nonzero(bad_t).squeeze(1).tolist()[:20]
            print(f"[MEL CHECK] non-finite mel dB in {wav_path}", flush=True)
            print(f"  shape={tuple(m.shape)}  finite={finite.sum().item()}/{m.numel()}", flush=True)
            print(f"  -inf={neginf.sum().item()}  +inf={posinf.sum().item()}  NaN={nans.sum().item()}", flush=True)
            print(f"  bad time indices (first 20): {bad_idx}", flush=True)
            import sys
            # hard exit so you can inspect the file immediately
            sys.exit(1)
        # TODO: may be ? NAN fix
        # self._mel_cache[stem] = m
        self._mel_cache[stem] = m.clone()  # cache an immutable copy
        return m

    def _pose_for(self, pdir: str):
        stem = self._stem(pdir)
        pc = self._pose_cache.get(stem)
        if pc is not None:
            return pc

        hands_coordinates, hands_scores = [], []
        data = load_pose_dicts(pdir)
        for pf in data:
            pose_frame = np.load(pf, allow_pickle=True).item()
            # hands_coordinate: A NumPy array of shape (2, 21, 2). (hand, landmarks, (x,y))
            # hands_score: (2, 21)
            hands_coordinate, hands_score = hands_from_pose_dict(pose_frame)
            hands_coordinates.append(hands_coordinate)
            hands_scores.append(hands_score)
        hands_coordinates = np.stack(hands_coordinates, 0)  # [Tf,2,21,2]
        hands_scores = np.stack(hands_scores, 0)  # _pose_for, data_simple.py:141

        pc = (hands_coordinates, hands_scores)
        self._pose_cache[stem] = pc
        return pc

    def __getitem__(self, i):
        audio, pdir, t0 = self.items[i]
        # audio window
        mel_full = self._mel_for(audio, pdir)  # TODO: inefficient
        Tm = mel_full.shape[0]
        start = self._pose_idx_to_mel_idx(t0, Tm)
        # feats = mel_full[start:start + self.cfg.win_T]  # TODO: boundary check
        feats = mel_full[start:start + self.cfg.win_T].clone()  # <— decouple from mel_full
        if feats.shape[0] < self.cfg.win_T:
            raise RuntimeError("should not happen")
        T = feats.shape[0]

        # pose window
        xy_full, cf_full = self._pose_for(pdir)  # TODO: inefficient
        hands_xy = xy_full[t0:t0 + T]
        hands_cf = cf_full[t0:t0 + T]
        if hands_xy.shape[0] < T:
            raise RuntimeError("something wrong with hands_xy")

        # pick an initial pose for the window (previous frame if available)
        if t0 > 0:
            init_pose = xy_full[t0 - 1]  # [2,21,2]
        else:
            init_pose = xy_full[t0]  # fallback: first frame (only for the very first window)

        init_pose = torch.from_numpy(init_pose).float()  # [2,21,2]

        return dict(
            audio=feats,
            target=torch.from_numpy(hands_xy).float(),
            conf=torch.from_numpy(hands_cf).float(),
            init_pose=init_pose
        )


def make_pairs(list_file: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(list_file, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            a, p = ln.split(",")
            pairs.append((a, p))
    return pairs


def make_loaders(
        train_list: str,
        val_list: str,
        cfg: A2PConfig,
        bs: int = 16,
        num_workers: int = 4,
        mel_root: str | None = "EMTD_dataset/features/mels",
        pose_root: str | None = "EMTD_dataset/features/poses",
        filter_missing: bool = True,
):
    train_pairs = make_pairs(train_list)  # [(*.wav, pose dir), ...]
    val_pairs = make_pairs(val_list)

    train_ds = A2PDataset(train_pairs, cfg, mel_root=mel_root, pose_root=pose_root)
    val_ds = A2PDataset(val_pairs, cfg, mel_root=mel_root, pose_root=pose_root)
    use_workers = max(0, int(num_workers))
    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=use_workers
    )
    val_dl = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=use_workers
    )

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Workers: {use_workers}")

    return train_dl, val_dl


if __name__ == '__main__':
    ap = argparse.ArgumentParser("Smoke test for A2PDataset/DataLoader")
    ap.add_argument("--mel_root", type=str, default="../../EMTD_dataset/features/mels")
    ap.add_argument("--pose_root", type=str, default="../../EMTD_dataset/features/poses")
    ap.add_argument("--no_filter_missing", action="store_true")
    ap.add_argument("--train_list", default="../../EMTD_dataset/lists/train.csv",
                    help="CSV with 'audio.wav,pose_dir' per line")
    ap.add_argument("--val_list", default="../../EMTD_dataset/lists/val.csv",
                    help="CSV with 'audio.wav,pose_dir' per line")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3, help="number of batches to time")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    # win_T = window length in frames.
    # Each training sample is a chunk of win_T consecutive frames (and the matching audio mels).
    # Example: at fps=24, win_T=12 ≈ 0.5 s of context.
    ap.add_argument("--win_T", type=int, default=12)
    # hop_T = stride between windows in frames.
    # It tells how far the next window starts from the previous one.
    # Example: hop_T=3 at 24 fps ⇒ move by 3 frames (0.125 s), so windows overlap by 75%.
    ap.add_argument("--hop_T", type=int, default=3)
    # the heatmap (height, width) in pixels used in heatmap mode.
    # With the current A2PHeatmapHead the decoder upsamples from a 64×64 latent to 128×128 → 256×256.
    # So the head effectively produces 256×256. Keep --heat_hw 256 256 (the default).
    # If you want other sizes, you’d need to change the upsampling stages to match
    # (e.g., drop one stage for 128×128, add one for 512×512), and then keep the dataset’s heat_H/W in sync.
    ap.add_argument("--heat_hw", type=int, nargs=2, default=[256, 256])
    args = ap.parse_args()

    cfg = A2PConfig(
        fps=args.fps, win_T=args.win_T, hop_T=args.hop_T,
        heat_H=args.heat_hw[0], heat_W=args.heat_hw[1]
    )
    print(cfg)

    train_dl, _ = make_loaders(
        args.train_list, args.val_list, cfg,
        bs=args.bs, num_workers=args.num_workers,
        mel_root=None, pose_root=None,
        filter_missing=False
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} | bs={args.bs} | workers={args.num_workers} "
          f"| win_T={cfg.win_T} hop_T={cfg.hop_T} | heat_hw=({cfg.heat_H},{cfg.heat_W})")
    print(f"dataset size (windows): {len(train_dl.dataset)}")
    it = iter(train_dl)
    times = []
    import time

    for i in range(args.iters):
        t1 = time.time()
        batch = next(it)
        if dev == "cuda":
            # touch tensors to exercise pinned-memory transfer
            batch["audio"] = batch["audio"].to(dev, non_blocking=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        times.append(t2 - t1)
