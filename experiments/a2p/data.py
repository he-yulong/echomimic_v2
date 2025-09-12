# experiments/a2p/data.py
import torch
from dataclasses import dataclass
from typing import List, Tuple, Literal
import os, re, numpy as np, torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import pathlib
from collections import OrderedDict
import math


# --------- small helpers ---------
def a2p_collate(batch):
    return {k: default_collate([b[k] for b in batch]) for k in batch[0]}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_pose_dicts(pose_dir: str) -> List[str]:
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


def gaussian_splats(points_xy01: np.ndarray, conf: np.ndarray, H: int, W: int, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    heat = np.zeros((H, W), np.float32)
    for (x01, y01), c in zip(points_xy01, conf):
        if c <= 0: continue
        x, y = x01 * (W - 1), y01 * (H - 1)
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
        heat = np.maximum(heat, (c * g).astype(np.float32))
    return np.clip(heat, 0, 1)


# --------- dataset & loaders ---------
@dataclass
class A2PConfig:
    fps: int = 24
    sample_rate: int = 16000
    win_T: int = 12
    hop_T: int = 3
    heat_H: int = 256
    heat_W: int = 256
    mode: Literal["heatmap", "keypoints"] = "heatmap"


class A2PDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], cfg: A2PConfig,
                 mel_root: str | None = "EMTD_dataset/features/mels",
                 pose_root: str | None = "EMTD_dataset/features/poses"):
        self.pairs = pairs
        self.cfg = cfg
        self.items = []
        self.mel_root = pathlib.Path(mel_root) if mel_root else None
        self.pose_root = pathlib.Path(pose_root) if pose_root else None

        for audio, pdir in pairs:
            T = len(load_pose_dicts(pdir))
            for t0 in range(0, max(0, T - cfg.win_T + 1), cfg.hop_T):
                self.items.append((audio, pdir, t0))

        # transforms for fallback compute
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=1024, hop_length=math.floor(cfg.sample_rate / cfg.fps),
            n_mels=80, f_min=50, f_max=7600, power=2.0, center=False
        )
        # self.amplog = torchaudio.transforms.AmplitudeToDB()
        # clamp power->dB to avoid -inf for silent bins
        self.amplog = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)  # NOTE: fix NAN

        # NEW: effective mel FPS (how many mel frames per second)
        self.mel_hop = int(self.mel.hop_length)
        self.mel_fps = self.cfg.sample_rate / self.mel_hop  # float

        # tiny LRU caches (per DataLoader worker)
        self._mel_cache = OrderedDict()  # key: stem -> Tensor [Tm,80]
        self._pose_cache = OrderedDict()  # key: stem -> (xy [Tf,2,21,2], cf [Tf,2,21])
        self._mel_cap, self._pose_cap = 32, 16

    def __len__(self) -> int:
        return len(self.items)

    def _lru_put(self, cache, key, val, cap):
        cache[key] = val
        if len(cache) > cap:
            cache.pop(next(iter(cache)))

    def _stem(self, pdir: str):
        return pathlib.Path(pdir).name

    def _mel_for(self, wav_path: str, pdir: str):
        stem = self._stem(pdir)

        # 1) in-memory cache (works for both precomputed and computed)
        m = self._mel_cache.get(stem)
        if m is not None:
            return m

        # 2) precomputed on disk
        if self.mel_root:
            p = self.mel_root / f"{stem}.pt"
            if p.exists():
                # NOTE: NAN fix
                # obj = torch.load(p, weights_only=True, map_location="cpu")
                # m = obj["mel"]  # [Tm,80]
                # self._lru_put(self._mel_cache, stem, m, self._mel_cap)
                # return m
                obj = torch.load(p, weights_only=True, map_location="cpu")
                m = obj["mel"]  # [Tm,80]
                m = torch.nan_to_num(m, nan=-80.0, posinf=0.0, neginf=-80.0)  # sanitize
                return m

        # 3) compute once and cache
        wav, sr = torchaudio.load(wav_path)
        if sr != self.cfg.sample_rate:
            # TODO: add a warning here
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        wav = wav.mean(0, keepdim=True)
        m = self.amplog(self.mel(wav)).squeeze(0).transpose(0, 1).contiguous()
        m = torch.nan_to_num(m, nan=-80.0, posinf=0.0, neginf=-80.0)  # sanitize  # NOTE: NAN fix
        self._lru_put(self._mel_cache, stem, m, self._mel_cap)
        return m

    def _pose_for(self, pdir: str):
        stem = self._stem(pdir)

        # 1) in-memory cache
        pc = self._pose_cache.get(stem)
        if pc is not None:
            return pc

        # 2) precomputed on disk (use memmap to avoid big copies)
        if self.pose_root:
            p = self.pose_root / f"{stem}.npz"
            if p.exists():
                z = np.load(p, mmap_mode="r")
                xy, cf = z["xy"], z["cf"]  # memmapped arrays
                pc = (xy, cf)
                self._lru_put(self._pose_cache, stem, pc, self._pose_cap)
                return pc

        # 3) build once from per-frame .npy and cache
        xy, cf = [], []
        for pf in load_pose_dicts(pdir):
            d = np.load(pf, allow_pickle=True).item()
            h, hs = hands_from_pose_dict(d)
            xy.append(h)
            cf.append(hs)
        pc = (np.stack(xy, 0), np.stack(cf, 0))
        self._lru_put(self._pose_cache, stem, pc, self._pose_cap)
        return pc

    def _pose_idx_to_mel_idx(self, t0: int, Tm: int) -> int:
        # Pose frame index -> time (sec)
        t_sec = t0 / self.cfg.fps
        # Time -> mel frame index (round to nearest)
        start = int(round(t_sec * self.mel_fps))
        # Clamp into valid windowable range
        start = max(0, min(start, max(0, Tm - self.cfg.win_T)))
        return start

    def __getitem__(self, i):
        audio, pdir, t0 = self.items[i]

        # audio window
        mel_full = self._mel_for(audio, pdir)  # <— stem-aware
        Tm = mel_full.shape[0]
        start = self._pose_idx_to_mel_idx(t0, Tm)
        feats = mel_full[start:start + self.cfg.win_T]
        if feats.shape[0] < self.cfg.win_T:
            raise RuntimeError("should not happen")
            feats = F.pad(feats, (0, 0, 0, self.cfg.win_T - feats.shape[0]))
        T = feats.shape[0]

        # pose window
        xy_full, cf_full = self._pose_for(pdir)
        hands_xy = xy_full[t0:t0 + T]
        hands_cf = cf_full[t0:t0 + T]
        if hands_xy.shape[0] < T:
            padT = T - hands_xy.shape[0]
            hands_xy = np.concatenate([hands_xy, np.zeros((padT, 2, 21, 2), np.float32)], 0)
            hands_cf = np.concatenate([hands_cf, np.zeros((padT, 2, 21), np.float32)], 0)

        if self.cfg.mode == "heatmap":
            H, W = self.cfg.heat_H, self.cfg.heat_W
            target = np.zeros((T, H, W), np.float32)
            for t in range(T):
                pts = hands_xy[t].reshape(-1, 2)
                cf = hands_cf[t].reshape(-1)
                target[t] = gaussian_splats(pts, cf, H, W, sigma=3.0)
            return dict(audio=feats, target=torch.from_numpy(target).unsqueeze(1))
        else:
            return dict(
                audio=feats,
                target=torch.from_numpy(hands_xy).float(),
                conf=torch.from_numpy(hands_cf).float()
            )


# ----- CSV → pairs -----
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
    train_pairs = make_pairs(train_list)
    val_pairs = make_pairs(val_list)

    # (optional) drop rows that don’t have precomputed files
    # if filter_missing and (mel_root or pose_root):
    #     def keep(pairs):
    #         kept, dropped = [], 0
    #         mr = Path(mel_root) if mel_root else None
    #         pr = Path(pose_root) if pose_root else None
    #         for a, pdir in pairs:
    #             stem = Path(pdir).name
    #             m_ok = (mr / f"{stem}.pt").exists() if mr else True
    #             z_ok = (pr / f"{stem}.npz").exists() if pr else True
    #             if m_ok and z_ok:
    #                 kept.append((a, pdir))
    #             else:
    #                 dropped += 1
    #         if dropped:
    #             print(f"[make_loaders] dropped {dropped} pair(s) without precomputed features")
    #         return kept
    #
    #     train_pairs = keep(train_pairs)
    #     val_pairs = keep(val_pairs)

    train_ds = A2PDataset(train_pairs, cfg, mel_root=mel_root, pose_root=pose_root)
    val_ds = A2PDataset(val_pairs, cfg, mel_root=mel_root, pose_root=pose_root)

    use_workers = max(0, int(num_workers))
    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=use_workers,
        pin_memory=True, collate_fn=a2p_collate, persistent_workers=(use_workers > 0),
        prefetch_factor=2 if use_workers > 0 else None
    )
    val_dl = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=use_workers,
        pin_memory=True, collate_fn=a2p_collate, persistent_workers=(use_workers > 0),
        prefetch_factor=2 if use_workers > 0 else None
    )
    return train_dl, val_dl


if __name__ == '__main__':
    """
    python experiments/a2p/data.py --list EMTD_dataset/lists/train.csv --mode heatmap --bs 16 --num_workers 4
    python experiments/a2p/data.py --list EMTD_dataset/lists/train.csv --mode keypoints --bs 32 --num_workers 4
    """
    import argparse, time, torch, numpy as _np

    ap = argparse.ArgumentParser("Smoke test for A2PDataset/DataLoader")
    ap.add_argument("--mel_root", type=str, default="../../EMTD_dataset/features/mels")
    ap.add_argument("--pose_root", type=str, default="../../EMTD_dataset/features/poses")
    ap.add_argument("--no_filter_missing", action="store_true")
    ap.add_argument("--train_list", default="../../EMTD_dataset/lists/train.csv",
                    help="CSV with 'audio.wav,pose_dir' per line")
    ap.add_argument("--val_list", default="../../EMTD_dataset/lists/val.csv",
                    help="CSV with 'audio.wav,pose_dir' per line")
    ap.add_argument("--mode", choices=["heatmap", "keypoints"], default="heatmap")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3, help="number of batches to time")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--win_T", type=int, default=12)
    ap.add_argument("--hop_T", type=int, default=3)
    ap.add_argument("--heat_hw", type=int, nargs=2, default=[256, 256])
    args = ap.parse_args()

    cfg = A2PConfig(
        fps=args.fps, win_T=args.win_T, hop_T=args.hop_T,
        heat_H=args.heat_hw[0], heat_W=args.heat_hw[1], mode=args.mode
    )

    # use the same list for both to keep the helper signature
    train_dl, _ = make_loaders(
        args.list, args.val_list, cfg,
        bs=args.bs, num_workers=args.num_workers,
        mel_root=args.mel_root, pose_root=args.pose_root,
        filter_missing=(not args.no_filter_missing),
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} | mode={args.mode} | bs={args.bs} | workers={args.num_workers} "
          f"| win_T={cfg.win_T} hop_T={cfg.hop_T} | heat_hw=({cfg.heat_H},{cfg.heat_W})")
    print(f"dataset size (windows): {len(train_dl.dataset)}")

    it = iter(train_dl)
    times = []
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


        def shp(x):
            return tuple(x.shape) if hasattr(x, "shape") else type(x)


        shapes = {k: shp(v) for k, v in batch.items()}
        print(f"batch {i}: {shapes}")

    print(f"avg dataloader time per batch: {(_np.mean(times) * 1000):.1f} ms")
