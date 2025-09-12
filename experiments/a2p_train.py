# a2p_train.py
# Minimal Audio→Pose trainer (Lightning, 1 GPU). Supports "heatmap" or "keypoints".
"""
Heatmap target (fastest to plug into EM2 PoseEncoder):
python experiments/a2p_train.py \
  --train_list /path/to/train.tsv \
  --val_list   /path/to/val.tsv \
  --mode heatmap --fps 24 --win_T 12 --hop_T 3 \
  --heat_hw 768 768 --bs 16 --epochs 20

python experiments/a2p_train.py \
  --train_list /path/to/train.tsv \
  --val_list   /path/to/val.tsv \
  --mode keypoints --fps 24 --win_T 12 --hop_T 3 \
  --bs 32 --epochs 20
"""
import os
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from torch.utils.data._utils.collate import default_collate

torch.set_float32_matmul_precision('high')  # better Tensor Core usage
torch.backends.cudnn.benchmark = True

def a2p_collate(batch):
    # top-level (picklable): stacks dict of tensors cleanly
    return {k: default_collate([b[k] for b in batch]) for k in batch[0]}


# ----------------------------
# Utils
# ----------------------------

def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_pose_dicts(pose_dir: str) -> List[str]:
    files = [f for f in os.listdir(pose_dir) if f.endswith(".npy")]
    files.sort(key=natural_key)
    return [os.path.join(pose_dir, f) for f in files]


def hands_from_pose_dict(p: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Always return (hands[2,21,2], hands_score[2,21]) in float32.
    Coerces shapes, drops extra channels, trims extra hands, pads missing ones,
    and removes NaNs.
    """
    H = np.zeros((2, 21, 2), dtype=np.float32)
    S = np.zeros((2, 21), dtype=np.float32)

    h = p.get("hands", None)
    hs = p.get("hands_score", None)

    if h is None or hs is None:
        return H, S

    h = np.asarray(h)
    hs = np.asarray(hs)

    # Keep x,y only if extra channels exist (e.g., x,y,score)
    if h.ndim == 3 and h.shape[-1] >= 2:
        h = h[..., :2]

    # Normalize to (2,21,2)
    if h.shape == (42, 2):
        h = h.reshape(2, 21, 2)
    elif h.shape == (21, 2):
        h = np.stack([h, np.zeros_like(h)], axis=0)
    elif h.ndim == 3 and h.shape[1] == 21 and h.shape[2] == 2:
        if h.shape[0] > 2:
            h = h[:2]  # slice 4→2
        elif h.shape[0] < 2:
            pad = np.zeros((2 - h.shape[0], 21, 2), dtype=np.float32)
            h = np.concatenate([h, pad], axis=0)
    else:
        h = H  # fallback

    # Scores → (2,21)
    if hs.ndim == 1 and hs.size == 42:
        hs = hs.reshape(2, 21)
    elif hs.ndim == 2 and hs.shape[1] == 21:
        if hs.shape[0] > 2:
            hs = hs[:2]
        elif hs.shape[0] < 2:
            pad = np.zeros((2 - hs.shape[0], 21), dtype=np.float32)
            hs = np.concatenate([hs, pad], axis=0)
    elif hs.shape != (2, 21):
        hs = S

    h = np.nan_to_num(h, nan=0.0).astype(np.float32)
    hs = np.nan_to_num(hs, nan=0.0).astype(np.float32)
    return h, hs


def gaussian_splats(points_xy01: np.ndarray, conf: np.ndarray, H: int, W: int, sigma: float = 3.0) -> np.ndarray:
    """points: [P,2] in [0,1]; conf: [P]; returns heatmap [H,W] in [0,1]."""
    yy, xx = np.mgrid[0:H, 0:W]
    heat = np.zeros((H, W), dtype=np.float32)
    for (x01, y01), c in zip(points_xy01, conf):
        if c <= 0:
            continue
        x, y = x01 * (W - 1), y01 * (H - 1)
        g = np.exp(-((xx - x) * (xx - x) + (yy - y) * (yy - y)) / (2 * sigma * sigma))
        heat = np.maximum(heat, (c * g).astype(np.float32))
    heat = np.clip(heat, 0, 1)
    return heat


# ----------------------------
# Dataset
# ----------------------------

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
    """
    Expects a list of (audio_path, pose_dir).
    pose_dir contains {0.npy,1.npy,...} produced by your DWPose pipeline.
    """

    def __init__(self, pairs: List[Tuple[str, str]], cfg: A2PConfig):
        self.pairs = pairs
        self.cfg = cfg
        self.items = []  # (audio_path, pose_dir, start_frame)
        self._mel_cache = {}  # {path: torch.Tensor [Tm,80]}
        self._pose_cache = {}  # {pdir: (np.ndarray [Tf,2,21,2], np.ndarray [Tf,2,21])}
        for audio, pdir in pairs:
            frames = load_pose_dicts(pdir)
            T = len(frames)
            for t0 in range(0, max(0, T - cfg.win_T + 1), cfg.hop_T):
                self.items.append((audio, pdir, t0))

        # audio features: mels with hop ≈ 1 frame
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=1024, hop_length=round(cfg.sample_rate / cfg.fps),
            n_mels=80, f_min=50, f_max=7600, power=2.0, center=False
        )
        self.amplog = torchaudio.transforms.AmplitudeToDB()

    def _mel_for(self, wav_path: str) -> torch.Tensor:
        m = self._mel_cache.get(wav_path)
        if m is None:
            wav, sr = torchaudio.load(wav_path)
            if sr != self.cfg.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
            wav = wav.mean(0, keepdim=True)
            m = self.amplog(self.mel(wav)).squeeze(0).transpose(0, 1).contiguous()  # [Tm,80]
            self._mel_cache[wav_path] = m
            # naive LRU: keep cache small
            if len(self._mel_cache) > 24:
                self._mel_cache.pop(next(iter(self._mel_cache)))
        return m

    def _pose_for(self, pdir: str):
        pc = self._pose_cache.get(pdir)
        if pc is None:
            files = load_pose_dicts(pdir)
            xy, cf = [], []
            for pf in files:
                d = np.load(pf, allow_pickle=True).item()
                h, hs = hands_from_pose_dict(d)  # already coerced to (2,21,2),(2,21)
                xy.append(h);
                cf.append(hs)
            pc = (np.stack(xy, 0), np.stack(cf, 0))  # [Tf,2,21,2], [Tf,2,21]
            self._pose_cache[pdir] = pc
            if len(self._pose_cache) > 16:
                self._pose_cache.pop(next(iter(self._pose_cache)))
        return pc

    def __len__(self):
        return len(self.items)

    def _audio_feats(self, wav_path: str, t0: int) -> torch.Tensor:
        # Load mono, 16k
        wav, sr = torchaudio.load(wav_path)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        wav = wav.mean(0, keepdim=True)  # [1,N]
        mel = self.amplog(self.mel(wav))  # [1,80, Tm]
        mel = mel.squeeze(0).transpose(0, 1)  # [Tm,80]
        # Align a window of length win_T centered at frame t0 (approx.)
        # Pose frame index ↔ mel index with ratio ≈ 1
        Tm = mel.shape[0]
        start = min(max(0, t0), max(0, Tm - self.cfg.win_T))
        mel = mel[start:start + self.cfg.win_T]  # [T,80]
        if mel.shape[0] < self.cfg.win_T:
            pad = self.cfg.win_T - mel.shape[0]
            mel = F.pad(mel, (0, 0, 0, pad))  # pad time
        return mel  # [T,80]

    def __getitem__(self, i):
        audio, pdir, t0 = self.items[i]

        # mel window
        mel_full = self._mel_for(audio)  # [Tm,80]
        start = min(max(0, t0), max(0, mel_full.shape[0] - self.cfg.win_T))
        feats = mel_full[start:start + self.cfg.win_T]
        if feats.shape[0] < self.cfg.win_T:
            feats = F.pad(feats, (0, 0, 0, self.cfg.win_T - feats.shape[0]))  # [T,80]
        T = feats.shape[0]

        # pose window
        xy_full, cf_full = self._pose_for(pdir)  # [Tf,2,21,2], [Tf,2,21]
        hands_xy = xy_full[t0:t0 + T]
        hands_cf = cf_full[t0:t0 + T]
        if hands_xy.shape[0] < T:  # pad if the last window overruns
            padT = T - hands_xy.shape[0]
            hands_xy = np.concatenate([hands_xy, np.zeros((padT, 2, 21, 2), np.float32)], 0)
            hands_cf = np.concatenate([hands_cf, np.zeros((padT, 2, 21), np.float32)], 0)

        # Truncate if longer (shouldn't happen here, but safe)
        hands_xy = np.stack(hands_xy[:T], axis=0)  # [T,2,21,2]
        hands_cf = np.stack(hands_cf[:T], axis=0)  # [T,2,21]

        if self.cfg.mode == "heatmap":
            # 1-ch heatmap per frame (hands only, but you can add body/face)
            H, W = self.cfg.heat_H, self.cfg.heat_W
            target = np.zeros((T, H, W), dtype=np.float32)
            for t in range(T):
                pts = hands_xy[t].reshape(-1, 2)  # [42,2] in 0..1
                cf = hands_cf[t].reshape(-1)  # [42]
                target[t] = gaussian_splats(pts, cf, H, W, sigma=3.0)
            target = torch.from_numpy(target)  # [T,H,W]
            out = dict(audio=feats, target=target.unsqueeze(1))  # [T,1,H,W]
        else:
            # keypoint regression (x,y) in 0..1 + confidence (for loss weights)
            target = torch.from_numpy(hands_xy).float()  # [T,2,21,2]
            conf = torch.from_numpy(hands_cf).float()  # [T,2,21]
            out = dict(audio=feats, target=target, conf=conf)

        return out


# ----------------------------
# Model
# ----------------------------

class TemporalAudioEncoder(nn.Module):
    """Tiny temporal encoder over per-frame mels. Fast and effective."""

    def __init__(self, d_in=80, d_mid=256, n_layers=3):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers):
            layers += [nn.Conv1d(d, d_mid, 5, padding=2), nn.GELU(),
                       nn.Conv1d(d_mid, d_mid, 5, padding=2), nn.GELU()]
            d = d_mid
        self.net = nn.Sequential(*layers)
        self.out_dim = d_mid

    def forward(self, x: torch.Tensor):  # x: [B,T,D]
        y = x.transpose(1, 2)  # [B,D,T]
        y = self.net(y)  # [B,d_mid,T]
        return y.transpose(1, 2)  # [B,T,d_mid]


class A2PHeatmapHead(nn.Module):
    def __init__(self, d_in, H=256, W=256):
        super().__init__()
        self.proj = nn.Linear(d_in, 64 * 64)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 4, stride=2, padding=1), nn.GELU(),  # 64→128
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GELU(),  # 128→256
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
        self.H, self.W = H, W

    def forward(self, x):  # [B,T,d]
        B, T, D = x.shape
        y = self.proj(x).view(B * T, 1, 64, 64)  # [B*T,1,64,64]
        y = self.up(y)  # [B*T,1,H,W]
        y = y.view(B, T, 1, self.H, self.W)  # [B,T,1,H,W]
        return y


class A2PKeypointHead(nn.Module):
    def __init__(self, d_in, n_kp=42):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, n_kp * 2)
        )
        self.n_kp = n_kp

    def forward(self, x):  # [B,T,d]
        B, T, D = x.shape
        y = self.net(x)  # [B,T,84]
        y = y.view(B, T, 2, 21, 2)  # [B,T,2,21,2] in 0..1 after sigmoid
        return torch.sigmoid(y)


class Audio2Pose(pl.LightningModule):
    def __init__(self, mode="heatmap", fps=24, heat_hw=(256, 256), lr=1e-4, wd=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.enc = TemporalAudioEncoder(d_in=80, d_mid=256, n_layers=2)
        if mode == "heatmap":
            self.head = A2PHeatmapHead(self.enc.out_dim, H=heat_hw[0], W=heat_hw[1])
        else:
            self.head = A2PKeypointHead(self.enc.out_dim, n_kp=42)
        self.mode = mode
        self.lr = lr;
        self.wd = wd

    def forward(self, audio_feats):  # [B,T,80]
        x = self.enc(audio_feats)  # [B,T,256]
        return self.head(x)

    def training_step(self, batch, _):
        audio = batch["audio"].float().to(self.device)  # [B,T,80]
        pred = self(audio)
        if self.mode == "heatmap":
            gt = batch["target"].float().to(self.device)  # [B,T,1,H,W]
            # heat loss + temporal smoothness
            l1 = F.l1_loss(pred, gt)
            tv = F.l1_loss(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])
            loss = l1 + 0.2 * tv
            self.log_dict({"train/l1": l1, "train/tv": tv, "train/loss": loss}, prog_bar=True)
        else:
            gt = batch["target"].float().to(self.device)  # [B,T,2,21,2] (0..1)
            cf = batch["conf"].float().to(self.device)  # [B,T,2,21]
            w = cf.unsqueeze(-1).clamp(min=0.05)  # weight low-conf a little
            loss = (w * (pred - gt).abs()).mean()
            vloss = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            loss = loss + 0.1 * vloss
            self.log_dict({"train/l1": loss, "train/v": vloss, "train/loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        audio = batch["audio"].float().to(self.device)
        pred = self(audio)
        if self.mode == "heatmap":
            gt = batch["target"].float().to(self.device)
            l1 = F.l1_loss(pred, gt)
            self.log("val/l1", l1, prog_bar=True)
        else:
            gt = batch["target"].float().to(self.device)
            cf = batch["conf"].float().to(self.device)
            w = cf.unsqueeze(-1).clamp(min=0.05)
            l1 = (w * (pred - gt).abs()).mean()
            self.log("val/l1", l1, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100_000)
        return [opt], [sch]


# ----------------------------
# Data helpers and trainer
# ----------------------------

def make_pairs(list_file: str) -> List[Tuple[str, str]]:
    """
    list_file format (tsv):
    /path/to/audio.wav    /path/to/pose_dir
    """
    pairs = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            a, p = line.split(",")
            pairs.append((a, p))
    return pairs


def make_loaders(train_list: str, val_list: str, cfg: A2PConfig, bs=16, num_workers=4):
    train_ds = A2PDataset(make_pairs(train_list), cfg)
    val_ds = A2PDataset(make_pairs(val_list), cfg)

    use_workers = max(0, int(num_workers))
    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=use_workers, pin_memory=True,
        collate_fn=a2p_collate,
        persistent_workers=(use_workers > 0),
    )
    val_dl = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=use_workers, pin_memory=True,
        collate_fn=a2p_collate,
        persistent_workers=(use_workers > 0),
    )
    return train_dl, val_dl


# ----------------------------
# Entry
# ----------------------------

if __name__ == "__main__":
    import argparse

    pl.seed_everything(1234, workers=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", required=True)  # tsv: audio \t pose_dir
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--mode", choices=["heatmap", "keypoints"], default="heatmap")
    ap.add_argument("--win_T", type=int, default=12)
    ap.add_argument("--hop_T", type=int, default=3)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--heat_hw", type=int, nargs=2, default=[768, 768])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="a2p_ckpts")
    args = ap.parse_args()

    cfg = A2PConfig(
        fps=args.fps, win_T=args.win_T, hop_T=args.hop_T,
        heat_H=args.heat_hw[0], heat_W=args.heat_hw[1], mode=args.mode
    )
    train_dl, val_dl = make_loaders(args.train_list, args.val_list, cfg, bs=args.bs, num_workers=args.num_workers)

    model = Audio2Pose(mode=args.mode, fps=args.fps, heat_hw=tuple(args.heat_hw), lr=args.lr)

    logger = pl.loggers.TensorBoardLogger(args.out, name=f"a2p_{args.mode}")
    ckpt = pl.callbacks.ModelCheckpoint(dirpath=logger.log_dir, save_top_k=2, monitor="val/l1", mode="min")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accum,
        devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=20
    )
    trainer.fit(model, train_dl, val_dl)
