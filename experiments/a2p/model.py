# experiments/a2p/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class TemporalAudioEncoder(nn.Module):
    # in:  [B, T, 80]  (80-bin mel per frame)
    # out: [B, T, d]   (d = 256 by default)
    def __init__(self, d_in=80, d_mid=256, n_layers=2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [nn.Conv1d(d, d_mid, 5, padding=2), nn.GELU(),
                       nn.Conv1d(d_mid, d_mid, 5, padding=2), nn.GELU()]
            d = d_mid
        self.net = nn.Sequential(*layers)
        self.out_dim = d_mid

    def forward(self, x):  # x: [B,T,D]
        y = self.net(x.transpose(1, 2))  # [B,d,T]
        return y.transpose(1, 2)  # [B,T,d]


class A2PHeatmapHead(nn.Module):
    # in:  [B, T, d]
    # out: [B, T, 1, H, W]  (H=W=256 expected)
    def __init__(self, d_in, H=256, W=256):
        super().__init__()
        self.proj = nn.Linear(d_in, 64 * 64)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid(),
        )
        self.H, self.W = H, W

    def forward(self, x):
        B, T, D = x.shape
        y = self.proj(x).view(B * T, 1, 64, 64)
        y = self.up(y).view(B, T, 1, self.H, self.W)
        return y


class A2PKeypointHead(nn.Module):
    # in:  [B, T, d]
    # out: [B, T, 2, 21, 2]   (2 hands × 21 joints × (x,y))
    def __init__(self, d_in, n_kp=42):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, n_kp * 2),
        )
        self.n_kp = n_kp

    def forward(self, x):
        B, T, D = x.shape
        y = self.net(x).view(B, T, 2, 21, 2)
        return torch.sigmoid(y)


class Audio2Pose(pl.LightningModule):
    def __init__(self, mode="heatmap", fps=24, heat_hw=(256, 256), lr=1e-4, wd=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.enc = TemporalAudioEncoder(d_in=80, d_mid=256, n_layers=2)
        self.mode = mode
        self.lr, self.wd = lr, wd
        self.head = A2PHeatmapHead(self.enc.out_dim, *heat_hw) if mode == "heatmap" \
            else A2PKeypointHead(self.enc.out_dim, n_kp=42)

    def forward(self, audio_feats):
        return self.head(self.enc(audio_feats))

    def training_step(self, batch, _):
        audio = batch["audio"].float().to(self.device, non_blocking=True)
        pred = self(audio)
        # NOTE: NAN fix
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction; check mel features / dB clamp.")
        if self.mode == "heatmap":
            gt = batch["target"].float().to(self.device)
            l1 = F.l1_loss(pred, gt)
            # NOTE: NAN fix
            # tv = F.l1_loss(pred[:,1:]-pred[:,:-1], gt[:,1:]-gt[:,:-1])
            if pred.size(1) > 1:
                tv = F.l1_loss(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])
            else:
                tv = torch.zeros((), device=pred.device)

            loss = l1 + 0.2 * tv
            self.log_dict({"train/l1": l1, "train/tv": tv, "train/loss": loss}, prog_bar=True)
        else:
            gt = batch["target"].float().to(self.device)
            cf = batch["conf"].float().to(self.device)
            w = cf.unsqueeze(-1).clamp(min=0.05)
            l1 = (w * (pred - gt).abs()).mean()
            vloss = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            loss = l1 + vloss
            self.log_dict({"train/l1": l1, "train/v": vloss, "train/loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        audio = batch["audio"].float().to(self.device)
        pred = self(audio)
        if self.mode == "heatmap":
            gt = batch["target"].float().to(self.device)
            self.log("val/l1", F.l1_loss(pred, gt), prog_bar=True)
        else:
            gt = batch["target"].float().to(self.device)
            cf = batch["conf"].float().to(self.device)
            w = cf.unsqueeze(-1).clamp(min=0.05)

            l1 = (w * (pred - gt).abs()).mean()
            vloss = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            val_loss = l1 + vloss
            self.log("val/loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100_000)
        return [opt], [sch]


def test_temporal_audio_encoder():
    tae = TemporalAudioEncoder()


if __name__ == '__main__':
    test_temporal_audio_encoder()

    """
    python experiments/a2p/model.py --mode heatmap --B 4 --T 12 --H 256 --W 256 --steps 3
    python experiments/a2p/model.py --mode keypoints --B 8 --T 12 --steps 3
    """
    import argparse, time, torch

    parser = argparse.ArgumentParser("Smoke test for Audio2Pose")
    parser.add_argument("--mode", choices=["heatmap", "keypoints"], default="heatmap")
    parser.add_argument("--B", type=int, default=4, help="batch size")
    parser.add_argument("--T", type=int, default=12, help="frames per clip")
    parser.add_argument("--H", type=int, default=256, help="heatmap height (heatmap mode)")
    parser.add_argument("--W", type=int, default=256, help="heatmap width  (heatmap mode)")
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps to run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Audio2Pose smoke test on {device} (mode={args.mode})")

    model = Audio2Pose(mode=args.mode, heat_hw=(args.H, args.W)).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Fake batch
    audio = torch.randn(args.B, args.T, 80, device=device)
    if args.mode == "heatmap":
        target = torch.rand(args.B, args.T, 1, args.H, args.W, device=device)
    else:
        target = torch.rand(args.B, args.T, 2, 21, 2, device=device)
        conf = torch.rand(args.B, args.T, 2, 21, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        pred = model(audio)

        if args.mode == "heatmap":
            l1 = torch.nn.functional.l1_loss(pred, target)
            tv = torch.nn.functional.l1_loss(pred[:, 1:] - pred[:, :-1],
                                             target[:, 1:] - target[:, :-1])
            loss = l1 + 0.2 * tv
        else:
            w = conf.unsqueeze(-1).clamp(min=0.05)
            loss = (w * (pred - target).abs()).mean()
            vloss = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            loss = loss + 0.1 * vloss

        loss.backward()
        opt.step()

        if device == "cuda":
            torch.cuda.synchronize()
        print(f"step {step}: pred shape={tuple(pred.shape)}, loss={loss.item():.4f}")

    dt = (time.time() - t0) / max(1, args.steps)
    print(f"Avg time/step: {dt * 1000:.1f} ms")
