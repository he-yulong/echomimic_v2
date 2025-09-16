# echomimic_v2/experiments/a2p/model_simple.py
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


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

PALM_ANCHORS = [5, 9, 13, 17]  # wrist=0 to these anchors defines a scale
EPS = 1e-6


def _to_motion_delta(seq_xy: torch.Tensor) -> torch.Tensor:
    """
    seq_xy: [B, T, 2, 21, 2] in [0,1]
    returns Δ along time: [B, T-1, 2, 21, 2]
    """
    if seq_xy.size(1) <= 1:
        return seq_xy[:, :0]  # empty along time
    return seq_xy[:, 1:] - seq_xy[:, :-1]


def _flatten_tp22(x: torch.Tensor) -> torch.Tensor:
    """
    [B, T', 2, 21, 2] -> [B, T', 84]
    """
    return x.reshape(x.size(0), x.size(1), -1)


class Discriminator1D(nn.Module):
    """
    Temporal Conv discriminator over sequences of flattened joints.
    Input: [B, T', C] where C=84 (pose or motion-delta).
    Output: [B, 1] (LSGAN score)
    """

    def __init__(self, c_in=84, d=128, n_blocks=3):
        super().__init__()
        layers = []
        ch = c_in
        for _ in range(n_blocks):
            layers += [
                nn.Conv1d(ch, d, 3, padding=1), nn.GELU(),
                nn.Conv1d(d, d, 3, padding=1), nn.GELU()
            ]
            ch = d
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        # x: [B, T', C] -> [B, C, T']
        x = x_btc.transpose(1, 2)
        h = self.backbone(x)  # [B, d, T']
        h = h.mean(dim=2)  # [B, d]
        logit = self.head(h)  # [B,1]
        return logit


class Audio2Pose(pl.LightningModule):
    # --- inner helper modules & funcs (self-contained) ---
    class Discriminator1D(nn.Module):
        """
        Temporal Conv discriminator over sequences of flattened joints.
        Input:  [B, T', C]  (C=84 for 2*21*(x,y))
        Output: [B, 1]      (LSGAN score)
        """
        def __init__(self, c_in=84, d=128, n_blocks=3):
            super().__init__()
            layers, ch = [], c_in
            for _ in range(n_blocks):
                layers += [
                    nn.Conv1d(ch, d, 3, padding=1), nn.GELU(),
                    nn.Conv1d(d, d, 3, padding=1), nn.GELU()
                ]
                ch = d
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(d, 1)

        def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
            x = x_btc.transpose(1, 2)     # [B, C, T']
            h = self.backbone(x)          # [B, d, T']
            h = h.mean(dim=2)             # [B, d] (GAP over time)
            return self.head(h)           # [B, 1]

    @staticmethod
    def _to_motion_delta(seq_xy: torch.Tensor) -> torch.Tensor:
        """
        seq_xy: [B, T, 2, 21, 2] in [0,1]
        returns Δ along time: [B, T-1, 2, 21, 2]
        """
        if seq_xy.size(1) <= 1:
            return seq_xy[:, :0]
        return seq_xy[:, 1:] - seq_xy[:, :-1]

    @staticmethod
    def _flatten_tp22(x: torch.Tensor) -> torch.Tensor:
        """
        [B, T', 2, 21, 2] -> [B, T', 84]
        """
        return x.reshape(x.size(0), x.size(1), -1)

    # ----------------------- main module -----------------------
    def __init__(
        self,
        mode="heatmap",
        fps=24,
        heat_hw=(256, 256),
        lr=1e-4,
        wd=0.01,
        # --- GAN options ---
        use_gan: bool = False,
        d_input: str = "motion",   # "motion" | "pose" | "both"
        lambda_gan: float = 1.0,
        lambda_d: float = 1.0,
        lr_d: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # base encoder+head (keep your original)
        self.enc = TemporalAudioEncoder(d_in=80, d_mid=256, n_layers=2)
        self.head = A2PKeypointHead(self.enc.out_dim, n_kp=42)
        self.mode = mode
        self.lr, self.wd = lr, wd

        # --- GAN setup ---
        self.use_gan = bool(use_gan and (mode == "keypoints"))
        self.d_input = d_input
        self.lambda_gan = float(lambda_gan)
        self.lambda_d = float(lambda_d)
        self.lr_d = float(lr_d)
        if self.use_gan:
            self.disc = self.Discriminator1D(c_in=84)

        # manual optimization for (G,D)
        self.automatic_optimization = False

    # --------- forward & supervised losses (unchanged semantics) ---------
    def forward(self, audio_feats):
        pred = self.head(self.enc(audio_feats))  # [B,T,2,21,2]
        return pred.clamp(0, 1)                  # stay in image range (no in-place)

    def _cal_and_log_loss(self, batch, pred, label: str):
        # NOTE: NAN guard
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction; check mel features / dB clamp.")

        gt = batch["target"].float().to(self.device)  # [B,T,2,21,2]
        cf = batch["conf"].float().to(self.device)    # [B,T,2,21]

        # 1) Position loss (confidence-weighted SmoothL1)
        pos_err = F.smooth_l1_loss(pred, gt, beta=0.02, reduction="none")  # [B,T,2,21,2]
        w_joint = (0.05 + cf).unsqueeze(-1).detach()                        # [B,T,2,21,1]
        l_pos = (w_joint * pos_err).mean()

        # 2) Bone length loss (normalized by palm size)
        EDGES = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
        ]
        PALM_ANCHORS = [5, 9, 13, 17]
        EPS = 1e-6

        def bone_lengths(kp):
            a = torch.stack([kp[..., i, :] for (i, j) in EDGES], dim=-2)  # [B,T,2,E,2]
            b = torch.stack([kp[..., j, :] for (i, j) in EDGES], dim=-2)
            return ((a - b).pow(2).sum(-1) + EPS).sqrt()                   # [B,T,2,E]

        def palm_scale(kp):
            wrist = kp[..., 0:1, :]                                        # [B,T,2,1,2]
            anchors = torch.stack([kp[..., i, :] for i in PALM_ANCHORS], dim=-2)  # [B,T,2,4,2]
            d = ((anchors - wrist).pow(2).sum(-1) + EPS).sqrt()            # [B,T,2,4]
            return d.mean(-1, keepdim=True) + EPS                           # [B,T,2,1]

        Lp = bone_lengths(pred)
        Lg = bone_lengths(gt).detach()
        Sg = palm_scale(gt).detach()
        Lp_n, Lg_n = Lp / Sg, Lg / Sg

        edge_w = []
        for (i, j) in EDGES:
            edge_w.append(torch.minimum(cf[..., i], cf[..., j]))           # [B,T,2]
        edge_w = torch.stack(edge_w, dim=-1)                                # [B,T,2,E]
        l_bone = (0.05 + edge_w).detach() * F.smooth_l1_loss(Lp_n, Lg_n, beta=0.02, reduction="none")
        l_bone = l_bone.mean()

        # 3) Velocity loss
        if pred.size(1) > 1:
            dv_pred = pred[:, 1:] - pred[:, :-1]                           # [B,T-1,2,21,2]
            dv_gt   = (gt[:, 1:] - gt[:, :-1]).detach()
            cf_mid  = 0.5 * (cf[:, 1:] + cf[:, :-1])                       # [B,T-1,2,21]
            l_vel = (0.05 + cf_mid.unsqueeze(-1)).detach() * F.smooth_l1_loss(
                dv_pred, dv_gt, beta=0.02, reduction="none"
            )
            l_vel = l_vel.mean()
        else:
            l_vel = torch.zeros((), device=self.device)

        # weights
        weight_pos, weight_bone, weight_vel = 5.0, 1.0, 0.1
        loss = weight_pos * l_pos + weight_bone * l_bone + weight_vel * l_vel

        self.log_dict(
            {f"{label}/l_pos": l_pos, f"{label}/l_bone": l_bone, f"{label}/l_vel": l_vel, f"{label}/loss": loss},
            prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    # ------------------------- GAN helpers -------------------------
    def _disc_inputs(self, pred_xy, gt_xy):
        """
        Build inputs to D per self.d_input.
        Returns (x_fake, x_real) each as [B, T', C].
        """
        f_pose = self._flatten_tp22(pred_xy)     # [B,T,84]
        r_pose = self._flatten_tp22(gt_xy)       # [B,T,84]
        f_mot  = self._flatten_tp22(self._to_motion_delta(pred_xy))  # [B,T-1,84]
        r_mot  = self._flatten_tp22(self._to_motion_delta(gt_xy))    # [B,T-1,84]

        if self.d_input == "motion":
            return f_mot, r_mot
        elif self.d_input == "pose":
            return f_pose, r_pose
        else:
            # both: align time length, concat over features
            Tf = min(f_pose.size(1), f_mot.size(1))
            Tr = min(r_pose.size(1), r_mot.size(1))
            f = torch.cat([f_pose[:, :Tf], f_mot[:, :Tf]], dim=2)    # [B,Tf,168]
            r = torch.cat([r_pose[:, :Tr], r_mot[:, :Tr]], dim=2)    # [B,Tr,168]
            # Also align real/fake time for batch concat inside D (GAP handles length)
            T = min(f.size(1), r.size(1))
            return f[:, :T], r[:, :T]

    def _gan_losses(self, pred_xy, gt_xy):
        """
        LSGAN losses:
          D_loss = MSE(D(real), 1) + λ_d * MSE(D(fake.detach), 0)
          G_adv  = MSE(D(fake), 1)
        """
        mse = F.mse_loss
        xf, xr = self._disc_inputs(pred_xy, gt_xy)    # [B,T',C]
        d_real = self.disc(xr)                        # [B,1]
        d_fake = self.disc(xf.detach())               # [B,1]
        tgt1 = torch.ones_like(d_real)
        tgt0 = torch.zeros_like(d_fake)
        d_loss = mse(d_real, tgt1) + self.lambda_d * mse(d_fake, tgt0)

        g_fake = self.disc(xf)                        # [B,1]
        g_adv  = mse(g_fake, tgt1)
        return g_adv, d_loss

    # ----------------------- training / validation -----------------------
    def training_step(self, batch, batch_idx):
        # fetch optimizers
        if self.use_gan:
            opt_g, opt_d = self.optimizers()
        else:
            (opt_g,) = self.optimizers()

        audio = batch["audio"].float().to(self.device, non_blocking=True)

        # --- no-GAN path ---
        if not self.use_gan:
            pred = self(audio)
            reg_loss = self._cal_and_log_loss(batch, pred, 'train')
            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(reg_loss)
            opt_g.step()
            return reg_loss

        # --- with GAN: D then G (recompute pred for G) ---
        gt = batch["target"].float().to(self.device)

        # Train D on detached fakes
        with torch.no_grad():
            pred_d = self(audio)                         # fake for D (no grad to G)
        _, d_loss = self._gan_losses(pred_d, gt)
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        # Train G (fresh forward)
        pred = self(audio)
        reg_loss = self._cal_and_log_loss(batch, pred, 'train')
        g_adv, _ = self._gan_losses(pred, gt)
        g_loss = reg_loss + self.lambda_gan * g_adv
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({
            "train/reg_loss": reg_loss.detach(),
            "train/g_loss_adv": g_adv.detach(),
            "train/d_loss": d_loss.detach(),
            "train/total_loss": g_loss.detach(),
        }, prog_bar=True, on_step=True, on_epoch=True)

        return g_loss

    def validation_step(self, batch, _):
        audio = batch["audio"].float().to(self.device)
        pred = self(audio)
        loss = self._cal_and_log_loss(batch, pred, 'val')

        if self.use_gan:
            with torch.no_grad():
                gt = batch["target"].float().to(self.device)
                g_adv, d_loss = self._gan_losses(pred, gt)
            self.log_dict({"val/g_loss_adv": g_adv, "val/d_loss": d_loss}, prog_bar=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        if not self.use_gan:
            opt_g = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
            sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=100_000)
            return [opt_g], [sch_g]

        # split params cleanly
        g_params = [p for n, p in self.named_parameters() if not n.startswith("disc.")]
        d_params = list(self.disc.parameters())

        opt_g = torch.optim.AdamW(g_params, lr=self.lr, weight_decay=self.wd)
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=100_000)
        opt_d = torch.optim.AdamW(d_params, lr=self.lr_d, weight_decay=0.0)

        return [opt_g, opt_d], [sch_g]

