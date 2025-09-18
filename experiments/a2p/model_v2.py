# echomimic_v2/experiments/a2p/model_v2.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .audio_encoder_tcn import TemporalAudioEncoderTCN
from .audio_encoder import TemporalAudioEncoder
from .keypoint_head import A2PKeypointHead
from .discriminator import Discriminator1D
from .losses import supervised_loss
from .gan_losses import gan_losses
import torch.nn as nn

class Audio2Pose(pl.LightningModule):
    def __init__(self, fps=24, lr=1e-4, wd=0.01,
                 # capacity knobs
                 d_mid: int = 512, enc_layers: int = 4, head_hidden: int = 512, head_layers: int = 3,
                 head_dropout: float = 0.1,
                 # --- GAN options ---
                 use_gan: bool = False, d_input: str = "motion", lambda_gan: float = 1.0, lambda_d: float = 1.0,
                 lr_d: float = 1e-4,
                 gan_warmup_epochs: int = 5, d_every: int = 1,
                 # other architecture choices
                 enc_type: str = "basic",  # "basic" or "tcn"
                 tcn_dilations: str = "1,2,4,8,16,32",
                 tcn_stacks: int = 1,
                 tcn_dropout: float = 0.1,
                 tcn_kernel: int = 5,

                 use_init_pose: bool = False, cond_mode: str = "add",
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.use_init_pose = use_init_pose
        self.cond_mode = cond_mode
        self.fps = int(fps)
        self.lr, self.wd = lr, wd
        # 1) encoder (build this first so we know out_dim)
        # base encoder+head (keep your original)
        if enc_type == "basic":
            self.enc = TemporalAudioEncoder(d_in=80, d_mid=d_mid, n_layers=enc_layers)
        elif enc_type == "tcn":
            dil = tuple(int(x) for x in tcn_dilations.split(",") if x)
            self.enc = TemporalAudioEncoderTCN(
                d_in=80, d_mid=d_mid, dilations=dil, n_stacks=tcn_stacks,
                dropout=tcn_dropout, k=tcn_kernel
            )
        else:
            raise ValueError(f"Unknown enc_type: {enc_type}")

        d_enc = self.enc.out_dim  # encoder feature width
        # 2) conditioning projector (pose->[d_enc])
        # (84 = 2 hands × 21 joints × (x,y))
        # projects a single frame pose [84] -> encoder width, then broadcast over time
        # (84 = 2 hands × 21 joints × (x,y))
        self.pose_cond = nn.Sequential(
            nn.Linear(84, d_mid), nn.GELU(),
            nn.Linear(d_mid, d_enc)
        )

        # 3) head input dim (if you ever set cond_mode="cat", head sees d_enc*2)
        head_in = d_enc * 2 if (self.use_init_pose and self.cond_mode == "cat") else d_enc
        self.head = A2PKeypointHead(head_in, n_kp=42,
                                    hidden=head_hidden, n_layers=head_layers, dropout=head_dropout)

        # --- GAN setup ---
        self.use_gan = bool(use_gan)
        self.d_input = d_input
        self.lambda_gan = float(lambda_gan)
        self.lambda_d = float(lambda_d)
        self.lr_d = float(lr_d)
        if self.use_gan:
            self.disc = Discriminator1D(c_in=84)

        self.gan_warmup_epochs = int(gan_warmup_epochs)
        self.curr_lambda_gan = 0.0
        self.d_every = int(d_every)  # train D every k steps

        # manual optimization for (G,D)
        self.automatic_optimization = False



    def on_train_epoch_start(self):
        # Warm up the GAN weight
        if self.use_gan:
            w = min(1.0, (self.current_epoch + 1) / max(1, self.gan_warmup_epochs))
            self.curr_lambda_gan = self.lambda_gan * w
            self.log("train/lambda_gan", self.curr_lambda_gan, prog_bar=True)

    # --------- forward & supervised losses (unchanged semantics) ---------
    def forward(self, audio_feats, init_pose=None):
        enc = self.enc(audio_feats)  # [B,T,d]

        if self.use_init_pose and init_pose is not None:
            # init_pose: [B, 2,21,2] -> [B,84]
            B, T = enc.shape[0], enc.shape[1]
            pose_flat = init_pose.view(B, -1)  # [B,84]
            cond = self.pose_cond(pose_flat).unsqueeze(1)  # [B,1,d]
            cond = cond.expand(-1, T, -1)  # [B,T,d]

            if self.cond_mode == "cat":
                # If you prefer concat, change the head’s input dim accordingly.
                x = torch.cat([enc, cond], dim=-1)
            else:
                # default: additive bias
                x = enc + cond
        else:
            x = enc

        pred = self.head(x)  # [B,T,2,21,2]
        return pred.clamp(0, 1)  # stay in image range (no in-place)

    def _cal_and_log_loss(self, batch, pred, label: str):
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite prediction; check mel features / dB clamp.")

        # set weights here (keeps model.py simple; tweak anytime)
        reg_loss, parts = supervised_loss(
            pred, batch,
            fps=self.fps,
            weight_pos=5.0,  # keep current behavior
            weight_bone=0.0,  # turn on later by setting >0
            weight_vel=0.0,  # turn on later by setting >0
        )

        # minimal logging (add more if you like)
        self.log_dict(
            {f"{label}/l_pos": parts["l_pos"], f"{label}/reg_loss": parts["reg_loss"]},
            prog_bar=True, on_step=False, on_epoch=True
        )
        return reg_loss

    # ----------------------- training / validation -----------------------
    def training_step(self, batch, batch_idx):
        # fetch optimizers
        if self.use_gan:
            opt_g, opt_d = self.optimizers()
        else:
            opt_g = self.optimizers()

        audio = batch["audio"].float().to(self.device, non_blocking=True)
        init_pose = batch.get("init_pose", None)
        if init_pose is not None:
            init_pose = init_pose.to(self.device)

        # --- no-GAN path ---
        if not self.use_gan:
            pred = self(audio, init_pose=init_pose)
            reg_loss = self._cal_and_log_loss(batch, pred, 'train')
            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(reg_loss)
            opt_g.step()
            self.log("train/loss", reg_loss.detach(), prog_bar=True, on_step=False, on_epoch=True)
            return reg_loss

        # --- with GAN: D then G (recompute pred for G) ---
        gt = batch["target"].float().to(self.device)

        # Train D on detached fakes
        with torch.no_grad():
            pred_d = self(audio, init_pose=init_pose)
        _, d_loss = gan_losses(pred_d, gt, batch, self.disc,
                               lambda_d=self.lambda_d, fps=self.fps, d_input=self.d_input)
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        # Train G (fresh forward)
        pred = self(audio, init_pose=init_pose)
        reg_loss = self._cal_and_log_loss(batch, pred, 'train')
        g_adv, _ = gan_losses(pred, gt, batch, self.disc,
                              lambda_d=self.lambda_d, fps=self.fps, d_input=self.d_input)
        g_loss = reg_loss + self.curr_lambda_gan * g_adv
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({
            "train/g_loss_adv": g_adv.detach(),
            "train/d_loss": d_loss.detach(),
            "train/loss": g_loss.detach(),
        }, prog_bar=True, on_step=False, on_epoch=True)

        return g_loss

    def validation_step(self, batch, _):
        audio = batch["audio"].float().to(self.device)
        init_pose = batch.get("init_pose", None)
        if init_pose is not None:
            init_pose = init_pose.to(self.device)
        with torch.no_grad():
            pred = self(audio, init_pose=init_pose)

            # supervised part (this already logs val/reg_loss, l_pos, l_bone, l_vel)
            reg_loss = self._cal_and_log_loss(batch, pred, 'val')

            if self.use_gan:
                gt = batch["target"].float().to(self.device)
                g_adv, d_loss = gan_losses(pred, gt, batch, self.disc,
                                           lambda_d=self.lambda_d, fps=self.fps, d_input=self.d_input)
                total = reg_loss + self.curr_lambda_gan * g_adv

                # log val metrics & a total that matches training's objective
                self.log_dict({"val/g_loss_adv": g_adv, "val/d_loss": d_loss, "val/loss": total})
                return total
            else:
                self.log("val/loss", reg_loss)
                return reg_loss

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


if __name__ == '__main__':
    model1 = TemporalAudioEncoder()
    mel = torch.randn(2, 64, 80)
    out1 = model1(mel)
    assert torch.Size([2, 64, 256]) == out1.shape  # [2,64,256]

    model2 = A2PKeypointHead(d_in=model1.out_dim, n_kp=42)
    out2 = model2(out1)
    assert torch.Size([2, 64, 2, 21, 2]) == out2.shape  # [2,64,2,21,2]

    # simple test
    # model = Audio2Pose(mode="keypoints", use_gan=True)
    # mel = torch.randn(2, 64, 80)
    # out = model(mel)
    # print(out.shape)  # [2,64,2,21,2]
    # print(out.min(), out.max())
