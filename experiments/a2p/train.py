# echomimic_v2/experiments/a2p/train.py
import torch
torch.autograd.set_detect_anomaly(True)
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .data_simple import A2PConfig, make_loaders
from .model import Audio2Pose

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def main():
    import argparse
    pl.seed_everything(1234, workers=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--mel_root", type=str, default="EMTD_dataset/features/mels")
    ap.add_argument("--pose_root", type=str, default="EMTD_dataset/features/poses")
    ap.add_argument("--no_filter_missing", action="store_true",
                    help="Don’t drop rows missing precomputed files")
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--val_list", required=True)
    ap.add_argument("--win_T", type=int, default=12)
    ap.add_argument("--hop_T", type=int, default=3)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--heat_hw", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    # --- GAN options ---
    ap.add_argument("--use_gan", action="store_true",
                    help="Enable GAN loss (only in keypoints mode).")
    ap.add_argument("--d_input", choices=["motion", "pose", "both"], default="motion",
                    help="What the discriminator sees: motion deltas, raw poses, or both.")
    ap.add_argument("--lambda_gan", type=float, default=1.0,
                    help="Multiplier for GAN loss in generator.")
    ap.add_argument("--lambda_d", type=float, default=1.0,
                    help="Weight for discriminator’s fake loss.")
    ap.add_argument("--lr_d", type=float, default=1e-4,
                    help="Learning rate for discriminator.")
    ap.add_argument("--gan_warmup_epochs", type=int, default=10)
    ap.add_argument("--d_every", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="a2p_ckpts")

    ap.add_argument("--d_mid", type=int, default=512)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--head_hidden", type=int, default=512)
    ap.add_argument("--head_layers", type=int, default=3)
    ap.add_argument("--head_dropout", type=float, default=0.1)

    ap.add_argument("--enc_type", choices=["basic", "tcn"], default="basic")
    ap.add_argument("--tcn_dilations", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--tcn_stacks", type=int, default=1)
    ap.add_argument("--tcn_dropout", type=float, default=0.1)
    ap.add_argument("--tcn_kernel", type=int, default=5)

    args = ap.parse_args()

    cfg = A2PConfig(
        fps=args.fps, win_T=args.win_T, hop_T=args.hop_T,
        heat_H=args.heat_hw[0], heat_W=args.heat_hw[1]
    )
    train_dl, val_dl = make_loaders(
        args.train_list, args.val_list, cfg,
        bs=args.bs, num_workers=args.num_workers,
        mel_root=args.mel_root, pose_root=args.pose_root,
        filter_missing=(not args.no_filter_missing),
    )

    model = Audio2Pose(
        fps=args.fps,
        heat_hw=tuple(args.heat_hw),
        lr=args.lr,
        # capacity
        d_mid=args.d_mid,
        enc_layers=args.enc_layers,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        head_dropout=args.head_dropout,
        # GAN
        use_gan=args.use_gan,
        d_input=args.d_input,
        lambda_gan=args.lambda_gan,
        lambda_d=args.lambda_d,
        lr_d=args.lr_d,
    )

    logger = TensorBoardLogger(args.out, name=f"a2p_keypoints")

    ckpt_val = ModelCheckpoint(
        dirpath=logger.log_dir,
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        filename="val-{epoch:02d}-{val_loss:.4f}"
    )

    ckpt_train = ModelCheckpoint(
        dirpath=logger.log_dir,
        save_top_k=2,
        monitor="train/loss",
        mode="min",
        filename="train-{epoch:02d}-{train_loss:.4f}"
    )

    ckpt_last = ModelCheckpoint(
        dirpath=logger.log_dir,
        save_last=True,  # <- this ensures "last.ckpt" is updated after every epoch
        filename="last"  # optional, default is "last.ckpt"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt_val, ckpt_train, ckpt_last],
        # precision="16-mixed",
        # gradient_clip_val=1.0,
        # accumulate_grad_batches=args.accum,
        # devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # log_every_n_steps=100,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
