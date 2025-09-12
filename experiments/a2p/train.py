# experiments/a2p/train.py
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# from .data import A2PConfig, make_loaders
from .data_simple import A2PConfig, make_loaders
from .model_simple import Audio2Pose

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
    ap.add_argument("--mode", choices=["heatmap", "keypoints"], default="heatmap")
    ap.add_argument("--win_T", type=int, default=12)
    ap.add_argument("--hop_T", type=int, default=3)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--heat_hw", type=int, nargs=2, default=[256, 256])
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
    train_dl, val_dl = make_loaders(
        args.train_list, args.val_list, cfg,
        bs=args.bs, num_workers=args.num_workers,
        mel_root=args.mel_root, pose_root=args.pose_root,
        filter_missing=(not args.no_filter_missing),
    )

    model = Audio2Pose(mode=args.mode, fps=args.fps, heat_hw=tuple(args.heat_hw), lr=args.lr)

    logger = TensorBoardLogger(args.out, name=f"a2p_{args.mode}")
    ckpt = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=3, monitor="val/loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt],
        # precision="16-mixed",
        # gradient_clip_val=1.0,
        # accumulate_grad_batches=args.accum,
        # devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=100,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
