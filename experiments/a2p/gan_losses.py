# echomimic_v2/experiments/a2p/gan_losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from .losses import _center_by_anchor  # uses the same centering rule as supervised loss

def _to_motion_delta(seq_xy: torch.Tensor) -> torch.Tensor:
    # [B,T,2,21,2] -> [B,T-1,2,21,2]
    if seq_xy.size(1) <= 1:
        return seq_xy[:, :0]
    return seq_xy[:, 1:] - seq_xy[:, :-1]

def _flatten_tp22(x: torch.Tensor) -> torch.Tensor:
    # [B,T,2,21,2] -> [B,T,84]
    return x.reshape(x.size(0), x.size(1), -1)

def build_disc_inputs(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    batch: dict,
    *,
    fps: int,
    d_input: str,  # "motion" | "pose" | "both"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center pose (neck if present, else wrist), build D inputs as [B,T',C]."""
    pred_c, gt_c = _center_by_anchor(pred_xy, gt_xy, batch)

    f_pose = _flatten_tp22(pred_c)                       # [B,T,84]
    r_pose = _flatten_tp22(gt_c)                         # [B,T,84]
    f_mot  = _flatten_tp22(_to_motion_delta(pred_c)) * fps  # [B,T-1,84], in px/sec
    r_mot  = _flatten_tp22(_to_motion_delta(gt_c))  * fps

    if d_input == "motion":
        return f_mot, r_mot
    if d_input == "pose":
        return f_pose, r_pose

    # both: align lengths, concat over features
    Tf = min(f_pose.size(1), f_mot.size(1))
    Tr = min(r_pose.size(1), r_mot.size(1))
    f = torch.cat([f_pose[:, :Tf], f_mot[:, :Tf]], dim=2)  # [B,Tf,168]
    r = torch.cat([r_pose[:, :Tr], r_mot[:, :Tr]], dim=2)  # [B,Tr,168]
    T = min(f.size(1), r.size(1))
    return f[:, :T], r[:, :T]

def gan_losses(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    batch: dict,
    disc: torch.nn.Module,
    *,
    lambda_d: float,
    fps: int,
    d_input: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LSGAN:
      D_loss = MSE(D(real), 1) + λ_d * MSE(D(fake.detach), 0)
      G_adv  = MSE(D(fake), 1)
    Returns: (g_adv, d_loss)
    """
    xf, xr = build_disc_inputs(pred_xy, gt_xy, batch, fps=fps, d_input=d_input)
    d_real = disc(xr)
    d_fake = disc(xf.detach())
    tgt1 = torch.ones_like(d_real)
    tgt0 = torch.zeros_like(d_fake)

    d_loss = F.mse_loss(d_real, tgt1) + lambda_d * F.mse_loss(d_fake, tgt0)
    g_adv  = F.mse_loss(disc(xf), tgt1)
    return g_adv, d_loss
