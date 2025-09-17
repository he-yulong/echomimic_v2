# echomimic_v2/experiments/a2p/losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F

# hand graph + palm anchors (for bone/palm terms)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
PALM_ANCHORS = [5, 9, 13, 17]
EPS = 1e-6


def _bone_lengths(kp: torch.Tensor) -> torch.Tensor:
    # kp: [B,T,2,21,2]  -> [B,T,2,E]
    a = torch.stack([kp[..., i, :] for (i, j) in EDGES], dim=-2)
    b = torch.stack([kp[..., j, :] for (i, j) in EDGES], dim=-2)
    return ((a - b).pow(2).sum(-1) + EPS).sqrt()


def _palm_scale(kp: torch.Tensor) -> torch.Tensor:
    # mean wrist->anchor distances; returns [B,T,2,1]
    wrist = kp[..., 0:1, :]
    anchors = torch.stack([kp[..., i, :] for i in PALM_ANCHORS], dim=-2)
    d = ((anchors - wrist).pow(2).sum(-1) + EPS).sqrt()
    return d.mean(-1, keepdim=True) + EPS


def _center_by_anchor(pred_xy: torch.Tensor, gt_xy: torch.Tensor, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Center both pred and gt by an anchor.
    Priority:
      1) if batch["neck"] exists: subtract that (shape [B,T,2])
      2) else: subtract wrist (joint 0) per hand (robust fallback)
    """
    if isinstance(batch, dict) and ("neck" in batch):
        neck = batch["neck"].float().to(pred_xy.device)  # [B, T, 2]
        anchor = neck.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, 2]
        return pred_xy - anchor, gt_xy - anchor

    wrist = gt_xy[..., 0:1, :].detach()  # [B, T, 2, 1, 2]
    return pred_xy - wrist, gt_xy - wrist


def supervised_loss(
        pred_xy: torch.Tensor,
        batch: dict,
        *,
        fps: int,
        weight_pos: float = 5.0,
        weight_bone: float = 0.0,
        weight_vel: float = 0.0,
        beta: float = 0.02,
        conf_floor: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes position/bone/velocity losses with confidence weighting and returns:
      (total_loss, {"l_pos":..., "l_bone":..., "l_vel":..., "reg_loss": total_loss})
    Notes:
      - Translation-invariant via neck (if present) or wrist(0) fallback.
      - Velocity and discriminator use px/second via `fps`.
    """
    device = pred_xy.device
    gt = batch["target"].float().to(device)  # [B,T,2,21,2]
    cf = batch["conf"].float().to(device)  # [B,T,2,21]

    # center
    pred_c, gt_c = _center_by_anchor(pred_xy, gt, batch)

    # 1) Position (confidence-weighted SmoothL1)
    pos_err = F.smooth_l1_loss(pred_c, gt_c, beta=beta, reduction="none")  # [B,T,2,21,2]
    w_joint = (conf_floor + cf).unsqueeze(-1)  # [B,T,2,21,1]
    l_pos = (w_joint * pos_err).mean()

    # 2) Bone length (normalized by palm size)
    Lp = _bone_lengths(pred_c)
    Lg = _bone_lengths(gt_c).detach()
    Sg = _palm_scale(gt_c).detach()
    Lp_n, Lg_n = Lp / Sg, Lg / Sg

    edge_w = torch.stack(
        [torch.minimum(cf[..., i], cf[..., j]) for (i, j) in EDGES], dim=-1
    )  # [B,T,2,E]
    l_bone = (conf_floor + edge_w) * F.smooth_l1_loss(Lp_n, Lg_n, beta=beta, reduction="none")
    l_bone = l_bone.mean()

    # 3) Velocity (Δ per sec)
    if pred_c.size(1) > 1:
        dv_pred = (pred_c[:, 1:] - pred_c[:, :-1]) * fps
        dv_gt = (gt_c[:, 1:] - gt_c[:, :-1]).detach() * fps
        cf_mid = 0.5 * (cf[:, 1:] + cf[:, :-1])  # [B,T-1,2,21]
        l_vel = (conf_floor + cf_mid.unsqueeze(-1)) * F.smooth_l1_loss(
            dv_pred, dv_gt, beta=beta, reduction="none"
        )
        l_vel = l_vel.mean()
    else:
        l_vel = torch.zeros((), device=device)

    total = weight_pos * l_pos + weight_bone * l_bone + weight_vel * l_vel
    parts = {"l_pos": l_pos, "l_bone": l_bone, "l_vel": l_vel, "reg_loss": total}
    return total, parts
