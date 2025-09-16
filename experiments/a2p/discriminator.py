# echomimic_v2/experiments/a2p/discriminator.py
import torch
import torch.nn as nn


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


__all__ = ["Discriminator1D"]
