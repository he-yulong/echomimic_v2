# echomimic_v2/experiments/a2p/keypoint_head.py
import torch
import torch.nn as nn


class A2PKeypointHead(nn.Module):
    # in:  [B, T, d]
    # out: [B, T, 2, 21, 2]   (2 hands × 21 joints × (x,y))
    def __init__(self, d_in, n_kp=42, hidden=512, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(d, hidden), nn.GELU(),
                       nn.LayerNorm(hidden), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, n_kp * 2)]
        self.net = nn.Sequential(*layers)
        self.n_kp = n_kp

    def forward(self, x):
        B, T, _ = x.shape
        y = self.net(x).view(B, T, 2, 21, 2)
        return torch.sigmoid(y)


__all__ = ["A2PKeypointHead"]
