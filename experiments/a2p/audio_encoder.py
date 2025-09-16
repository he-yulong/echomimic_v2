# echomimic_v2/experiments/a2p/audio_encoder.py
import torch.nn as nn


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
