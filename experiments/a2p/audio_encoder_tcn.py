# echomimic_v2/experiments/a2p/audio_encoder_tcn.py
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.act2 = nn.GELU()
        self.do = nn.Dropout(dropout)
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.ln = nn.LayerNorm(c_out)

    def forward(self, x_bct):  # [B,C,T]
        h = self.conv1(x_bct);
        h = self.act1(h)
        h = self.conv2(h);
        h = self.act2(h)
        h = self.do(h)
        h = h + self.proj(x_bct)  # residual
        # layernorm over channel -> swap to [B,T,C], norm, back to [B,C,T]
        h = h.transpose(1, 2)
        h = self.ln(h)
        return h.transpose(1, 2)


class TemporalAudioEncoderTCN(nn.Module):
    # in: [B,T,80]  out: [B,T,d_mid]
    def __init__(self, d_in=80, d_mid=512, dilations=(1, 2, 4, 8, 16, 32), n_stacks=1, dropout=0.1, k=5):
        super().__init__()
        layers = [nn.Conv1d(d_in, d_mid, 1)]  # cheap lift
        for _ in range(n_stacks):
            for d in dilations:
                layers.append(TCNBlock(d_mid, d_mid, k=k, dilation=d, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.out_dim = d_mid

    def forward(self, x):  # [B,T,D]
        y = self.net(x.transpose(1, 2))  # [B,d,T]
        return y.transpose(1, 2)  # [B,T,d]
