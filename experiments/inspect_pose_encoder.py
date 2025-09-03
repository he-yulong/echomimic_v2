# experiments/inspect_pose_encoder.py
import torch
import torch.nn.functional as F
from src.models.pose_encoder import PoseEncoder

def inspect_pose_encoder(
    T=12, H=128, W=128,
    conditioning_channels=3,
    block_out=(16,32,96,256),
    emb_ch=320,
    device="cuda"
):
    pe = PoseEncoder(
        conditioning_embedding_channels=emb_ch,
        conditioning_channels=conditioning_channels,
        block_out_channels=block_out,
    ).to(device).eval()

    x = torch.zeros(1, conditioning_channels, T, H, W, device=device)

    logs = []
    h = F.silu(pe.conv_in(x));                logs.append(("conv_in", h.shape))
    for k, block in enumerate(pe.blocks):
        h = F.silu(block(h));                 logs.append((f"block[{k}]", h.shape))
    y = pe.conv_out(h);                       logs.append(("conv_out", y.shape))

    # Pretty print
    print("\nPoseEncoder shape trace:")
    for name, shp in logs:
        print(f"{name:>12}: {tuple(shp)}")
    n_params = sum(p.numel() for p in pe.parameters())
    n_train  = sum(p.numel() for p in pe.parameters() if p.requires_grad)
    print(f"\nParams: total={n_params/1e6:.2f}M, trainable={n_train/1e6:.2f}M")

if __name__ == "__main__":
    inspect_pose_encoder(T=12, H=768, W=768, device="cuda")  # feel free to try 768 if VRAM allows
