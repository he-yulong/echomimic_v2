# infer_a2p_keypoints.py
"""
# from your repo root
python infer_a2p_keypoints.py \
  --audio EMTD_dataset/processed/audio/019_R7jm0-R9N_o.wav \
  --ckpt a2p_ckpts/a2p_keypoints/version_2/checkpoints/epoch=XX-step=YYYY.ckpt \
  --out  demos/019_keypoints.mp4 \
  --fps  24 \
  --win_T 12
"""
import math, os
import numpy as np
import torch
import torchaudio
import cv2

# your training code
from experiments.a2p.model_simple import Audio2Pose
# keep these in sync with training
SAMPLE_RATE = 16000
N_FFT       = 1024
N_MELS      = 80
F_MIN, F_MAX= 50, 7600

# MediaPipe Hands topology (21 points)
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20)    # pinky
    # optional palm chain: (5,9),(9,13),(13,17)
]

def build_mel(fps: int):
    hop_length = round(SAMPLE_RATE / fps)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=hop_length,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0, center=False
    )
    amplog = torchaudio.transforms.AmplitudeToDB()  # match your data_simple.py
    mel_fps = SAMPLE_RATE / hop_length
    return mel, amplog, hop_length, mel_fps

@torch.no_grad()
def predict_keypoints(
    audio_path: str,
    ckpt_path: str,
    fps: int = 24,
    win_T: int = 12,
    device: str | None = None,
):
    """
    Returns:
      kp_seq: np.ndarray [T_out, 2, 21, 2] in normalized [0,1] coords
      out_fps: int (the video FPS you should render at, usually `fps`)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) model
    model = Audio2Pose.load_from_checkpoint(
        ckpt_path, mode="keypoints", fps=fps, heat_hw=(256,256)
    ).to(device).eval()

    # 2) audio -> mels (same as training)
    mel, amplog, hop_length, mel_fps = build_mel(fps)
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(0, keepdim=True)  # mono
    m = amplog(mel(wav)).squeeze(0).transpose(0,1).contiguous()  # [Tm,80]
    m = torch.nan_to_num(m, neginf=-80.0, posinf=0.0)            # just in case

    Tm = m.shape[0]
    if Tm < win_T:
        raise RuntimeError(f"Audio too short: mel frames {Tm} < win_T {win_T}")

    # 3) slide with hop=1 in mel space, take center frame prediction
    center = win_T // 2
    preds = []
    centers = []
    for t0 in range(0, Tm - win_T + 1):      # [t0, t0+win_T)
        clip = m[t0:t0+win_T].unsqueeze(0).to(device)  # [1,win_T,80]
        out  = model(clip)                             # [1,win_T,2,21,2]
        mid  = out[0, center].clamp(0,1).detach().cpu().numpy()
        preds.append(mid)                              # [2,21,2]
        centers.append(t0 + center)                    # mel index of this pred

    preds    = np.stack(preds, 0)          # [Tc,2,21,2], Tc = Tm-win_T+1
    centers  = np.asarray(centers)         # [Tc]
    # convert mel indices to seconds
    t_sec_centers = centers * hop_length / SAMPLE_RATE

    # 4) resample to exact `fps` timeline (simple nearest)
    dur_s  = (Tm * hop_length) / SAMPLE_RATE
    T_out  = max(1, int(round(dur_s * fps)))
    t_sec_out = np.arange(T_out) / float(fps)
    # for each target time, choose nearest center pred
    idx = np.searchsorted(t_sec_centers, t_sec_out, side="left")
    idx = np.clip(idx, 0, len(t_sec_centers)-1)
    # better nearest (left vs right):
    left  = np.clip(idx-1, 0, len(t_sec_centers)-1)
    right = idx
    idx = np.where(
        (t_sec_out - t_sec_centers[left]) <= (t_sec_centers[right] - t_sec_out),
        left, right
    )
    kp_seq = preds[idx]   # [T_out,2,21,2] in [0,1]
    return kp_seq, fps

def draw_hand(canvas, pts_xy01, color, thickness=2, radius=3):
    H, W, _ = canvas.shape
    pts = (pts_xy01 * np.array([W-1, H-1])[None]).astype(int)  # [21,2] -> px
    # bones
    for a,b in HAND_EDGES:
        pa, pb = tuple(pts[a]), tuple(pts[b])
        cv2.line(canvas, pa, pb, color, thickness, lineType=cv2.LINE_AA)
    # joints
    for p in pts:
        cv2.circle(canvas, tuple(p), radius, color, -1, lineType=cv2.LINE_AA)

def save_keypoint_video(
    kp_seq: np.ndarray, out_path: str, fps: int = 24,
    size=(512,512), bg_color=(255,255,255),
    audio_path: str | None = None
):
    """
    kp_seq: [T,2,21,2] normalized [0,1]
    Produces an MP4 with just the keypoints on a blank background.
    If moviepy is installed and audio_path is given, it will mux the audio track.
    """
    H, W = size[1], size[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W,H))
    for t in range(len(kp_seq)):
        frame = np.full((H,W,3), bg_color, np.uint8)
        left, right = kp_seq[t,0], kp_seq[t,1]   # [21,2] each
        draw_hand(frame, left,  (60,90,255))     # BGR colors
        draw_hand(frame, right, (60,180,60))
        vw.write(frame)
    vw.release()

    # optional: mux the original audio
    if audio_path is not None:
        try:
            from moviepy import editor as mpy
            v = mpy.VideoFileClip(out_path)
            a = mpy.AudioFileClip(audio_path).set_duration(v.duration)
            (v.set_audio(a)).write_videofile(
                out_path.replace(".mp4","_with_audio.mp4"),
                codec="libx264", audio_codec="aac",
                temp_audiofile="__temp_aac.m4a", remove_temp=True, verbose=False, logger=None
            )
        except Exception as e:
            print(f"[warn] moviepy not available or failed to mux audio: {e}")

def run(audio_path: str, ckpt_path: str, out_video: str,
        fps=24, win_T=12, device=None):
    kp_seq, out_fps = predict_keypoints(audio_path, ckpt_path, fps=fps, win_T=win_T, device=device)
    save_keypoint_video(kp_seq, out_video, fps=out_fps, size=(512,512), audio_path=audio_path)
    print(f"saved: {out_video}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Audio→Pose (keypoints) inference to video")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt",  required=True, help="Lightning checkpoint (.ckpt)")
    ap.add_argument("--out",   default="a2p_keypoints.mp4")
    ap.add_argument("--fps",   type=int, default=24)
    ap.add_argument("--win_T", type=int, default=12)
    args = ap.parse_args()
    run(args.audio, args.ckpt, args.out, fps=args.fps, win_T=args.win_T)
