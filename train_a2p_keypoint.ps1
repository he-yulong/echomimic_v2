# train_a2p.ps1
param(
  [string]$Root = $PSScriptRoot
)

$lists = Join-Path $Root "EMTD_dataset\lists"
$out   = Join-Path $Root "a2p_runs"
$MODE = "keypoints"   # was "heatmap"

  #--mel_root  EMTD_dataset\features\mels `
  #--pose_root EMTD_dataset\features\poses `
# Easiest: run in the EM2 env without activating the shell
python -m experiments.a2p.train  `
  --train_list (Join-Path $lists "train.csv") `
  --val_list   (Join-Path $lists "val.csv") `
  --mode $MODE --fps 24 --win_T 12 --hop_T 3 `
  --bs 8 --epochs 1000 --num_workers 12 `
  --out $out
