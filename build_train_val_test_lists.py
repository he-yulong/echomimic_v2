import random, pathlib

ROOT = pathlib.Path("./EMTD_dataset")
AUDIO = ROOT / "processed" / "audio"
POSE = ROOT / "image_audio_features" / "pose"
LISTS = ROOT / "lists"
LISTS.mkdir(parents=True, exist_ok=True)

pairs = []
for wav in sorted(AUDIO.glob("*.wav")):
    stem = wav.stem  # e.g. 0BF2…-Scene-001
    pdir = POSE / stem
    if pdir.exists() and any(pdir.glob("*.npy")):
        pairs.append((str(wav.resolve()), str(pdir.resolve())))

random.seed(1234)
random.shuffle(pairs)
n = len(pairs)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
n_test = n - n_train - n_val
splits = {
    "train.csv": pairs[:n_train],
    "val.csv": pairs[n_train:n_train + n_val],
    "test.csv": pairs[n_train + n_val:],
}
for name, rows in splits.items():
    with open(LISTS / name, "w", encoding="utf-8") as f:
        for a, p in rows: f.write(f"{a},{p}\n")

print("counts:", {k: len(v) for k, v in splits.items()})
print("example line:", splits["train.csv"][0] if splits["train.csv"] else "EMPTY")
