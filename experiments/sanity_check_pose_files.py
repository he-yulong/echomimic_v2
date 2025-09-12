import numpy as np, pathlib, collections

POSE = pathlib.Path("../EMTD_dataset/image_audio_features/pose")
shapes = collections.Counter()
bad = []
four = []
for d in sorted(POSE.iterdir()):
    if not d.is_dir(): continue
    for f in sorted(d.glob("*.npy")):
        x = np.load(f, allow_pickle=True).item()
        h = x.get("hands", None)
        if h is None: continue
        h = np.asarray(h)
        shapes[str(getattr(h, "shape", None))] += 1
        # strict check: require EXACT (2,21,2)
        if not (h.ndim == 3 and h.shape[0] >= 2 and h.shape[1] == 21 and h.shape[2] == 2):
            bad.append((str(f), getattr(h, "shape", None)))
        if h.ndim == 3 and h.shape == (4, 21, 2):
            four.append(str(f))
print("shape counts:", shapes)
print("bad:", len(bad))
for b in bad[:20]: print(b)
print("4-hand frames:", len(four))
for p in four[:20]: print(p)
