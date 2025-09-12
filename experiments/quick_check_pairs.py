# quick_check_pairs.py
from a2p_train import make_pairs
for lf in ("./train.tsv", "./val.tsv"):
    try:
        pairs = make_pairs(lf)
        print(f"{lf}: OK, {len(pairs)} pairs")
    except Exception as e:
        print(f"{lf}: ERROR → {e}")
