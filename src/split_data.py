import os
import json
import numpy as np


def pair_records(data_dir):
    """
    Pair PSG and Hypnogram EDF files by subject ID (first 6 chars).
    Example:
      SC4001E0-PSG.edf  <->  SC4001EC-Hypnogram.edf
    """
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".edf")]
    pairs = []
    ids = sorted(set(f[:6] for f in files))

    for sid in ids:
        # find PSG and Hypnogram for this subject
        psg = next(
            (f for f in files if f.startswith(sid) and "-psg" in f.lower()), None
        )
        hyp = next(
            (f for f in files if f.startswith(sid) and "hypnogram" in f.lower()), None
        )

        if psg and hyp:
            pairs.append((os.path.join(data_dir, psg), os.path.join(data_dir, hyp)))
        else:
            print(f"[WARN] Missing pair for {sid}: PSG={psg}, HYP={hyp}")

    print(f"[INFO] Found {len(pairs)} valid PSG–Hypnogram pairs")
    return pairs


def split_records(pairs, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split paired records into train, validation, and test sets."""
    np.random.seed(seed)
    np.random.shuffle(pairs)

    n = len(pairs)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]

    print(f"[INFO] Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def save_splits(train, val, test, out_file="data/split/splits.json"):
    data = {
        "train": [{"psg": p, "hyp": h} for p, h in train],
        "val": [{"psg": p, "hyp": h} for p, h in val],
        "test": [{"psg": p, "hyp": h} for p, h in test],
    }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved split file: {out_file}")


if __name__ == "__main__":
    data_dir = "data/raw/sleep-cassette"  # change this to your folder
    pairs = pair_records(data_dir)
    train, val, test = split_records(pairs)
    save_splits(train, val, test)
