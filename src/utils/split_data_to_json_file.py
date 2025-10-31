import os
import json
import numpy as np


def pair_records(data_dir):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".edf")]
    pairs = []
    ids = sorted(set(f[:6] for f in files))

    for sid in ids:
        psg = next(
            (f for f in files if f.startswith(sid) and "-psg" in f.lower()), None
        )
        hyp = next(
            (f for f in files if f.startswith(sid) and "hypnogram" in f.lower()), None
        )

        if psg and hyp:
            pairs.append((os.path.join(data_dir, psg), os.path.join(data_dir, hyp)))

    return pairs


def split_records(pairs, train_ratio=0.7, test_ratio=0.3, seed=42):
    np.random.seed(seed)
    np.random.shuffle(pairs)

    n = len(pairs)
    n_train = int(train_ratio * n)

    train = pairs[:n_train]
    test = pairs[n_train:]

    return train, test


def save_splits(train, test, out_file="data/split/splits.json"):
    data = {
        "train": [{"psg": p, "hyp": h} for p, h in train],
        "test": [{"psg": p, "hyp": h} for p, h in test],
    }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    data_dir = "data/raw/sleep-cassette"
    pairs = pair_records(data_dir)
    train, test = split_records(pairs)
    save_splits(train, test)
