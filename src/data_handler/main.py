import os
import json
import numpy as np
from raw_data_loader import load_psg, load_hyp, attach
from preprocessing import preprocess_signals, segment_data
from feature_extraction import extract_features


def process_split(split_data, out_dir, split_name):
    X_all, y_all = [], []

    for file_pair in split_data:
        try:
            psg_path = file_pair["psg"]
            hyp_path = file_pair["hyp"]

            # Check if files exist
            if not os.path.exists(psg_path):
                continue
            if not os.path.exists(hyp_path):
                continue

            # Step 1: Load data
            raw = load_psg(psg_path)
            annots = load_hyp(hyp_path)
            raw = attach(raw, annots)

            # Step 2: Preprocess signals
            data, sfreq = preprocess_signals(
                raw,
                channels=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
                l_freq=0.3,
                h_freq=35.0,
            )

            # Extract labels from annotations
            labels = []
            for desc in annots.description:
                if desc in [
                    "Sleep stage W",
                    "Sleep stage 1",
                    "Sleep stage 2",
                    "Sleep stage 3",
                    "Sleep stage R",
                ]:
                    # Map sleep stages to numeric labels
                    stage_map = {
                        "Sleep stage W": 0,  # Wake
                        "Sleep stage 1": 1,  # N1
                        "Sleep stage 2": 2,  # N2
                        "Sleep stage 3": 3,  # N3
                        "Sleep stage R": 4,  # REM
                    }
                    labels.append(stage_map[desc])
                else:
                    labels.append(-1)  # Unknown stage

            # Convert to numpy array
            labels = np.array(labels)

            # Step 3: Segment data into epochs
            X_seg, y_seg = segment_data(data, labels, sfreq, epoch_len=30)

            # Filter out unknown stages
            valid_mask = y_seg != -1
            X_seg = X_seg[valid_mask]
            y_seg = y_seg[valid_mask]

            if len(X_seg) == 0:
                continue

            # Step 4: Extract features
            X_feat = extract_features(X_seg, sfreq)

            # Collect
            X_all.append(X_feat)
            y_all.append(y_seg)

        except Exception as e:
            continue

    if len(X_all) == 0:
        return

    # Step 5: Combine and balance
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    # Step 6: Save processed data
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{split_name}.npz")

    np.savez_compressed(output_path, X=X_all, y=y_all)


def main():
    splits_file = "data/split/splits.json"
    out_base = "data/processed"

    if not os.path.exists(splits_file):
        return

    with open(splits_file, "r") as f:
        splits_data = json.load(f)

    # Process each split
    for split_name in ["train", "test"]:
        if split_name not in splits_data:
            continue

        split_data = splits_data[split_name]
        out_dir = os.path.join(out_base, split_name)

        process_split(split_data, out_dir, split_name)


if __name__ == "__main__":
    main()
