import os
import json
import numpy as np
import logging
from data_loader import load_psg, load_hyp, attach
from preprocessing import preprocess_signals, segment_data
from feature_extraction import extract_features
from balance import balance_classes

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_split(split_data, out_dir, split_name):
    """
    Load, preprocess, extract features, balance, and save for one dataset split.
    
    Parameters
    ----------
    split_data : list
        List of dictionaries containing 'psg' and 'hyp' file paths
    out_dir : str
        Output directory for processed data
    split_name : str
        Name of the split (train/valid/test)
    """
    X_all, y_all = [], []
    
    logger.info(f"Processing {split_name} split with {len(split_data)} files")
    
    for file_pair in split_data:
        try:
            psg_path = file_pair['psg']
            hyp_path = file_pair['hyp']
            
            # Extract record ID from filename
            record_id = os.path.basename(psg_path).split("-")[0]
            
            # Check if files exist
            if not os.path.exists(psg_path):
                logger.warning(f"PSG file not found: {psg_path}")
                continue
            if not os.path.exists(hyp_path):
                logger.warning(f"Hypnogram file not found: {hyp_path}")
                continue

            logger.info(f"Processing {record_id}...")

            # Step 1: Load data
            raw = load_psg(psg_path)
            annots = load_hyp(hyp_path)
            raw = attach(raw, annots)

            # Step 2: Preprocess signals
            data, sfreq = preprocess_signals(raw, channels=['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'], l_freq=0.3, h_freq=35.0)
            
            # Extract labels from annotations
            labels = []
            for desc in annots.description:
                if desc in ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage R']:
                    # Map sleep stages to numeric labels
                    stage_map = {
                        'Sleep stage W': 0,  # Wake
                        'Sleep stage 1': 1,  # N1
                        'Sleep stage 2': 2,  # N2
                        'Sleep stage 3': 3,  # N3
                        'Sleep stage R': 4   # REM
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
                logger.warning(f"No valid epochs found for {record_id}")
                continue

            # Step 4: Extract features
            X_feat, feature_names = extract_features(X_seg, sfreq)

            # Collect
            X_all.append(X_feat)
            y_all.append(y_seg)
            
            logger.info(f"Processed {record_id}: {len(X_feat)} epochs, {len(feature_names)} features per epoch")

        except Exception as e:
            logger.error(f"Error processing {record_id}: {str(e)}")
            continue

    if len(X_all) == 0:
        logger.error(f"No valid data found in {split_name} split")
        return

    # Step 5: Combine and balance
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    
    logger.info(f"Combined data shape: {X_all.shape}, labels shape: {y_all.shape}")
    logger.info(f"Class distribution: {np.bincount(y_all)}")
    
    # Balance classes
    X_bal, y_bal = balance_classes(X_all, y_all, method="smote")
    
    logger.info(f"Balanced data shape: {X_bal.shape}, labels shape: {y_bal.shape}")
    logger.info(f"Balanced class distribution: {np.bincount(y_bal)}")

    # Step 6: Save processed data
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{split_name}.npz")
    
    np.savez_compressed(output_path,
                        X=X_bal, 
                        y=y_bal,
                        feature_names=feature_names)
    
    logger.info(f"Saved processed data to {output_path}")

def main():
    """Main function to process all dataset splits."""
    splits_file = "data/split/splits.json"
    out_base = "data/processed"

    # Load splits configuration
    if not os.path.exists(splits_file):
        logger.error(f"Splits file not found: {splits_file}")
        return
    
    with open(splits_file, 'r') as f:
        splits_data = json.load(f)
    
    # Process each split
    for split_name in ["train", "val", "test"]:
        if split_name not in splits_data:
            logger.warning(f"Split '{split_name}' not found in splits file")
            continue
            
        split_data = splits_data[split_name]
        out_dir = os.path.join(out_base, split_name)
        
        logger.info(f"Processing {split_name} split...")
        process_split(split_data, out_dir, split_name)
        logger.info(f"Completed processing {split_name} split")

if __name__ == "__main__":
    main()
