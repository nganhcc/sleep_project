import numpy as np
import os

def load_processed_data(data_path):
    """
    Load processed data from .npz file
    
    Parameters
    ----------
    data_path : str
        Path to the .npz file
        
    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    feature_names : list
        List of feature names
    """
    # Load the .npz file
    data = np.load(data_path)
    
    # Extract the arrays
    X = data['X']  # Features
    y = data['y']  # Labels
    feature_names = data['feature_names']  # Feature names
    
    print(f"Loaded data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names

def load_all_splits(data_dir="data/processed"):
    """
    Load all dataset splits (train, val, test)
    
    Parameters
    ----------
    data_dir : str
        Directory containing the processed data
        
    Returns
    -------
    dict
        Dictionary containing train, val, test data
    """
    splits = {}
    
    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(data_dir, split_name, f"{split_name}.npz")
        
        if os.path.exists(split_path):
            print(f"\nLoading {split_name} data...")
            X, y, feature_names = load_processed_data(split_path)
            splits[split_name] = {
                'X': X,
                'y': y,
                'feature_names': feature_names
            }
        else:
            print(f"Warning: {split_path} not found")
    
    return splits

# Example usage
if __name__ == "__main__":
    # Method 1: Load just the training data
    print("=== Loading Training Data ===")
    train_path = "data/processed/train/train.npz"
    X_train, y_train, feature_names = load_processed_data(train_path)
    
    # Method 2: Load all splits
    print("\n=== Loading All Splits ===")
    all_data = load_all_splits()
    
    # Access the data
    if 'train' in all_data:
        print(f"\nTrain data shape: {all_data['train']['X'].shape}")
        print(f"Train labels shape: {all_data['train']['y'].shape}")
    
    if 'val' in all_data:
        print(f"Validation data shape: {all_data['val']['X'].shape}")
    
    if 'test' in all_data:
        print(f"Test data shape: {all_data['test']['X'].shape}")
