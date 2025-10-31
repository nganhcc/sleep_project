import numpy as np


def preprocess_signals(raw, channels=None, l_freq=0.3, h_freq=35.0):
    if channels:
        raw.pick_channels(channels)
    raw.filter(l_freq=l_freq, h_freq=h_freq)  # filter
    raw.resample(100)  # resample to 100 Hz
    data = raw.get_data()  # shape: (n_channels, n_samples)
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(
        data, axis=1, keepdims=True
    )  # normalize
    return data, raw.info["sfreq"]


def segment_data(data, labels, sfreq, epoch_len=30):
    samples_per_epoch = int(epoch_len * sfreq)
    n_epochs = len(labels)
    X = np.zeros((n_epochs, data.shape[0], samples_per_epoch))
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        X[i] = data[:, start:end]
    return X, labels
