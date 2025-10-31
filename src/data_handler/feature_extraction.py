import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis


def extract_features(X, sfreq):
    feature_list = []
    for epoch in X:
        epoch_features = []
        for ch in epoch:
            # ---- Time-domain features ----
            mean_val = np.mean(ch)
            std_val = np.std(ch)
            skew_val = skew(ch)
            kurt_val = kurtosis(ch)
            rms = np.sqrt(np.mean(ch**2))
            zero_cross = ((ch[:-1] * ch[1:]) < 0).sum()

            # ---- Frequency-domain features (using Welch PSD) ----
            f, psd = welch(ch, fs=sfreq, nperseg=sfreq * 2)
            total_power = np.sum(psd)

            # Band powers
            delta = np.sum(psd[(f >= 0.5) & (f < 4)]) / total_power
            theta = np.sum(psd[(f >= 4) & (f < 8)]) / total_power
            alpha = np.sum(psd[(f >= 8) & (f < 12)]) / total_power
            beta = np.sum(psd[(f >= 12) & (f < 30)]) / total_power

            # Append all features for this channel
            ch_features = [
                mean_val,
                std_val,
                skew_val,
                kurt_val,
                rms,
                zero_cross,
                delta,
                theta,
                alpha,
                beta,
            ]
            epoch_features.extend(ch_features)

        feature_list.append(epoch_features)

    features = np.array(feature_list)

    return features
