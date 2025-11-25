import numpy as np
from scipy import stats

def extract_eda_features(eda_segment, sampling_rate):
    """
    Extracts statistical features from a segment of an EDA signal.

    Args:
        eda_segment (np.array): A 1D array representing a segment of EDA data.
        sampling_rate (int): The sampling rate of the signal.

    Returns:
        dict: A dictionary of extracted features.
    """
    features = {}

    # Time-domain features
    features['mean'] = np.mean(eda_segment)
    features['std_dev'] = np.std(eda_segment)
    features['variance'] = np.var(eda_segment)
    features['min'] = np.min(eda_segment)
    features['max'] = np.max(eda_segment)
    features['range'] = features['max'] - features['min']
    features['skewness'] = stats.skew(eda_segment)
    features['kurtosis'] = stats.kurtosis(eda_segment)
    features['mean_abs_diff'] = np.mean(np.abs(np.diff(eda_segment)))

    # You can add frequency-domain features (e.g., using FFT) here as well.

    return features
