import numpy as np
from tensorflow.keras.models import load_model
import os

def clean_signal_with_autoencoder(raw_signal, model_path, segment_length=700):
    """
    Cleans a raw EDA signal using a pre-trained autoencoder model.

    Args:
        raw_signal (np.array): The raw EDA signal.
        model_path (str): Path to the saved .h5 model file.
        segment_length (int): The length of segments the model was trained on.

    Returns:
        np.array: The cleaned EDA signal.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model exists.")

    autoencoder = load_model(model_path, compile=False)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Pad the signal to be divisible by segment_length
    n_samples = len(raw_signal)
    n_pad = segment_length - (n_samples % segment_length)
    padded_signal = np.pad(raw_signal, (0, n_pad), 'edge')

    # Reshape into segments
    segments = padded_signal.reshape(-1, segment_length, 1)

    # Predict (clean) the segments
    cleaned_segments = autoencoder.predict(segments)

    # Reshape back and remove padding
    cleaned_signal_padded = cleaned_segments.flatten()
    cleaned_signal = cleaned_signal_padded[:n_samples]

    return cleaned_signal
