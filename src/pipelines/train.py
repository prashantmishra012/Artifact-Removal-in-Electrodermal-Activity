import numpy as np
import os
import sys

# Add src to path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.autoencoder import build_autoencoder
from data.preprocess import preprocess_eda

def generate_synthetic_data(num_samples=10000, signal_len=128, freq=5):
    """
    Generates synthetic clean and noisy EDA-like signals.
    """
    print("Generating synthetic data...")
    # Generate clean signals (e.g., sine waves to simulate phasic responses)
    t = np.linspace(0, 1, signal_len)
    clean_signals = np.sin(2 * np.pi * freq * t).reshape(-1, signal_len)
    clean_signals = np.tile(clean_signals, (num_samples, 1))
    # Add some variability
    clean_signals += np.random.normal(0, 0.05, clean_signals.shape)

    # Generate noise
    noise = np.random.normal(0, 0.2, clean_signals.shape)
    # Add some motion artifacts (spikes)
    for i in range(num_samples):
        spike_pos = np.random.randint(0, signal_len)
        spike_height = np.random.uniform(0.5, 1.5)
        noise[i, spike_pos] *= spike_height * 5

    noisy_signals = clean_signals + noise

    # Normalize data to be between 0 and 1 for the sigmoid output
    noisy_signals = (noisy_signals - np.min(noisy_signals)) / (np.max(noisy_signals) - np.min(noisy_signals))
    clean_signals = (clean_signals - np.min(clean_signals)) / (np.max(clean_signals) - np.min(clean_signals))

    # Reshape for the model (samples, timesteps, features)
    noisy_signals = noisy_signals[..., np.newaxis]
    clean_signals = clean_signals[..., np.newaxis]

    print(f"Data generated. Shapes: Noisy={noisy_signals.shape}, Clean={clean_signals.shape}")
    return noisy_signals, clean_signals

def train_model():
    """
    Main function to train the autoencoder model.
    """
    # --- Parameters ---
    WINDOW_SIZE = 700
    EPOCHS = 20
    BATCH_SIZE = 32
    MODEL_DIR = 'src/models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'autoencoder.h5')

    # --- 1. Data Preparation ---
    # In a real scenario, you would load your data here.
    # X_train should be your noisy signals, and y_train should be the clean ones.
    X_train, y_train = generate_synthetic_data(signal_len=WINDOW_SIZE)

    # --- 2. Build Model ---
    print("Building autoencoder model...")
    input_shape = (WINDOW_SIZE, 1)
    autoencoder = build_autoencoder(input_shape)
    autoencoder.summary()

    # --- 3. Train Model ---
    print("\nStarting model training...")
    autoencoder.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_split=0.2)

    # --- 4. Save Model ---
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    autoencoder.save(MODEL_PATH)
    print(f"\nTraining complete. Model saved to '{MODEL_PATH}'")

if __name__ == '__main__':
    train_model()
