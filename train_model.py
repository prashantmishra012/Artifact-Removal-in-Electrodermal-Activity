import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys

# Add src to path to import custom modules
sys.path.append('src')
from models.autoencoder import build_autoencoder

# --- 1. Configuration ---
DATA_DIR = 'data/training_data/clean'  # <-- IMPORTANT: Directory with your CLEAN EDA signal CSVs
MODEL_SAVE_PATH = 'src/models/autoencoder_v2.h5' # Path to save the new model
SEGMENT_LENGTH = 700  # Must match the segment length used in the app
VALIDATION_SPLIT = 0.2
EPOCHS = 100  # Increased from 50 for more training
BATCH_SIZE = 32

print("--- Starting Model Training ---")
print(f"TensorFlow Version: {tf.__version__}")

# --- Helper function to generate sample data ---
def generate_sample_data(data_dir, num_files=3, num_samples=50000):
    """Generates sample clean EDA CSV files for testing if none are found."""
    print(f"\nWarning: No CSV files found in '{data_dir}'.")
    print("Generating sample data for demonstration purposes...")
    os.makedirs(data_dir, exist_ok=True)

    for i in range(num_files):
        # Create a plausible, clean-looking EDA signal with tonic and phasic components
        time = np.linspace(0, 100, num_samples)
        tonic = 0.5 * np.sin(time / 10) + 1.5  # Slow-moving tonic component
        phasic = 0.2 * np.exp(-((time - (i * 25 + 20))**2) / 2) # Simulated phasic peak
        phasic += 0.15 * np.exp(-((time - (i * 25 + 50))**2) / 2.5) # Another peak
        clean_signal = tonic + phasic

        df = pd.DataFrame({'Clean_EDA': clean_signal})
        filepath = os.path.join(data_dir, f'sample_clean_eda_{i+1}.csv')
        df.to_csv(filepath, index=False)
        print(f" -> Created sample file: {filepath}")
    print("Sample data generation complete.\n")

# --- Check for data and generate if needed ---
if not os.path.exists(DATA_DIR) or not any(f.endswith('.csv') for f in os.listdir(DATA_DIR)):
    generate_sample_data(DATA_DIR)

# --- 2. Data Loading and Preprocessing ---
def load_and_segment_data(data_dir, segment_length):
    """Loads all CSVs from a directory and segments them."""
    all_segments = []
    if not os.path.exists(data_dir):
        print(f"Error: Training data directory not found at '{data_dir}'")
        print("Please create this directory and add your clean EDA signal CSV files.")
        sys.exit(1)

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading data from: {filepath}")
            df = pd.read_csv(filepath)
            signal = df.iloc[:, 0].to_numpy()

            # Create segments
            num_segments = len(signal) // segment_length
            for i in range(num_segments):
                segment = signal[i * segment_length : (i + 1) * segment_length]
                all_segments.append(segment)

    if not all_segments:
        print(f"Error: No data segments were created. Check if '{data_dir}' contains valid CSV files.")
        sys.exit(1)

    # Reshape for the autoencoder (samples, timesteps, features)
    segments_array = np.array(all_segments)
    segments_array = segments_array.reshape(-1, segment_length, 1)
    return segments_array

print("\n--- Loading and preparing data... ---")
X = load_and_segment_data(DATA_DIR, SEGMENT_LENGTH)
print(f"Successfully created {X.shape[0]} segments of length {X.shape[1]}.")

# The autoencoder learns to reconstruct the input, so X is both the input and the target.
X_train, X_val = train_test_split(X, test_size=VALIDATION_SPLIT, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# --- 3. Model Building and Compilation ---
print("\n--- Building and compiling model... ---")
autoencoder = build_autoencoder(input_shape=(SEGMENT_LENGTH, 1))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# --- 4. Model Training ---
print("\n--- Starting training... ---")
history = autoencoder.fit(
    X_train,
    X_train,  # Input and target are the same for an autoencoder
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val, X_val)
)

print("\n--- Training complete. ---")

# --- 5. Save the Trained Model ---
print(f"Saving new model to: {MODEL_SAVE_PATH}")
autoencoder.save(MODEL_SAVE_PATH)
print("✅ Model saved successfully!")
print("\nTo use the new model, update the 'model_path' in your Streamlit app.")

# --- 6. Plot Training History ---
print("\n--- Plotting training history... ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_title('Model Training History', fontsize=16)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.legend()
ax.grid(True)
plt.savefig('training_history.png')
print("✅ Training history plot saved to 'training_history.png'")
