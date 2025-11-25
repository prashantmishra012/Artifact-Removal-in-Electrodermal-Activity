import pickle
import pandas as pd

def load_wesad_subject(base_path, subject_id):
    """
    Loads the EDA signal and labels for a single subject from the WESAD dataset.

    Args:
        base_path (str): The path to the root WESAD directory.
        subject_id (str): The subject identifier (e.g., 'S2').

    Returns:
        tuple: A tuple containing:
            - np.array: The raw EDA signal from the chest sensor.
            - np.array: The corresponding labels.
    """
    file_path = f'{base_path}/{subject_id}/{subject_id}.pkl'
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # Extract EDA from the chest sensor at 700Hz
        eda_signal = data['signal']['chest']['EDA'].flatten()
        labels = data['label']

        print(f"Successfully loaded data for subject {subject_id}.")
        return eda_signal, labels

    except FileNotFoundError:
        print(f"Error: Data file not found for subject {subject_id} at {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading data for {subject_id}: {e}")
        return None, None
