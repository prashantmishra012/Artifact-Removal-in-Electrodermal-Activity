import pandas as pd
import neurokit2 as nk

def preprocess_eda(raw_eda_signal, sampling_rate=100):
    """
    Processes raw EDA signal using NeuroKit2 to extract components.

    Args:
        raw_eda_signal (np.array): The raw EDA signal.
        sampling_rate (int): The sampling rate of the signal.

    Returns:
        pd.DataFrame: A DataFrame containing the processed EDA signal and its components.
    """
    eda_signals, info = nk.eda_process(raw_eda_signal, sampling_rate=sampling_rate)
    return eda_signals
