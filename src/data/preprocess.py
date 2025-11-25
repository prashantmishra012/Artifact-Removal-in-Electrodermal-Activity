import pandas as pd
try:
    import neurokit2 as nk
except ModuleNotFoundError:
    nk = None
    import logging
    logging.warning("Warning: neurokit2 not found. Some preprocessing functions may not work.")

def preprocess_eda(raw_eda_signal, sampling_rate=100):
    """
    Processes raw EDA signal using NeuroKit2 to extract components.

    Args:
        raw_eda_signal (np.array): The raw EDA signal.
        sampling_rate (int): The sampling rate of the signal.

    Returns:
        pd.DataFrame: A DataFrame containing the processed EDA signal and its components.
    """
    if nk is None:
        raise ImportError("neurokit2 is not installed. preprocessing cannot be performed.")
    eda_signals, info = nk.eda_process(raw_eda_signal, sampling_rate=sampling_rate)
    return eda_signals
