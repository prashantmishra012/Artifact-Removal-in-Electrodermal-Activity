import matplotlib.pyplot as plt
import numpy as np

def plot_eda_comparison(raw_signal, cleaned_signals, title="EDA Signal Comparison"):
    """
    Plots the raw EDA signal against one or more cleaned versions.

    Args:
        raw_signal (np.array): The original raw signal.
        cleaned_signals (dict): A dictionary where keys are labels (str) and
                                values are the cleaned signals (np.array).
        title (str): The title of the plot.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 6))

    # Set background color to transparent to match Streamlit theme
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Plot raw signal with some transparency
    ax.plot(raw_signal, label='Raw Signal', color='#6c757d', alpha=0.7, linewidth=1)

    for label, signal in cleaned_signals.items():
        ax.plot(signal, label=label, color='#0779e4', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")

    ax.grid(True, linestyle='--', alpha=0.6)
    return fig
