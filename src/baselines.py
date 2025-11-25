import numpy as np
from sklearn.decomposition import FastICA, PCA

def apply_ica_denoising(signal, n_components=2):
    """
    Applies Independent Component Analysis (ICA) to denoise a signal.
    This is a simplified example; real-world use requires more sophisticated
    component selection.

    Args:
        signal (np.array): The input signal (1D).
        n_components (int): The number of components to estimate.

    Returns:
        np.array: The reconstructed signal, potentially with artifact
                  components removed.
    """
    # ICA expects multiple signals, so we'll create a dummy second signal.
    # A better approach would be to use multiple sensor channels if available.
    S = np.c_[signal, np.random.randn(len(signal))]
    S /= S.std(axis=0)

    ica = FastICA(n_components=n_components, random_state=42)
    ica_components = ica.fit_transform(S)  # Get the sources

    # In a real scenario, you would identify and zero-out the artifact component.
    # For this example, we'll just assume the first component is the clean one.
    reconstructed_signal = ica.inverse_transform(np.c_[ica_components[:, 0], np.zeros(len(ica_components))])

    return reconstructed_signal[:, 0]
