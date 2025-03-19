import numpy as np
from scipy.signal import hilbert

def synchrony(data: np.ndarray, **kwargs) -> tuple[np.ndarray, float]:
    """Calculate the Kuramoto order parameter for a set of time series
    Args:
        data (np.array): 2D array where each row is a time series
    Returns:
        np.array: Kuramoto order parameter for each time point
        float: Mean Kuramoto order parameter
    """

    N, _ = data.shape
    analytical_signals = hilbert(data)
    assert analytical_signals.shape == data.shape
    
    phases = np.angle(analytical_signals, deg=False)
    assert phases.shape == data.shape
    
    r_t = np.abs(np.sum(np.exp(1j * phases), axis=0)) / N
    R = np.mean(r_t)

    return r_t, R
