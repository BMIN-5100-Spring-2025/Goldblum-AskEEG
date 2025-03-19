import numpy as np
from app.backend.utils import bandpower


def alpha_delta_ratio(data: np.ndarray, fs: int, **kwargs) -> np.ndarray:
    """Calculate the alpha delta ratio of the data
    Args:
        data (np.ndarray): The data to calculate the alpha delta ratio of
        fs (int): The sampling frequency of the data
    Returns:
        np.ndarray: The alpha delta ratio of the data
    """

    data = data.values
    delta_power = np.nanmean(bandpower(data, fs, [1, 4]))
    alpha_power = np.nanmean(bandpower(data, fs, [8, 13]))

    return alpha_power / delta_power
