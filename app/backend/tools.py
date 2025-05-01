import numpy as np
from langchain.tools import tool
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging

# Import EEG functions
from app.eeg_synchrony import (
    calculate_synchrony,
    bandpass_filter,
    common_average_montage,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eeg_tools")


# Define the base Tool class for UI representation
class Tool(BaseModel):
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="A description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters required by the tool"
    )

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


# Define available tools for the LLM
AVAILABLE_TOOLS = [
    Tool(
        name="synchrony_analyzer",
        description="Analyze brain synchrony between EEG channels using the Kuramoto order parameter. Useful for measuring how synchronized different brain regions are.",
        parameters={
            "data": "The EEG data to analyze as a numpy array",
            "time_window": "The time window to analyze, specified as [start_time, end_time] in seconds",
            "frequency_band": "The frequency band to analyze (delta, theta, alpha, beta, gamma)",
        },
    ),
    Tool(
        name="extract_time_window",
        description="Extract a time window from the EEG data based on start and end times.",
        parameters={
            "start_time": "Start time in seconds",
            "end_time": "End time in seconds",
            "channel_ids": "List of channel IDs to include (optional)",
        },
    ),
    Tool(
        name="filter_frequency_band",
        description="Filter EEG data to isolate a specific frequency band.",
        parameters={
            "data": "The EEG data to filter as a numpy array",
            "band": "The frequency band to isolate (delta, theta, alpha, beta, gamma)",
            "fs": "The sampling frequency (Hz) of the data",
        },
    ),
]


def get_available_tools() -> List[Tool]:
    """Return the list of available tools"""
    return AVAILABLE_TOOLS


@tool
def extract_time_window(
    data: np.ndarray,
    fs: float,
    start_time: float,
    end_time: float,
    channel_ids: List[int] = None,
) -> np.ndarray:
    """
    Extract a segment of EEG data based on start and end times.

    Args:
        data: np.ndarray - The full EEG data array (channels x time)
        fs: float - The sampling frequency (Hz)
        start_time: float - Start time in seconds
        end_time: float - End time in seconds
        channel_ids: List[int] - List of channel indices to include (optional)

    Returns:
        np.ndarray - The extracted data segment
    """
    # Convert seconds to samples
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    # Extract time window
    if channel_ids is not None:
        return data[channel_ids, start_sample:end_sample]
    return data[:, start_sample:end_sample]


@tool
def filter_frequency_band(data: np.ndarray, band: str, fs: float) -> np.ndarray:
    """
    Filter EEG data to isolate a specific frequency band.

    Args:
        data: np.ndarray - The EEG data to filter
        band: str - The frequency band to isolate (delta, theta, alpha, beta, gamma)
        fs: float - The sampling frequency (Hz)

    Returns:
        np.ndarray - The filtered EEG data
    """
    # Define frequency ranges for common bands
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100),
    }

    if band.lower() not in bands:
        raise ValueError(f"Invalid band: {band}. Must be one of {list(bands.keys())}")

    # Get the frequency range for the specified band
    low_freq, high_freq = bands[band.lower()]

    # Apply bandpass filter
    return bandpass_filter(data, low_freq, high_freq, fs)


@tool
def run_synchrony_analysis(
    data: np.ndarray, fs: float, band: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze brain synchrony between EEG channels.

    Args:
        data: np.ndarray - The EEG data to analyze (channels x time)
        fs: float - The sampling frequency (Hz)
        band: str - The frequency band to analyze (optional)

    Returns:
        dict - Results containing synchrony metrics
    """
    # If a specific band is requested, filter the data
    if band:
        data = filter_frequency_band(data, band, fs)

    # Apply common average reference to reduce volume conduction effects
    data = common_average_montage(data)

    # Calculate synchrony
    r_t, R = calculate_synchrony(data)

    # Return results
    return {
        "r_t": r_t.tolist(),  # Convert to list for JSON serialization
        "mean_synchrony": float(R),
        "band": band if band else "broadband",
        "n_channels": data.shape[0],
        "n_samples": data.shape[1],
        "duration_seconds": data.shape[1] / fs,
    }
