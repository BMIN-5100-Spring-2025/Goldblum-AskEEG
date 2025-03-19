import numpy as np
from langchain.tools import tool
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from app.backend.features.synchrony import synchrony
from app.backend.features.alpha_delta_ratio import alpha_delta_ratio
from app.backend.features.spike_detector import spike_detector


# Define the base Tool class for our UI representation
class Tool(BaseModel):
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="A description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters required by the tool"
    )

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


# Define our available tools for display in the UI
AVAILABLE_TOOLS = [
    Tool(
        name="synchrony_analyzer",
        description="Analyze brain synchrony between EEG channels using the Kuramoto order parameter. Useful for measuring how synchronized different brain regions are.",
        parameters={
            "data": "The EEG data to analyze as a numpy array",
        },
    ),
    Tool(
        name="alpha_delta_ratio",
        description="Calculate the ratio of alpha to delta power. This is useful for awareness analysis.",
        parameters={
            "data": "The EEG data to analyze as a numpy array",
            "fs": "The sampling frequency (Hz) of the data",
        },
    ),
    Tool(
        name="spike_detector",
        description="Detect epileptiform spikes in EEG data. This is useful for seizure monitoring and epilepsy diagnosis.",
        parameters={
            "data": "The EEG data to analyze as a numpy array",
            "fs": "The sampling frequency (Hz) of the data",
        },
    ),
]


def get_available_tools() -> List[Tool]:
    """Return the list of available tools"""
    return AVAILABLE_TOOLS


@tool
def run_synchrony_analyzer(data: np.ndarray, **kwargs) -> tuple[np.ndarray, float]:
    """
    Analyze brain synchrony between EEG channels.

    Args:
        data: np.ndarray - The data to analyze

    Returns:
        r_t: np.ndarray - Kuramoto order parameter for each time point
        R: float - Mean Kuramoto order parameter
    """

    r_t, R = synchrony(data, **kwargs)

    return r_t, R


@tool
def run_alpha_delta_ratio(data: np.ndarray, fs: int, **kwargs) -> np.ndarray:
    """
    Calculate the ratio of alpha to delta power.

    Args:
        data: np.ndarray - The data to analyze
        fs: int - The sampling rate of the data

    Returns:
        ad_ratio: np.ndarray - Alpha/delta ratio for each time point
    """
    ad_ratio = alpha_delta_ratio(data, fs, **kwargs)

    return ad_ratio


@tool
def run_spike_detector(data: np.ndarray, fs: int, **kwargs) -> np.ndarray:
    """
    Detect epileptiform spikes in EEG data.

    Args:
        data: np.ndarray - The data to analyze
        fs: int - The sampling rate of the data

    Returns:
        spikes: np.ndarray - Spike locations (m spikes x (peak index, channel))
    """
    spikes = spike_detector(data, fs, **kwargs)

    return spikes
