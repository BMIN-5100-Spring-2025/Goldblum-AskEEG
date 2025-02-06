import os
import mne
import yasa
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import warnings
from edf_to_postgres import get_db_connection

# Suppress specific scikit-learn warnings about LabelEncoder version
warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator LabelEncoder.*",
    category=UserWarning,
)


def get_eeg_metadata(conn):
    """Fetch EEG metadata from PostgreSQL database"""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT recording_start, original_sfreq, channels
            FROM eeg_metadata
            ORDER BY recording_start DESC
            LIMIT 1
        """
        )
        metadata = cursor.fetchone()
        if not metadata:
            raise ValueError("No metadata found in database")

        return {
            "recording_start": metadata[0],
            "sfreq": metadata[1],
            "channels": metadata[2],
        }


def fetch_eeg_data(conn, metadata):
    """Fetch EEG data from PostgreSQL database and return as MNE RawArray"""
    # Get data from database
    query = "SELECT * FROM eeg_data ORDER BY timestamp"
    df = pd.read_sql(query, conn)

    # Extract channels from metadata
    channels = metadata["channels"]

    # Verify all expected channels exist in the data
    missing_channels = set(channels) - set(df.columns)
    if missing_channels:
        raise ValueError(f"Missing channels in data: {missing_channels}")

    # Select only the channels from metadata and convert to numpy array
    data = df[channels].values.T  # Transpose to (n_channels, n_samples)

    # Create MNE Info object using metadata
    info = mne.create_info(
        ch_names=channels,
        sfreq=metadata["sfreq"],  # Use sampling frequency from metadata
        ch_types="eeg",
    )

    # Create RawArray (convert from μV to volts if needed)
    return mne.io.RawArray(data / 1e6, info)  # Assuming data was stored in μV


# Add the consensus function from YASA_EPC.py
def determine_consensus_stage(predicted_c3, predicted_cz, predicted_c4):
    """Determine the consensus stage based on stages from C3, C4, and Cz"""
    consensus_stage = []
    for i in range(len(predicted_c3)):
        stages = [predicted_c3[i], predicted_cz[i], predicted_c4[i]]
        stage_counts = Counter(stages)
        if (
            stage_counts.most_common(1)[0][1] >= 2
        ):  # Check if the most common stage appears at least twice
            consensus_stage.append(stage_counts.most_common(1)[0][0])
        else:
            consensus_stage.append(np.nan)  # No consensus
    return consensus_stage


def yasa_postgres_pipeline(conn):
    """Main processing pipeline using database data"""
    # Get metadata first
    metadata = get_eeg_metadata(conn)

    # Fetch and prepare data using metadata
    raw = fetch_eeg_data(conn, metadata)

    # Match processing from YASA_EPC.py
    raw.resample(100, npad="auto")
    raw.filter(l_freq=0.4, h_freq=30, fir_design="firwin")

    # Apply common average reference
    channels_to_include = [
        "C3",
        "C4",
        "Cz",
        "F3",
        "F4",
        "F7",
        "F8",
        "Fp1",
        "Fp2",
        "Fz",
        "O1",
        "O2",
        "P3",
        "P4",
        "T3",
        "T4",
        "T5",
        "T6",
    ]
    raw.pick(channels_to_include)

    # Apply reference and projections in one step to avoid warning
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()  # Apply projections immediately

    # Sleep staging (same as YASA_EPC.py)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warnings during YASA processing
        sls_c3 = yasa.SleepStaging(raw, eeg_name="C3")
        predicted_c3 = sls_c3.predict()

        sls_cz = yasa.SleepStaging(raw, eeg_name="Cz")
        predicted_cz = sls_cz.predict()

        sls_c4 = yasa.SleepStaging(raw, eeg_name="C4")
        predicted_c4 = sls_c4.predict()

    # Calculate consensus stages
    consensus_stages = determine_consensus_stage(
        predicted_c3, predicted_cz, predicted_c4
    )

    return {
        "C3": predicted_c3,
        "Cz": predicted_cz,
        "C4": predicted_c4,
        "consensus": consensus_stages,
    }


if __name__ == "__main__":
    conn = get_db_connection()
    try:
        predictions = yasa_postgres_pipeline(conn)
        print("YASA predictions from PostgreSQL data:")
        print("C3:", predictions["C3"])
        print("Cz:", predictions["Cz"])
        print("C4:", predictions["C4"])
        print("Consensus:", predictions["consensus"])
    finally:
        conn.close()
