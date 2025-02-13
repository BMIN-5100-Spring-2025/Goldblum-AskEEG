import os
import mne
import yasa
import numpy as np
import pandas as pd
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def load_edf_file(input_path):
    """Load EDF file and preprocess data"""
    raw = mne.io.read_raw_edf(input_path, preload=True)
    raw.resample(100, npad="auto")
    raw.filter(l_freq=0.4, h_freq=30, fir_design="firwin")

    # Select channels and apply reference
    channels = [
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
    raw.pick(channels)
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()

    return raw


def process_staging(raw):
    """Run YASA sleep staging on multiple channels"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sls_c3 = yasa.SleepStaging(raw, eeg_name="C3")
        predicted_c3 = sls_c3.predict()

        sls_cz = yasa.SleepStaging(raw, eeg_name="Cz")
        predicted_cz = sls_cz.predict()

        sls_c4 = yasa.SleepStaging(raw, eeg_name="C4")
        predicted_c4 = sls_c4.predict()

    return predicted_c3, predicted_cz, predicted_c4


def save_results(consensus_stages):
    """Save results to CSV"""
    results_df = pd.DataFrame({"consensus": consensus_stages})
    output_dir = os.path.join("/app/data/output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "yasa_predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Get input file from environment variable
    input_filename = os.getenv("EDF_FILENAME", "EMU1371_Day02_1_5006_to_5491.edf")
    input_path = os.path.join("/app/data/input", input_filename)
    
    # Add file existence check
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"EDF file not found at {input_path}")

    # Process pipeline
    raw = load_edf_file(input_path)
    c3, cz, c4 = process_staging(raw)

    # Generate consensus (same as previous version)
    consensus_stages = []
    for stages in zip(c3, cz, c4):
        stage_counts = Counter(stages)
        if stage_counts.most_common(1)[0][1] >= 2:
            consensus_stages.append(stage_counts.most_common(1)[0][0])
        else:
            consensus_stages.append(np.nan)

    save_results(consensus_stages)
