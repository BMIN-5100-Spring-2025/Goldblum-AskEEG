import os
import sys
import dotenv
import numpy as np
import pandas as pd
from ieeg.auth import Session  # >_ cd ieegy --> pip install -e .
import argparse


def main(total_mins, remove_nan=False):
    dotenv.load_dotenv()

    session = Session(os.getenv("IEEG_USERNAME"), os.getenv("IEEG_PASSWORD"))
    dataset = session.open_dataset(os.getenv("IEEG_DATASET"))

    # Get the channel labels
    channel_labels = dataset.get_channel_labels()
    num_channels = len(channel_labels)

    # Get the temporal details of the first channel
    timeseries = dataset.get_time_series_details(channel_labels[0])
    fs = int(timeseries.sample_rate)  # Hz

    print(f"Sampling rate: {fs} Hz")

    # Calculate chunk size that's valid for the sampling rate
    max_chunk_mins = 10
    samples_per_min = fs * 60
    max_samples = max_chunk_mins * samples_per_min
    valid_samples = max_samples - (
        max_samples % fs
    )  # Round down to nearest multiple of fs
    chunk_mins = valid_samples / samples_per_min
    chunk_usec = chunk_mins * 60 * 1e6  # Convert to microseconds

    # Initialize empty list to store chunks
    dfs = []

    # Calculate number of chunks needed
    num_chunks = int(np.ceil(total_mins / chunk_mins))

    # Loop through chunks
    for i in range(num_chunks):
        start_usec = i * chunk_usec
        end_min = min((i + 1) * chunk_mins, total_mins)
        print(
            f"Collecting chunk {i+1}/{num_chunks}: Minutes {i*chunk_mins:.2f} to {end_min:.2f}"
        )

        # Get data chunk
        chunk_df = dataset.get_dataframe(
            start_usec, chunk_usec, np.arange(num_channels)
        )
        dfs.append(chunk_df)

    # Concatenate all chunks
    full_df = pd.concat(dfs, axis=0)
    print(f"Dataset shape: {full_df.shape}")

    # Drop NaN rows if specified
    if remove_nan:
        full_df = full_df.dropna()
        print(f"Dataset shape after NaN removal: {full_df.shape}")

    # Save dataset to CSV
    data_filename = "data_no_nan.csv" if remove_nan else "data.csv"

    # Create path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(main_dir, "data", "input")

    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)

    data_filepath = os.path.join(data_dir, data_filename)

    # Save without index column
    full_df.to_csv(data_filepath, index=False)
    print(f"Data saved to {data_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull data from IEEG dataset")
    parser.add_argument("minutes", type=int, help="Number of minutes to pull")
    parser.add_argument(
        "--no-nan",
        action="store_true",
        help="Remove NaN values from the dataset (default: keep NaN)",
    )

    args = parser.parse_args()

    if args.minutes <= 0:
        print("Error: Number of minutes must be positive")
        sys.exit(1)

    main(args.minutes, remove_nan=args.no_nan)
