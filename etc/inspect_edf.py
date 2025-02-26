import os
import mne


def inspect_edf(edf_path):
    """Read an EDF file and print its attributes."""
    print(f"Reading EDF file: {edf_path}")

    # Load the file (without loading data into memory)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=True)

    # Print basic info
    print("\n=== EDF File Information ===")
    print(f"Filename: {os.path.basename(edf_path)}")
    print(f"File size: {os.path.getsize(edf_path) / (1024*1024):.2f} MB")

    # Print channels information
    print(f"\nChannels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")

    # Print recording information
    print(f"\nSampling rate: {raw.info['sfreq']} Hz")
    print(f"Number of timepoints: {raw.n_times}")
    print(
        f"Recording duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)"
    )

    # Print additional metadata if available
    if raw.info.get("meas_date"):
        print(f"\nRecording date: {raw.info['meas_date']}")

    # Print high/low pass filter settings if available
    print(f"Highpass filter: {raw.info['highpass']} Hz")
    print(f"Lowpass filter: {raw.info['lowpass']} Hz")

    # Print data type and range info
    try:
        # Load a small sample to get data type info
        data, times = raw[:, : int(10 * raw.info["sfreq"])]
        print(f"\nData sample:\n {data}")
        print(f"\nData type: {data.dtype}")
        print(f"Data range (from sample): {data.min():.2f} to {data.max():.2f}")
    except:
        print("\nCouldn't load data sample")

    # Print annotations if present
    if raw.annotations:
        print(f"\nAnnotations: {len(raw.annotations)}")
        if len(raw.annotations) > 0:
            print("Annotation types:", set(raw.annotations.description))
            print("First 5 annotations:", raw.annotations.description[:5])
    else:
        print("\nNo annotations found in file")

    return raw


if __name__ == "__main__":
    # Get input file from environment variable or use default
    input_filename = os.getenv("EDF_FILENAME", "EMU1371_Day02_1_5006_to_5491.edf")
    input_path = os.path.join("data/input", input_filename)

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"ERROR: EDF file not found at {input_path}")
    else:
        inspect_edf(input_path)
