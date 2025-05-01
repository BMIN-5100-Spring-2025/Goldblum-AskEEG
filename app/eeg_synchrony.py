import os
import glob
import mne
import numpy as np

# Set matplotlib to use a non-interactive backend to avoid GUI thread issues
import matplotlib

matplotlib.use("Agg")  # Use the 'Agg' backend which doesn't require a GUI
import matplotlib.pyplot as plt
import logging
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from app.s3_operations import S3Operations
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eeg_synchrony")


def calculate_synchrony(time_series):
    """
    Calculate the Kuramoto order parameter for a set of time series
    Args:
        time_series (np.array): 2D array where each row is a time series
    Returns:
        np.array: Kuramoto order parameter for each time point
    """
    # Extract the number of time series and the number of time points
    N, _ = time_series.shape
    # Apply the Hilbert Transform to get an analytical signal
    analytical_signals = hilbert(time_series)
    # Extract the instantaneous phase for each time series
    phases = np.angle(analytical_signals, deg=False)
    # Compute the Kuramoto order parameter for each time point
    r_t = np.abs(np.sum(np.exp(1j * phases), axis=0)) / N
    R = np.mean(r_t)
    return r_t, R


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the data
    Args:
        data (np.array): 2D array where each row is a time series
        lowcut (float): Low frequency cutoff
        highcut (float): High frequency cutoff
        fs (float): Sampling frequency
        order (int): Filter order
    Returns:
        np.array: Filtered data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data, axis=1)  # Filter along the time axis
    return y


def notch_filter(data, low_cut, high_cut, fs, order=4):
    """
    Apply a notch filter to remove line noise
    Args:
        data (np.array): 2D array where each row is a time series
        low_cut (float): Low frequency cutoff
        high_cut (float): High frequency cutoff
        fs (float): Sampling frequency
        order (int): Filter order
    Returns:
        np.array: Filtered data
    """
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = iirnotch(w0=(low + high) / 2, Q=30, fs=fs)
    y = filtfilt(b, a, data, axis=1)  # Filter along the time axis
    return y


def common_average_montage(ieeg_data):
    """
    Compute the common average montage for iEEG data.
    Parameters:
    - ieeg_data: 2D numpy array
        Rows are channels, columns are data points.
    Returns:
    - cam_data: 2D numpy array
        Data after applying the common average montage.
    """
    # Compute the average across all channels
    avg_signal = np.nanmean(ieeg_data, axis=0)
    result = ieeg_data - avg_signal[np.newaxis, :]
    return result


def find_edf_file(input_directory):
    """Find a single EDF file in the input directory"""
    print(f"Searching for EDF file in: {input_directory}")

    # First try direct search
    edf_files = glob.glob(os.path.join(input_directory, "*.edf"))

    # If no files found, try subdirectories
    if not edf_files:
        edf_files = glob.glob(
            os.path.join(input_directory, "**", "*.edf"), recursive=True
        )

    if edf_files:
        print(f"Found EDF file: {edf_files[0]}")
        return edf_files[0]
    else:
        print("No EDF file found")
        return None


def create_synchrony_plots(results, fs, output_dir, start_time_seconds=0):
    """Create plots of synchrony results

    Args:
        results (dict): Dictionary of analysis results
        fs (float): Sampling frequency in Hz
        output_dir (str): Directory to save the plots
        start_time_seconds (float): Start time of the analyzed segment in seconds
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot mean synchrony by frequency band
    plt.figure(figsize=(10, 6))
    bands = list(results.keys())
    mean_values = [results[band]["mean_synchrony"] for band in bands]

    plt.bar(bands, mean_values)
    plt.ylabel("Mean Synchrony (R)")
    plt.title("Mean Neural Synchrony by Frequency Band")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add text labels on top of each bar
    for i, v in enumerate(mean_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center")

    plt.savefig(
        os.path.join(output_dir, "mean_synchrony_by_band.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create a combined plot of all bands
    plt.figure(figsize=(15, 8))

    # Use start_time_seconds for proper time axis
    samples = len(next(iter(results.values()))["r_t"])
    time_secs = np.linspace(
        start_time_seconds, start_time_seconds + samples / fs, samples
    )

    for band_name in results:
        plt.plot(
            time_secs,
            results[band_name]["r_t"],
            label=f"{band_name} (mean={results[band_name]['mean_synchrony']:.4f})",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Synchrony (R)")
    plt.title("Neural Synchrony Across Frequency Bands")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(output_dir, "all_bands_synchrony.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot time-varying synchrony for each band
    for band in results:
        r_t = results[band]["r_t"]

        # Use start_time_seconds for proper time axis
        time = np.linspace(
            start_time_seconds, start_time_seconds + len(r_t) / fs, len(r_t)
        )

        plt.figure(figsize=(12, 6))
        plt.plot(time, r_t)
        plt.xlabel("Time (s)")
        plt.ylabel("Synchrony (R)")
        plt.title(f"{band.capitalize()} Band Neural Synchrony")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        # Add horizontal line at mean
        plt.axhline(
            results[band]["mean_synchrony"],
            color="r",
            linestyle="--",
            label=f"Mean: {results[band]['mean_synchrony']:.4f}",
        )
        plt.legend()

        plt.savefig(
            os.path.join(output_dir, f"{band}_synchrony.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Generate a spectrogram of synchrony
    plt.figure(figsize=(14, 8))

    # Create a matrix of synchrony values
    band_names = list(results.keys())
    synchrony_matrix = np.vstack([results[band]["r_t"] for band in band_names])

    # Calculate end time for correct x-axis
    duration = len(next(iter(results.values()))["r_t"]) / fs
    end_time = start_time_seconds + duration

    # Plot heatmap with correct time range
    im = plt.imshow(
        synchrony_matrix,
        aspect="auto",
        cmap="viridis",
        extent=[start_time_seconds, end_time, 0, len(band_names)],
        vmin=0,
        vmax=1,
    )

    # Set y-axis to show frequency band names
    plt.yticks(np.arange(len(band_names)) + 0.5, band_names)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency Band")
    plt.title("Neural Synchrony Over Time and Frequency")

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Synchrony (R)")

    plt.savefig(
        os.path.join(output_dir, "synchrony_spectrogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def analyze_synchrony_from_edf(
    input_path, output_dir, freq_bands=None, start_time_seconds=0, end_time_seconds=None
):
    """
    Analyze synchrony from an EDF file
    Args:
        input_path (str): Path to the EDF file
        output_dir (str): Directory to save the results
        freq_bands (dict): Dictionary of frequency bands to analyze
        start_time_seconds (float): Start time in seconds to analyze (default: 0)
        end_time_seconds (float): End time in seconds to analyze (default: full recording)
    Returns:
        results (dict): Dictionary of results
    """
    # Default frequency bands if none provided
    if freq_bands is None:
        freq_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 80),
        }

    # Load the EDF file
    print(f"Loading EDF file: {input_path}")
    raw = mne.io.read_raw_edf(input_path, preload=True, verbose=True)

    # Get channel names and data
    ch_names = raw.ch_names
    fs = raw.info["sfreq"]

    # Apply time cropping if end_time_seconds is specified
    if end_time_seconds is not None:
        raw.crop(start_time_seconds, end_time_seconds)
    elif start_time_seconds > 0:
        raw.crop(tmin=start_time_seconds)

    data = raw.get_data()

    # Print data stats
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    print(f"Raw data range: {data_min} to {data_max}")
    print(f"Raw data mean: {data_mean}")

    # Scale data values to make them usable
    # If values are extremely small, scale them up
    if np.abs(data_max) < 1e-3 and np.abs(data_min) < 1e-3:
        scaling_factor = 1e6  # Scale to microvolts
        data = data * scaling_factor
        print(f"Data values were very small. Scaled by {scaling_factor}")
        print(f"New data range: {np.min(data)} to {np.max(data)}")

    # Apply common average reference
    print("Applying common average reference...")
    car_data = common_average_montage(data)

    # Apply notch filter to remove 60 Hz noise
    print("Applying notch filter for 60 Hz noise...")
    notched_data = notch_filter(car_data, 58, 62, fs)

    # Calculate synchrony for each frequency band
    results = {}
    for band_name, (low_freq, high_freq) in freq_bands.items():
        print(f"Analyzing {band_name} band ({low_freq}-{high_freq} Hz)...")

        # Apply bandpass filter
        filtered_data = bandpass_filter(notched_data, low_freq, high_freq, fs)

        # Calculate synchrony
        r_t, R = calculate_synchrony(filtered_data)

        # Store results
        results[band_name] = {
            "r_t": r_t,
            "mean_synchrony": R,
            "frequency_range": (low_freq, high_freq),
        }

        print(f"{band_name} band mean synchrony: {R:.4f}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save results to file
    results_path = os.path.join(output_dir, "synchrony_results.npz")
    np.savez(
        results_path,
        frequency_bands=freq_bands,
        sampling_rate=fs,
        channel_names=ch_names,
        start_time_seconds=start_time_seconds,
        **{f"{band}_rt": results[band]["r_t"] for band in results},
        **{f"{band}_mean": results[band]["mean_synchrony"] for band in results},
    )

    print(f"Results saved to {results_path}")

    # Create plots
    create_synchrony_plots(results, fs, output_dir, start_time_seconds)

    return results


def process_edf_file():
    """Process a single EDF file from the input directory"""
    # Get directories from environment variables
    input_directory = os.getenv("INPUT_DIR", "./data/input")
    output_directory = os.getenv("OUTPUT_DIR", "./data/output")
    run_mode = os.getenv("RUN_MODE", "local").lower()

    # Get session ID if available
    session_id = os.getenv("SESSION_ID")

    # Print debug information
    logger.info(f"Running in {run_mode} mode")
    if session_id:
        logger.info(f"Session ID: {session_id}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Input directory: {input_directory}")
    logger.info(f"Output directory: {output_directory}")

    # Create directories if they don't exist
    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    # Initialize S3 operations if we're in Fargate mode
    if run_mode == "fargate":
        # Get S3 bucket name from environment
        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError(
                "S3_BUCKET_NAME environment variable must be set when RUN_MODE=fargate"
            )

        logger.info(f"Using S3 bucket: {bucket_name}")
        s3_ops = S3Operations(bucket_name)

        # Download data from S3
        logger.info("Downloading data from S3...")
        s3_ops.download_data("data/input", input_directory)

    # List all directories to help with debugging
    logger.info("Listing directory contents for debugging:")
    try:
        for dirpath, dirnames, filenames in os.walk(input_directory):
            logger.info(f"Directory: {dirpath}")
            for file in filenames:
                logger.info(f"  File: {file}")
    except Exception as e:
        logger.error(f"Error listing directory contents: {e}")

    # Find the EDF file
    edf_file = find_edf_file(input_directory)

    if not edf_file:
        raise FileNotFoundError(f"No EDF file found in {input_directory}")

    # Define frequency bands of interest
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma_low": (30, 50),
        "gamma_high": (50, 80),
    }

    # Process the EDF file
    try:
        logger.info(f"==== Processing {edf_file} ====")
        analyze_synchrony_from_edf(edf_file, output_directory, freq_bands)
        logger.info(f"Completed processing {edf_file}")
    except Exception as e:
        logger.error(f"Error processing {edf_file}: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Upload results to S3 if we're in Fargate mode
    if run_mode == "fargate":
        logger.info("Uploading results to S3...")
        s3_ops.upload_results(output_directory, "data/output")

    logger.info("Finished processing EDF file")


if __name__ == "__main__":
    load_dotenv()
    logging.info("Loaded environment variables from .env file")

    # Run the main processing function
    process_edf_file()
