from flask import Blueprint, jsonify, request, current_app
import logging
from app.backend.services.pennsieve_metadata import PennsieveMetadataService
from app.backend.services.data_retrieval import DataRetrievalService
import os
from functools import wraps
import json
import datetime
import pandas as pd
import numpy as np

# Create a logger
logger = logging.getLogger("askeeg_routes")

# Create a blueprint for our API
api_bp = Blueprint("api", __name__, url_prefix="/api")

# Services
pennsieve_service = None
data_retrieval_service = None


# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check if we're in development mode
        dev_mode = (
            os.getenv("FLASK_ENV") == "development" or os.getenv("DEBUG") == "True"
        )

        token = None
        auth_header = request.headers.get("Authorization")

        # Check if Authorization header exists and is properly formatted
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            logger.info(f"Auth token received: {token[:10]}...")
        else:
            logger.warning(f"Missing or malformed Authorization header: {auth_header}")

        # In development mode, bypass token requirement
        if dev_mode:
            logger.info("Development mode: bypassing token verification")
            return f(*args, **kwargs)

        if not token:
            logger.warning("Authentication token is missing")
            return jsonify({"message": "Authentication token is missing"}), 403

        try:
            # Verify the JWT token using the same variables as the frontend
            cognito_user_pool_id = os.getenv("COGNITO_USER_POOL_ID")

            logger.info(f"Verifying token with user pool: {cognito_user_pool_id}")

            # For now, just pass through since we're setting up the authentication flow
            # In production, you would properly verify the token with AWS Cognito
            logger.info("Token verification bypassed for development")

        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return jsonify({"message": "Invalid authentication token"}), 403

        return f(*args, **kwargs)

    return decorated


# Replace the deprecated decorator with a function to register with the app
def init_services():
    """Initialize services when app starts"""
    global pennsieve_service, data_retrieval_service

    try:
        # Initialize services
        pennsieve_service = PennsieveMetadataService()
        data_retrieval_service = DataRetrievalService(pennsieve_service)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")


# Function to register the initialization with Flask app
def register_services_init(app):
    """Register the service initialization with the Flask app"""
    app.before_first_request(init_services)


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "AskEEG API is running"}), 200


@api_bp.route("/metadata", methods=["GET"])
@token_required
def get_metadata():
    """
    Get metadata for a Pennsieve dataset/package

    Query Parameters:
    - dataset_id: Pennsieve dataset ID (optional if set in environment)
    - package_id: Pennsieve package ID (optional if set in environment)
    """
    try:
        # Initialize services if not already done
        if pennsieve_service is None:
            logger.info("Initializing services on-demand")
            init_services()
        else:
            logger.info("Using existing pennsieve_service")

        # Get query parameters
        dataset_id = request.args.get("dataset_id")
        package_id = request.args.get("package_id")

        logger.info(
            f"Fetching metadata for dataset_id={dataset_id}, package_id={package_id}"
        )

        # Log environment variables
        logger.info(f"PENNSIEVE_DATASET env: {os.getenv('PENNSIEVE_DATASET')}")
        logger.info(f"PENNSIEVE_PACKAGE env: {os.getenv('PENNSIEVE_PACKAGE')}")
        logger.info(
            f"PENNSIEVE_API_TOKEN env: {'Set' if os.getenv('PENNSIEVE_API_TOKEN') else 'Not set'}"
        )

        # Get metadata
        metadata = pennsieve_service.get_metadata(dataset_id, package_id)

        logger.info(
            f"Successfully retrieved metadata with {len(metadata.get('channels', []))} channels"
        )
        return jsonify(metadata), 200
    except Exception as e:
        logger.error(f"Error getting metadata: {e}", exc_info=True)
        return jsonify({"error": str(e), "message": "Failed to retrieve metadata"}), 500


@api_bp.route("/channels", methods=["GET"])
def get_channels():
    """
    Get channel information for a Pennsieve package

    Query Parameters:
    - dataset_id: Pennsieve dataset ID (optional if set in environment)
    - package_id: Pennsieve package ID (optional if set in environment)
    """
    try:
        # Initialize services if not already done
        if pennsieve_service is None:
            init_services()

        # Get query parameters
        dataset_id = request.args.get("dataset_id")
        package_id = request.args.get("package_id")

        # Get metadata
        metadata = pennsieve_service.get_metadata(dataset_id, package_id)

        # Return just the channels
        return (
            jsonify(
                {
                    "channels": metadata.get("channels", []),
                    "total_channels": len(metadata.get("channels", [])),
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        return jsonify({"error": str(e), "message": "Failed to retrieve channels"}), 500


@api_bp.route("/data", methods=["GET"])
def get_data():
    """
    Get time-series data for specific channels and time range

    Query Parameters:
    - dataset_id: Pennsieve dataset ID (optional if set in environment)
    - package_id: Pennsieve package ID (optional if set in environment)
    - channel_ids: Comma-separated list of channel IDs (optional)
    - start_time: Start time in microseconds (optional)
    - end_time: End time in microseconds (optional)
    - force_refresh: Whether to force refresh cached data (default: false)
    - is_relative_time: Whether time values are relative (default: false)
    """
    try:
        # Initialize services if not already done
        if data_retrieval_service is None:
            init_services()

        # Get query parameters
        dataset_id = request.args.get("dataset_id")
        package_id = request.args.get("package_id")

        # Parse channel IDs
        channel_ids = request.args.get("channel_ids")
        if channel_ids:
            channel_ids = channel_ids.split(",")

        # Parse time range
        start_time = request.args.get("start_time")
        if start_time:
            start_time = int(start_time)

        end_time = request.args.get("end_time")
        if end_time:
            end_time = int(end_time)

        # Parse flags
        force_refresh = request.args.get("force_refresh", "false").lower() == "true"
        is_relative_time = (
            request.args.get("is_relative_time", "false").lower() == "true"
        )

        # Get data
        data = data_retrieval_service.get_data_range(
            dataset_id,
            package_id,
            channel_ids,
            start_time,
            end_time,
            force_refresh,
            is_relative_time,
        )

        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        return jsonify({"error": str(e), "message": "Failed to retrieve data"}), 500


@api_bp.route("/query", methods=["POST"])
def process_query():
    """
    Process a natural language query for EEG analysis

    Request Body:
    - query: Natural language query string
    - dataset_id: Pennsieve dataset ID (optional if set in environment)
    - package_id: Pennsieve package ID (optional if set in environment)
    """
    try:
        # Initialize services if not already done
        if pennsieve_service is None:
            init_services()

        # Get request data
        data = request.get_json()

        if not data or "query" not in data:
            return (
                jsonify(
                    {
                        "error": "Missing required parameter: query",
                        "message": "Please provide a natural language query",
                    }
                ),
                400,
            )

        # Extract parameters
        query = data["query"]
        dataset_id = data.get("dataset_id")
        package_id = data.get("package_id")

        # Process query (placeholder until NLP component is implemented)
        result = {
            "query": query,
            "status": "received",
            "message": "Query received and will be processed",
            "job_id": "placeholder_job_id",  # Will be replaced with actual job ID system
        }

        return jsonify(result), 202
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e), "message": "Failed to process query"}), 500


@api_bp.route("/results/<job_id>", methods=["GET"])
def get_results(job_id):
    """
    Get results for a processed query

    Path Parameters:
    - job_id: ID of the processing job
    """
    try:
        # Initialize services if not already done
        if pennsieve_service is None:
            init_services()

        # Placeholder until job system is implemented
        result = {
            "job_id": job_id,
            "status": "pending",
            "message": "Results not yet available",
        }

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        return jsonify({"error": str(e), "message": "Failed to retrieve results"}), 500


@api_bp.route("/retrieve-data-segment", methods=["POST"])
@token_required
def retrieve_data_segment():
    """
    Retrieve a specified data segment from Pennsieve and save it to the local filesystem.

    Expects JSON with:
    - dataset_id: Pennsieve dataset ID (optional if set in environment)
    - package_id: Pennsieve package ID (optional if set in environment)
    - channel_ids: List of channel IDs to include
    - start_time: Start time in microseconds (absolute time)
    - end_time: End time in microseconds (absolute time)
    - start_time_seconds: Start time in seconds (relative to recording start)
    - end_time_seconds: End time in seconds (relative to recording start)
    - output_filename: Name for the output file (without extension)

    Returns:
    - JSON response with status and output file path
    """
    # Initialize services if needed - declare global variables first
    global pennsieve_service, data_retrieval_service

    logger.info("Entered retrieve_data_segment endpoint")

    try:
        request_data = request.get_json()
        logger.info(f"Request data: {request_data}")

        if not request_data:
            logger.warning("No data provided in request")
            return jsonify({"error": "No data provided"}), 400

        # Get parameters from request
        dataset_id = request_data.get("dataset_id") or os.getenv("PENNSIEVE_DATASET")
        package_id = request_data.get("package_id") or os.getenv("PENNSIEVE_PACKAGE")
        channel_ids = request_data.get("channel_ids", [])
        output_filename = request_data.get("output_filename", "eeg_segment")

        logger.info(
            f"Processing request - dataset: {dataset_id}, package: {package_id}, channels: {len(channel_ids)}"
        )

        # Get time parameters - accept either absolute microseconds or relative seconds
        start_time = request_data.get("start_time")
        end_time = request_data.get("end_time")
        start_time_seconds = request_data.get("start_time_seconds")
        end_time_seconds = request_data.get("end_time_seconds")
        is_relative_time = request_data.get(
            "is_relative_time", True
        )  # Default to relative time

        logger.info(
            f"Time parameters - start_time: {start_time}, end_time: {end_time}, "
            f"start_time_seconds: {start_time_seconds}, end_time_seconds: {end_time_seconds}, "
            f"is_relative_time: {is_relative_time}"
        )

        # When using relative time, prioritize seconds values
        if is_relative_time:
            # If we're using relative time, we want to work with seconds directly
            # (not microseconds as previously assumed)
            if start_time_seconds is not None:
                start_time = start_time_seconds
                logger.info(f"Using relative start_time_seconds: {start_time}")

            if end_time_seconds is not None:
                end_time = end_time_seconds
                logger.info(f"Using relative end_time_seconds: {end_time}")
        else:
            # Only convert seconds to microseconds if absolute times are not provided
            if start_time is None and start_time_seconds is not None:
                start_time = int(start_time_seconds * 1000000)
                logger.info(
                    f"Converted start_time_seconds to microseconds: {start_time}"
                )

            if end_time is None and end_time_seconds is not None:
                end_time = int(end_time_seconds * 1000000)
                logger.info(f"Converted end_time_seconds to microseconds: {end_time}")

        # Log the final time values we'll use
        logger.info(
            f"Final time values - start_time: {start_time}, end_time: {end_time}, is_relative_time: {is_relative_time}"
        )

        # Validate required parameters
        if not dataset_id:
            logger.warning("Dataset ID is required but not provided")
            return jsonify({"error": "Dataset ID is required"}), 400
        if not package_id:
            logger.warning("Package ID is required but not provided")
            return jsonify({"error": "Package ID is required"}), 400
        if not channel_ids:
            logger.warning("No channel IDs provided")
            return jsonify({"error": "At least one channel ID is required"}), 400
        if start_time is None:
            logger.warning("Start time is required but not provided")
            return jsonify({"error": "Start time is required"}), 400
        if end_time is None:
            logger.warning("End time is required but not provided")
            return jsonify({"error": "End time is required"}), 400

        # Ensure start_time is before end_time
        if start_time >= end_time:
            logger.warning(
                f"Start time {start_time} must be before end time {end_time}"
            )
            return jsonify({"error": "Start time must be before end time"}), 400

        if pennsieve_service is None:
            logger.info("Initializing services")
            init_services()

        # Retrieve the data from Pennsieve
        logger.info(
            f"Retrieving data segment from Pennsieve: dataset={dataset_id}, "
            + f"package={package_id}, channels={len(channel_ids)}, "
            + f"start={start_time}, end={end_time}"
        )

        try:
            # Get data from Pennsieve
            logger.info("Calling data_retrieval_service.get_data_range")
            data = data_retrieval_service.get_data_range(
                dataset_id=dataset_id,
                package_id=package_id,
                channel_ids=channel_ids,
                start_time=start_time,
                end_time=end_time,
                force_refresh=False,
                is_relative_time=is_relative_time,  # Pass through the relative time flag
            )
            logger.info(
                f"Data retrieved successfully: {len(data.get('channels', {}))} channels, {data.get('samples', 0)} samples"
            )
        except Exception as e:
            logger.error(f"Error retrieving data from Pennsieve: {e}", exc_info=True)
            return jsonify({"error": f"Failed to retrieve data: {str(e)}"}), 500

        # Check if data is empty
        if data.get("is_empty", True):
            logger.warning("No data found for the specified parameters")
            return jsonify({"error": "No data found for the specified parameters"}), 404

        try:
            # Create the base data/input directory if it doesn't exist
            base_input_dir = os.path.join(os.getcwd(), "data", "input")
            os.makedirs(base_input_dir, exist_ok=True)
            logger.info(f"Created/verified base input directory: {base_input_dir}")

            # Generate a timestamp for the folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create a timestamped folder within the data/input directory
            timestamped_dir = os.path.join(base_input_dir, timestamp)
            os.makedirs(timestamped_dir, exist_ok=True)
            logger.info(f"Created timestamped directory: {timestamped_dir}")

            # Create the output file path
            output_csv = f"{output_filename}.csv"
            output_path = os.path.join(timestamped_dir, output_csv)
            logger.info(f"Output file path: {output_path}")

            # Convert the data to a pandas DataFrame for saving to CSV
            times = data.get("times", [])  # Times in milliseconds
            channel_data = data.get("channels", {})
            logger.info(
                f"Processing data: {len(times)} timestamps, {len(channel_data)} channels"
            )

            # Get channel metadata for the header
            channel_info = data_retrieval_service.get_channel_info(
                dataset_id, package_id
            )
            channel_names = {
                channel["id"]: channel["name"]
                for channel in channel_info
                if channel["id"] in channel_ids
            }
            logger.info(f"Retrieved channel names: {channel_names}")

            # Also create reverse mapping for lookup by part of the ID
            channel_id_to_name = {}
            for channel_id, name in channel_names.items():
                # Store the full ID
                channel_id_to_name[channel_id] = name
                # Also store just the UUID part if it has a format like 'N:channel:UUID'
                if ":" in channel_id:
                    uuid_part = channel_id.split(":")[-1]
                    channel_id_to_name[uuid_part] = name

            logger.info(f"Channel ID to name mapping: {channel_id_to_name}")

            # Check if times list is empty
            if not times:
                logger.warning("No time points in the retrieved data")
                return jsonify({"error": "Retrieved data has no time points"}), 500

            # Create a DataFrame from the data returned by Pennsieve
            try:
                # Get the raw dataframe directly from the data retrieval service
                raw_df = data.get("raw_dataframe")

                if raw_df is not None and not raw_df.empty:
                    # Use the raw dataframe directly
                    df = raw_df
                    logger.info(f"Using raw dataframe with {len(df)} rows")
                else:
                    # Fallback to the previous approach if raw_dataframe is not available
                    df = pd.DataFrame(index=pd.to_datetime(np.array(times), unit="ms"))
                    logger.info(f"Created new dataframe with {len(df)} rows")

                # If df doesn't have data columns yet, add them
                if len(df.columns) == 0:
                    # Add each channel as a column
                    for channel_id, values in channel_data.items():
                        channel_name = channel_names.get(
                            channel_id, f"Channel {channel_id}"
                        )
                        df[channel_name] = values
                        logger.info(f"Added channel {channel_name} to DataFrame")
                else:
                    # Rename existing columns to use channel names
                    column_mapping = {}
                    for col in df.columns:
                        if col in channel_id_to_name:
                            column_mapping[col] = channel_id_to_name[col]

                    if column_mapping:
                        df.rename(columns=column_mapping, inplace=True)
                        logger.info(
                            f"Renamed columns using channel names: {list(column_mapping.values())}"
                        )
                    else:
                        logger.warning(
                            f"No column mapping created. Column names: {list(df.columns)}"
                        )
                        # Use original channel names if we can't map them
                        logger.info("Falling back to using original channel names")

                # Save to CSV
                logger.info(f"Saving DataFrame to CSV: {output_path}")
                # Use index=True to include the timestamp index as the first column
                df.to_csv(output_path, index=True, index_label="timestamp")
                logger.info("CSV file saved successfully")

            except Exception as e:
                logger.error(
                    f"Error creating DataFrame or saving to CSV: {e}", exc_info=True
                )
                return jsonify({"error": f"Error processing data: {str(e)}"}), 500

            # Create metadata file with information about the data segment
            try:
                metadata_filename = f"{output_filename}.meta.json"
                metadata_path = os.path.join(timestamped_dir, metadata_filename)
                logger.info(f"Creating metadata file: {metadata_path}")

                # Get package metadata
                pkg_metadata = pennsieve_service.get_metadata(dataset_id, package_id)

                # Build segment metadata
                segment_metadata = {
                    "dataset_id": dataset_id,
                    "package_id": package_id,
                    "channel_ids": channel_names,  # Map of channel IDs to channel names
                    "requested_start_time_seconds": start_time_seconds,
                    "requested_end_time_seconds": end_time_seconds,
                    "requested_duration_seconds": (
                        end_time_seconds - start_time_seconds
                        if end_time_seconds is not None
                        and start_time_seconds is not None
                        else None
                    ),
                    "is_relative_time": is_relative_time,
                    "absolute_start_time_ns_epoch": pkg_metadata.get(
                        "time_range", {}
                    ).get("start"),
                    "absolute_end_time_ns_epoch": pkg_metadata.get(
                        "time_range", {}
                    ).get("end"),
                    "retrieved_duration_seconds": int(
                        data.get("duration_seconds") / 1e6
                    ),
                    "sampling_rate": pkg_metadata.get("time_range", {}).get(
                        "sampling_rate"
                    ),
                    "samples": data.get("samples", 0),
                    "retrieved_at": datetime.datetime.now().isoformat(),
                }

                with open(metadata_path, "w") as f:
                    json.dump(segment_metadata, f, indent=2)
                logger.info("Metadata file created successfully")

            except Exception as e:
                logger.error(f"Error creating metadata file: {e}", exc_info=True)
                # Continue even if metadata file creation fails

            # Get file size
            try:
                file_size = os.path.getsize(output_path)
                logger.info(f"File size: {file_size} bytes")
            except Exception as e:
                logger.error(f"Error getting file size: {e}")
                file_size = 0

            # Return success response with file path
            relative_path = os.path.join("data", "input", timestamp, output_csv)
            response_data = {
                "success": True,
                "output_path": relative_path,
                "timestamp_folder": timestamp,
                "samples": data.get("samples", 0),
                "duration_seconds": data.get(
                    "duration_seconds", end_time_seconds - start_time_seconds
                ),
                "file_size_bytes": file_size,
            }
            logger.info(f"Returning success response: {response_data}")
            return jsonify(response_data), 200

        except Exception as e:
            logger.error(f"Error processing data for saving: {e}", exc_info=True)
            return jsonify({"error": f"Error saving data: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Error processing data segment retrieval: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500


@api_bp.route("/data-files", methods=["GET"])
@token_required
def get_data_files():
    """
    List data files available in the /data/input directory
    """
    try:
        # Base directory for input files - use a relative path
        base_dir = os.path.join(os.getcwd(), "data", "input")

        # Ensure the directory exists
        if not os.path.exists(base_dir):
            return jsonify({"error": "Input directory does not exist"}), 404

        files = []

        # Walk through the input directory to find all CSV files
        for root, dirs, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.endswith(".csv"):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, start=base_dir)

                    # Create a readable display name
                    display_name = rel_path

                    # Add to the list
                    files.append({"path": file_path, "displayName": display_name})

        return jsonify({"files": files}), 200
    except Exception as e:
        logger.error(f"Error listing data files: {e}", exc_info=True)
        return jsonify({"error": str(e), "message": "Failed to list data files"}), 500


@api_bp.route("/run-analysis", methods=["POST"])
@token_required
def run_analysis():
    """
    Run analysis on a data file

    Request Body:
    - input_file: Path to the input file
    - analysis_type: Type of analysis to run (currently supports 'eeg_synchrony')
    - output_name: Name for the output files
    - selected_bands: List of frequency bands to analyze (for 'eeg_synchrony')
    """
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        input_file = data.get("input_file")
        analysis_type = data.get("analysis_type")
        output_name = data.get("output_name", "analysis_result")

        # Validate required fields
        if not input_file:
            return jsonify({"error": "Input file path is required"}), 400
        if not analysis_type:
            return jsonify({"error": "Analysis type is required"}), 400

        # Ensure input file exists
        if not os.path.exists(input_file):
            return jsonify({"error": f"Input file {input_file} does not exist"}), 404

        # Create output directory in data/output with relative path
        base_output_dir = os.path.join(os.getcwd(), "data", "output")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"{output_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Handle different analysis types
        if analysis_type == "eeg_synchrony":
            # Get EEG synchrony specific parameters
            selected_bands = data.get("selected_bands", [])

            if not selected_bands:
                return (
                    jsonify({"error": "No frequency bands selected for analysis"}),
                    400,
                )

            # Define frequency bands based on selection
            freq_bands = {}
            all_bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 80),
            }

            for band in selected_bands:
                if band in all_bands:
                    freq_bands[band] = all_bands[band]

            # Run EEG synchrony analysis
            # First, load the CSV file into a pandas DataFrame
            try:
                df = pd.read_csv(input_file)

                # Extract the sample rate from metadata file instead of using a hardcoded value
                metadata_file = None

                # Determine the metadata file path (we expect it to be in the same directory as input_file)
                input_dir = os.path.dirname(input_file)
                input_filename = os.path.basename(input_file)
                metadata_filename = input_filename.replace(".csv", ".meta.json")
                metadata_path = os.path.join(input_dir, metadata_filename)

                # Try to read the sampling rate from the metadata file
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            if metadata.get("sampling_rate"):
                                sample_rate = float(metadata.get("sampling_rate"))
                                logger.info(
                                    f"Using sampling rate from metadata file: {sample_rate} Hz"
                                )
                            else:
                                logger.warning(
                                    f"Metadata file found but no sampling_rate field: {metadata_path}"
                                )
                    except Exception as e:
                        logger.error(f"Error reading metadata file: {e}")
                else:
                    logger.warning(
                        f"Metadata file not found: {metadata_path}, using default sampling rate: {sample_rate} Hz"
                    )

                # Convert DataFrame to numpy array for analysis
                data_array = df.iloc[
                    :, 1:
                ].values.T  # Transpose to get channels as rows

                # Import eeg_synchrony module
                from app.eeg_synchrony import (
                    calculate_synchrony,
                    bandpass_filter,
                )

                # Initialize results dictionary
                results = {}

                # Process each frequency band
                for band_name, (low_freq, high_freq) in freq_bands.items():
                    # Apply bandpass filter to isolate frequency band
                    filtered_data = bandpass_filter(
                        data_array, low_freq, high_freq, sample_rate
                    )

                    # Calculate synchrony
                    r_t, R = calculate_synchrony(filtered_data)

                    # Store results
                    results[band_name] = {
                        "r_t": r_t.tolist(),  # Convert numpy array to list for JSON serialization
                        "mean_synchrony": float(
                            R
                        ),  # Ensure it's a Python float for JSON
                    }

                # Generate plots
                from app.eeg_synchrony import create_synchrony_plots

                create_synchrony_plots(results, sample_rate, output_dir)

                # Create a dictionary to return to the frontend
                response_data = {
                    "message": "Analysis completed successfully",
                    "timestamp": timestamp,
                    "output_dir": output_dir,
                    "synchrony": {
                        **{
                            f"{band}_mean": results[band]["mean_synchrony"]
                            for band in results
                        }
                    },
                    "images": [],
                }

                # Get static URL base based on the current app's configuration
                static_url_base = "/static/output"

                # Add generated images to the response
                for file in os.listdir(output_dir):
                    if file.endswith(".png"):
                        # Create a relative path for the image URL
                        relative_output_dir = f"{output_name}_{timestamp}"
                        response_data["images"].append(
                            {
                                "name": file,
                                "url": f"{static_url_base}/{relative_output_dir}/{file}",
                            }
                        )

                return jsonify(response_data), 200

            except Exception as analysis_error:
                logger.error(
                    f"Error during synchrony analysis: {analysis_error}", exc_info=True
                )
                return (
                    jsonify(
                        {
                            "error": str(analysis_error),
                            "message": "Failed to analyze data",
                        }
                    ),
                    500,
                )

        else:
            return (
                jsonify({"error": f"Unsupported analysis type: {analysis_type}"}),
                400,
            )

    except Exception as e:
        logger.error(f"Error running analysis: {e}", exc_info=True)
        return jsonify({"error": str(e), "message": "Failed to run analysis"}), 500
