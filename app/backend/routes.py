from flask import Blueprint, jsonify, request
import logging
from app.backend.services.pennsieve_metadata import PennsieveMetadataService
from app.backend.services.data_retrieval import DataRetrievalService
import os
from functools import wraps
import json
import datetime
import pandas as pd
import numpy as np
from app.backend.process_query import llm_analyze_query
import time

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

        # Get the timestamp from the input file path if it follows the expected structure
        # Format: /data/input/<timestamp>/filename.csv
        input_dir = os.path.dirname(input_file)
        if "/input/" in input_dir:
            path_parts = input_dir.split("/")
            for part in path_parts:
                if (
                    len(part) == 15 and part.count("_") == 1
                ):  # Looks like a timestamp: YYYYMMDD_HHMMSS
                    timestamp = part
                    logger.info(f"Using timestamp from input directory: {timestamp}")
                    break

        # Create output directory using timestamp
        output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

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
                # Try both with and without index
                try:
                    # First try reading with auto index detection
                    df = pd.read_csv(input_file, index_col=0)
                    logger.info(
                        f"Successfully loaded CSV with index column: {input_file}, shape: {df.shape}"
                    )
                except Exception as e:
                    # If that fails, try without index column
                    logger.warning(
                        f"Error loading CSV with index: {e}, trying without index"
                    )
                    df = pd.read_csv(input_file)
                    logger.info(
                        f"Successfully loaded CSV without index: {input_file}, shape: {df.shape}"
                    )

                # Validate the DataFrame format
                if df.empty:
                    error_msg = f"CSV file is empty: {input_file}"
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), 400

                # Log some information about the DataFrame columns and values
                logger.info(f"CSV columns: {list(df.columns)}")
                logger.info(f"Index name: {df.index.name}")
                if len(df) > 0:
                    logger.info(f"First row: {df.iloc[0].to_dict()}")

                # Ensure we have at least one data column
                if len(df.columns) < 1:
                    error_msg = f"CSV file has insufficient columns: {list(df.columns)}"
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), 400

                # Get the sampling rate from the metadata file
                # Determine the metadata file path (we expect it to be in the same directory as input_file)
                input_dir = os.path.dirname(input_file)
                input_filename = os.path.basename(input_file)
                metadata_filename = input_filename.replace(".csv", ".meta.json")
                metadata_path = os.path.join(input_dir, metadata_filename)

                logger.info(f"Looking for metadata file at: {metadata_path}")

                # Try to read the sampling rate from the metadata file
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            meta = json.load(f)
                            logger.info(f"Metadata keys: {list(meta.keys())}")

                            # Direct sampling_rate field (main approach for consistency)
                            if (
                                "sampling_rate" in meta
                                and meta["sampling_rate"] is not None
                            ):
                                sample_rate = float(meta["sampling_rate"])
                                logger.info(
                                    f"Using sampling rate from metadata file: {sample_rate} Hz"
                                )
                            else:
                                # If no sampling_rate in metadata, raise an error with details
                                error_msg = f"No valid sampling rate found in metadata file. Available keys: {list(meta.keys())}"
                                logger.error(error_msg)
                                return jsonify({"error": error_msg}), 400
                    except Exception as e:
                        error_msg = f"Error reading metadata file: {str(e)}"
                        logger.error(error_msg)
                        return jsonify({"error": error_msg}), 500
                else:
                    # If metadata file doesn't exist, raise an error
                    error_msg = f"Metadata file not found: {metadata_path}. Cannot proceed without sampling rate."
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), 400

                # Convert DataFrame to numpy array for analysis
                logger.info(f"CSV file: {df.shape}, columns: {list(df.columns)}")

                # Handle different CSV structures
                if "timestamp" in df.columns:
                    # Standard format with timestamp column
                    data_columns = [col for col in df.columns if col != "timestamp"]
                    logger.info(f"Extracting data from columns: {data_columns}")
                    if len(data_columns) == 0:
                        error_msg = "CSV has timestamp column but no data columns"
                        logger.error(error_msg)
                        return jsonify({"error": error_msg}), 400
                    data_array = df[
                        data_columns
                    ].values.T  # Transpose to get channels as rows
                elif df.columns[0] == "Unnamed: 0" or df.index.name == "timestamp":
                    # Format with timestamp as index or first unnamed column
                    logger.info(
                        f"Assuming first column is timestamp, using remaining columns"
                    )
                    if len(df.columns) <= 1:
                        error_msg = "CSV file has index/timestamp but no data columns"
                        logger.error(error_msg)
                        return jsonify({"error": error_msg}), 400
                    data_array = df.iloc[:, 1:].values.T  # Skip first column, transpose
                else:
                    # Unknown format, try to use all columns as data
                    logger.warning(
                        f"CSV format not recognized, using all columns as data"
                    )
                    data_array = df.values.T

                # Final validation
                logger.info(f"Final data array shape: {data_array.shape}")
                if data_array.shape[0] == 0 or data_array.shape[1] == 0:
                    error_msg = f"Extracted data array is empty: {data_array.shape}"
                    logger.error(error_msg)
                    return jsonify({"error": error_msg}), 400

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

                # Get start_time_seconds from metadata for correct x-axis
                start_time_seconds = 0
                try:
                    with open(metadata_path, "r") as f:
                        meta_for_plots = json.load(f)
                        start_time_seconds = meta_for_plots.get(
                            "requested_start_time_seconds", 0
                        )
                        logger.info(
                            f"Using start_time_seconds for plots: {start_time_seconds}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not get start_time_seconds from metadata: {e}. Using 0."
                    )

                create_synchrony_plots(
                    results, sample_rate, output_dir, start_time_seconds
                )

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


@api_bp.route("/nlp-query", methods=["POST"])
@token_required
def handle_nlp_query():
    """
    Process a natural language query for EEG analysis and run it through the existing pipeline

    Request Body:
    - query: The natural language query string
    - dataset_id: Pennsieve dataset ID (optional if set in environment)
    - package_id: Pennsieve package ID (optional if set in environment)
    """
    try:
        # Initialize services if not already done
        if data_retrieval_service is None:
            init_services()

        # Get request data
        data = request.json
        query = data.get("query")
        dataset_id = data.get("dataset_id")
        package_id = data.get("package_id")

        logger.info(f"NLP Query received: {query}")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # First use LLM to analyze the query and extract parameters
        llm_params = llm_analyze_query(query)

        logger.info(f"LLM extracted parameters: {llm_params}")

        # Set up parameters for the data retrieval and analysis
        if "error" in llm_params:
            logger.warning(f"LLM analysis failed: {llm_params['error']}")
            # Use default parameters as fallback
            time_window = [0, 30]  # Default to first 30 seconds
            frequency_band = None
            channel_ids = None
            logger.info(f"Using default time window: {time_window}")
        else:
            # Extract parameters from LLM analysis
            time_window = llm_params.get("time_window", [0, 30])
            frequency_band = llm_params.get("frequency_band")
            channel_ids = llm_params.get("channel_ids")
            logger.info(f"Using LLM-extracted time window: {time_window}")

        # Prepare parameters for data retrieval
        # When is_relative_time=True, the Pennsieve SDK expects the times in seconds
        start_time_seconds = time_window[0]
        end_time_seconds = time_window[1]

        logger.info(
            f"Using time window in seconds: [{start_time_seconds}, {end_time_seconds}]"
        )

        # Retrieve and save the data segment using the existing data retrieval functionality
        try:
            # Prepare request for data retrieval
            segment_request = {
                "dataset_id": dataset_id,
                "package_id": package_id,
                "channel_ids": channel_ids,
                "start_time_seconds": start_time_seconds,
                "end_time_seconds": end_time_seconds,
                "output_filename": f"nlp_query_{int(time.time())}",
                "is_relative_time": True,
            }

            # Call the data retrieval service to save data
            logger.info(f"Retrieving data segment with parameters: {segment_request}")

            data_response = data_retrieval_service.get_data_range(
                dataset_id=dataset_id,
                package_id=package_id,
                channel_ids=channel_ids,
                start_time=start_time_seconds,
                end_time=end_time_seconds,
                force_refresh=False,
                is_relative_time=True,  # This flag means times are in seconds from recording start
            )

            # Check for error
            if "error" in data_response:
                logger.error(f"Error retrieving data: {data_response['error']}")
                return jsonify(data_response), 500

            # Log data response details to help debug time window issues
            if "times" in data_response:
                logger.info(
                    f"Retrieved data with {len(data_response['times'])} time points"
                )
                if len(data_response["times"]) > 0:
                    first_time = data_response["times"][0]
                    last_time = data_response["times"][-1]
                    duration_ms = last_time - first_time
                    logger.info(
                        f"Time range in response: {first_time}ms to {last_time}ms (duration: {duration_ms}ms)"
                    )

            # Create timestamp for the input directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_input_dir = os.path.join(os.getcwd(), "data", "input")
            os.makedirs(base_input_dir, exist_ok=True)
            timestamped_dir = os.path.join(base_input_dir, timestamp)
            os.makedirs(timestamped_dir, exist_ok=True)

            output_filename = f"nlp_query_{int(time.time())}"
            output_csv = f"{output_filename}.csv"
            output_path = os.path.join(timestamped_dir, output_csv)

            # Get metadata for the current data
            metadata = data_retrieval_service.metadata_service.get_metadata(
                dataset_id, package_id
            )

            # Extract the sampling frequency from metadata
            sampling_rate = metadata.get("time_range", {}).get("sampling_rate")
            logger.info(f"Using sampling rate from metadata: {sampling_rate} Hz")

            # Get channel metadata for names
            channel_info = data_retrieval_service.get_channel_info(
                dataset_id, package_id
            )

            # Create mapping of channel IDs to names
            channel_names = {}
            if channel_ids:
                channel_names = {
                    channel["id"]: channel["name"]
                    for channel in channel_info
                    if channel["id"] in channel_ids
                }
            else:
                # If no specific channels requested, include all
                channel_names = {
                    channel["id"]: channel["name"] for channel in channel_info
                }

            logger.info(f"Created mapping of {len(channel_names)} channel IDs to names")

            # Create metadata JSON file with the query information and sampling rate
            metadata_json = {
                "dataset_id": dataset_id,
                "package_id": package_id,
                "channel_ids": (
                    channel_names if channel_ids else {}
                ),  # Map of channel IDs to channel names
                "requested_start_time_seconds": start_time_seconds,
                "requested_end_time_seconds": end_time_seconds,
                "requested_duration_seconds": (end_time_seconds - start_time_seconds),
                "is_relative_time": True,
                "absolute_start_time_ns_epoch": metadata.get("time_range", {}).get(
                    "start"
                ),
                "absolute_end_time_ns_epoch": metadata.get("time_range", {}).get("end"),
                "retrieved_duration_seconds": data_response.get("duration_seconds", 0)
                / 1e6,
                "sampling_rate": sampling_rate,
                "samples": data_response.get("samples", 0),
                "retrieved_at": datetime.datetime.now().isoformat(),
                "query": query,  # Add the query for NLP workflow
                "llm_analysis": llm_params,  # Add LLM analysis for NLP workflow
            }

            # Save metadata JSON file
            metadata_path = os.path.join(
                timestamped_dir, f"{output_filename}.meta.json"
            )
            with open(metadata_path, "w") as f:
                json.dump(metadata_json, f, indent=2)

            logger.info(f"Saved metadata to {metadata_path}")

            # Extract and save data
            try:
                # Get the raw dataframe
                raw_df = data_response.get("raw_dataframe")

                # Verify that the dataframe reflects the requested time window
                if raw_df is not None and not raw_df.empty:
                    # Save to CSV
                    raw_df.to_csv(output_path, index=True, index_label="timestamp")
                    logger.info(f"Saved data to {output_path}")
                else:
                    raise ValueError("Dataframe is empty or None")
            except Exception as e:
                logger.error(f"Error saving data to CSV: {str(e)}", exc_info=True)
                return jsonify({"error": f"Error saving data: {str(e)}"}), 500

            # Now run the analysis
            # Determine which bands to analyze based on the query
            selected_bands = []
            if frequency_band:
                selected_bands = [frequency_band]
            else:
                # If no specific band mentioned, analyze all standard bands
                selected_bands = ["delta", "theta", "alpha", "beta", "gamma"]

            logger.info(
                f"Running analysis on: {output_path} with bands: {selected_bands}"
            )

            try:
                # Load the CSV file into a pandas DataFrame
                try:
                    df = pd.read_csv(output_path, index_col=0)
                except Exception:
                    df = pd.read_csv(output_path)

                if df.empty:
                    raise ValueError("CSV file is empty")

                # Get the sampling rate from the metadata file
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    if "sampling_rate" in meta and meta["sampling_rate"] is not None:
                        sample_rate = float(meta["sampling_rate"])
                    else:
                        raise ValueError("No sampling rate found in metadata")

                    # Extract start_time_seconds from metadata for correct plot x-axis
                    start_time_seconds = meta.get("requested_start_time_seconds", 0)
                    logger.info(
                        f"Using start_time_seconds from metadata: {start_time_seconds}"
                    )

                # Prepare the data array for analysis
                if "timestamp" in df.columns:
                    data_columns = [col for col in df.columns if col != "timestamp"]
                    data_array = df[data_columns].values.T
                else:
                    data_array = df.iloc[:, 1:].values.T

                # Import analysis functions
                from app.eeg_synchrony import (
                    calculate_synchrony,
                    bandpass_filter,
                    create_synchrony_plots,
                )

                # Define frequency bands
                freq_bands = {
                    "delta": (0.5, 4),
                    "theta": (4, 8),
                    "alpha": (8, 13),
                    "beta": (13, 30),
                    "gamma": (30, 80),
                }

                # Filter to selected bands
                selected_freq_bands = {}
                for band in selected_bands:
                    if band in freq_bands:
                        selected_freq_bands[band] = freq_bands[band]

                # Create output directory
                output_directory = os.path.join(
                    os.getcwd(), "data", "output", timestamp
                )
                os.makedirs(output_directory, exist_ok=True)

                # Initialize results dictionary
                results = {}

                # Process each frequency band
                for band_name, (low_freq, high_freq) in selected_freq_bands.items():
                    logger.info(
                        f"Processing {band_name} band ({low_freq}-{high_freq} Hz)"
                    )

                    # Apply bandpass filter
                    filtered_data = bandpass_filter(
                        data_array, low_freq, high_freq, sample_rate
                    )

                    # Calculate synchrony
                    r_t, R = calculate_synchrony(filtered_data)

                    # Store results
                    results[band_name] = {
                        "r_t": r_t.tolist(),
                        "mean_synchrony": float(R),
                    }

                    logger.info(f"{band_name} band mean synchrony: {R:.4f}")

                # Generate plots
                create_synchrony_plots(
                    results, sample_rate, output_directory, start_time_seconds
                )

                # Create a dictionary to return to the frontend
                static_url_base = "/static/output"
                response_data = {
                    "message": "Analysis completed successfully",
                    "query": query,
                    "timestamp": timestamp,
                    "output_dir": output_directory,
                    "synchrony": {
                        **{
                            f"{band}_mean": results[band]["mean_synchrony"]
                            for band in results
                        }
                    },
                    "images": [],
                    "redirect_to_gallery": True,  # Add flag to redirect to gallery
                }

                # Add generated images to the response
                for file in os.listdir(output_directory):
                    if file.endswith(".png"):
                        response_data["images"].append(
                            {
                                "name": file,
                                "url": f"{static_url_base}/{timestamp}/{file}",
                            }
                        )

                # Save analysis to gallery history
                analysis_name = (
                    f"NLP Query: {query[:40]}{'...' if len(query) > 40 else ''}"
                )
                new_analysis = {
                    "id": f"analysis_{int(time.time())}",
                    "name": analysis_name,
                    "timestamp": timestamp,
                    "path": output_directory,
                    "query": query,  # Include the query for reference
                    "nlp_analysis": True,  # Flag as NLP analysis
                }

                # The frontend will handle saving to localStorage
                response_data["gallery_entry"] = new_analysis

                return jsonify(response_data), 200

            except Exception as analysis_error:
                logger.error(
                    f"Error running analysis: {str(analysis_error)}", exc_info=True
                )
                return (
                    jsonify(
                        {
                            "error": f"Error running analysis: {str(analysis_error)}",
                            "query": query,
                            "timestamp": timestamp,
                        }
                    ),
                    500,
                )

        except Exception as e:
            logger.error(f"Error processing the query: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error in NLP query handler: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/nlp-analyze", methods=["POST"])
@token_required
def analyze_nlp_query():
    """
    Analyze a natural language query without running the analysis

    Request Body:
    - query: The natural language query string
    """
    try:
        # Get request data
        data = request.json
        query = data.get("query")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Use LLM to analyze the query and extract parameters
        analysis = llm_analyze_query(query)

        # Log the raw analysis result from the LLM
        logger.info(f"Raw LLM analysis result: {analysis}")

        return jsonify({"query": query, "analysis": analysis}), 200

    except Exception as e:
        logger.error(f"Error analyzing NLP query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
