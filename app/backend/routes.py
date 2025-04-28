from flask import Blueprint, jsonify, request, current_app
import logging
from app.backend.services.pennsieve_metadata import PennsieveMetadataService
from app.backend.services.data_retrieval import DataRetrievalService
import os
from functools import wraps

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
        token = None
        auth_header = request.headers.get("Authorization")

        # Check if Authorization header exists and is properly formatted
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            logger.info(f"Auth token received: {token[:10]}...")
        else:
            logger.warning(f"Missing or malformed Authorization header: {auth_header}")

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
