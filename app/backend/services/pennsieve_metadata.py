import os
import logging
import time
from pennsieve import Pennsieve
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("pennsieve_metadata_service")


class PennsieveMetadataService:
    """
    Service for retrieving metadata from Pennsieve datasets and packages.
    Includes caching to reduce API calls.
    """

    def __init__(self):
        """Initialize the Pennsieve metadata service"""
        self.pennsieve_client = None

        # Simple cache for metadata with TTL
        self.metadata_cache = {}
        self.cache_ttl = 300  # Cache TTL in seconds (5 minutes)

        # Connect to Pennsieve
        logger.info("Initializing PennsieveMetadataService")
        self.connect_to_pennsieve()

    def connect_to_pennsieve(self):
        """Connect to Pennsieve API"""
        try:
            logger.info("Connecting to Pennsieve...")
            self.pennsieve_client = Pennsieve()
            logger.info("Successfully connected to Pennsieve")

        except Exception as e:
            logger.error(f"Error connecting to Pennsieve: {e}")
            raise ConnectionError(f"Failed to connect to Pennsieve: {e}")

    def get_metadata(self, dataset_id=None, package_id=None):
        """
        Get metadata for a dataset and package.
        Uses cache if available and not expired.

        Args:
            dataset_id (str, optional): Pennsieve dataset ID
            package_id (str, optional): Pennsieve package ID

        Returns:
            dict: Metadata information
        """
        # Use environment variables as fallback
        dataset_id = dataset_id or os.getenv("PENNSIEVE_DATASET")
        package_id = package_id or os.getenv("PENNSIEVE_PACKAGE")

        logger.info(
            f"Retrieving metadata for dataset:{dataset_id}, package:{package_id}"
        )

        if not dataset_id:
            logger.error("Dataset ID is required but not provided")
            raise ValueError("Dataset ID is required but not provided")

        # Generate cache key
        cache_key = f"{dataset_id}:{package_id}"

        # Check if data is in cache and not expired
        if cache_key in self.metadata_cache:
            cache_entry = self.metadata_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                logger.info(f"Using cached metadata for {cache_key}")
                return cache_entry["data"]

        # Data not in cache or expired, fetch from Pennsieve
        try:
            # Access dataset using use_dataset method
            logger.info(f"Setting active dataset to {dataset_id}")
            self.pennsieve_client.use_dataset(dataset_id)

            # Get dataset object reference
            dataset = self.pennsieve_client.dataset

            # Get dataset info
            dataset_info = {
                "id": dataset_id,
                "name": dataset.name if hasattr(dataset, "name") else dataset_id,
                "description": (
                    dataset.description if hasattr(dataset, "description") else ""
                ),
            }

            # Get channels for the specified package
            channels_info = []
            time_range = {}

            if package_id:
                try:
                    logger.info(f"Retrieving channels for package {package_id}")
                    channels = self.pennsieve_client.timeseries.getChannels(
                        dataset,
                        package_id,
                        True,  # Include additional information
                    )
                    logger.info(
                        f"Retrieved {len(channels) if channels else 0} channels"
                    )

                    # Extract channel information
                    for channel in channels:
                        channel_info = {
                            "id": channel.id,
                            "name": channel.name,
                        }

                        # Add optional attributes if they exist
                        for attr in ["unit", "rate", "type", "group"]:
                            if hasattr(channel, attr):
                                channel_info[attr] = getattr(channel, attr)

                        channels_info.append(channel_info)

                    # Get time range if we have channels
                    if channels:
                        # Get time range attributes
                        time_range = {
                            "start": channels[0].start_time,
                            "end": channels[0].end_time,
                            "duration_seconds": (
                                channels[0].end_time - channels[0].start_time
                            )
                            / 1e6,  # Convert μs to s
                            "sampling_rate": (
                                channels[0].rate
                                if hasattr(channels[0], "rate")
                                else None
                            ),
                            "unit": (
                                channels[0].unit
                                if hasattr(channels[0], "unit")
                                else None
                            ),
                        }
                except Exception as e:
                    logger.error(f"Error getting channels: {e}")

            # Compile metadata
            metadata = {
                "dataset": dataset_info,
                "package": {"id": package_id} if package_id else None,
                "channels": channels_info,
                "time_range": time_range,
                "total_channels": len(channels_info),
            }

            # Cache the result
            self.metadata_cache[cache_key] = {
                "timestamp": time.time(),
                "data": metadata,
            }

            return metadata

        except Exception as e:
            logger.error(f"Error retrieving metadata from Pennsieve: {e}")
            raise
