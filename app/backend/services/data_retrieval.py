import os
import logging
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("data_retrieval_service")


class DataRetrievalService:
    """
    Service for retrieving time-series data from Pennsieve.
    """

    def __init__(self, metadata_service):
        """
        Initialize the data retrieval service

        Args:
            metadata_service: An instance of PennsieveMetadataService
        """
        self.metadata_service = metadata_service

    def get_data_range(
        self,
        dataset_id=None,
        package_id=None,
        channel_ids=None,
        start_time=None,
        end_time=None,
        force_refresh=False,
        is_relative_time=True,
    ):
        """
        Retrieve a range of time-series data from Pennsieve

        Args:
            dataset_id (str, optional): Pennsieve dataset ID
            package_id (str, optional): Pennsieve package ID
            channel_ids (list, optional): List of channel IDs to retrieve
            start_time (int): Start time in microseconds
            end_time (int): End time in microseconds
            force_refresh (bool): Force refresh of cached data
            is_relative_time (bool): Whether the time values are relative to the start of recording

        Returns:
            dict: Dictionary containing the data and metadata
        """
        # Use environment variables as fallback
        dataset_id = dataset_id or os.getenv("PENNSIEVE_DATASET")
        package_id = package_id or os.getenv("PENNSIEVE_PACKAGE")

        if not dataset_id:
            raise ValueError("Dataset ID is required but not provided")
        if not package_id:
            raise ValueError("Package ID is required but not provided")

        try:
            # Get client from metadata service
            pennsieve_client = self.metadata_service.pennsieve_client

            # Access dataset using use_dataset method
            pennsieve_client.use_dataset(dataset_id)

            # Get dataset object reference
            dataset = pennsieve_client.dataset

            # If no channel IDs provided, get all available channels
            if not channel_ids:
                metadata = self.metadata_service.get_metadata(dataset_id, package_id)
                channel_ids = [channel["id"] for channel in metadata["channels"]]

            if not channel_ids:
                raise ValueError("No channels found for the specified package")

            # If no start_time or end_time provided, try to get from metadata
            if start_time is None or end_time is None:
                metadata = self.metadata_service.get_metadata(dataset_id, package_id)
                time_range = metadata.get("time_range", {})

                if time_range:
                    # Use metadata time range if available
                    if start_time is None:
                        start_time = time_range.get("start", 0)
                    if end_time is None:
                        end_time = time_range.get("end")

                # If we still don't have a valid time range, use defaults
                if start_time is None:
                    start_time = 0
                if end_time is None:
                    # Default to 10 seconds if we can't determine end time
                    end_time = start_time + (10 * 1000000)  # 10 seconds in microseconds
                    logger.warning(
                        f"Using default end_time: start + 10 seconds ({end_time})"
                    )

            # Log the request details
            logger.info(
                f"Retrieving data from Pennsieve: dataset={dataset_id}, "
                f"package={package_id}, channels={len(channel_ids)}, "
                f"start={start_time}, end={end_time}, "
                f"force_refresh={force_refresh}, is_relative_time={is_relative_time}"
            )

            # Get data from Pennsieve
            data_frame = pennsieve_client.timeseries.getRangeForChannels(
                dataset,
                package_id,
                channel_ids,
                start_time,
                end_time,
                force_refresh,  # is_refresh - whether to force refresh the cache
                is_relative_time,  # is_relative_time - whether time values are relative
            )

            # Process the data
            return self._process_data_frame(data_frame, channel_ids)

        except Exception as e:
            logger.error(f"Error retrieving data from Pennsieve: {e}")
            raise

    def _process_data_frame(self, data_frame, channel_ids):
        """
        Process the data frame returned from Pennsieve

        Args:
            data_frame (pandas.DataFrame): DataFrame from Pennsieve
            channel_ids (list): List of channel IDs in the data

        Returns:
            dict: Processed data with additional information
        """
        # Check if data is empty
        if data_frame.empty:
            return {
                "data": None,
                "is_empty": True,
                "channels": channel_ids,
                "samples": 0,
                "duration_seconds": 0,
            }

        # Get basic information
        start_time = data_frame.index[0]
        end_time = data_frame.index[-1]
        duration_seconds = (end_time - start_time).total_seconds()
        sample_count = len(data_frame)

        # Get sampling rate (samples per second)
        if sample_count > 1:
            sampling_rate = sample_count / duration_seconds
        else:
            sampling_rate = None

        # Convert DataFrame to dictionary format suitable for JSON serialization
        data_dict = {
            "times": data_frame.index.astype(np.int64)
            // 10**6,  # Convert to milliseconds
            "channels": {},
            "is_empty": False,
            "samples": sample_count,
            "duration_seconds": duration_seconds,
            "sampling_rate": sampling_rate,
            "start_time": start_time.timestamp() * 1000,  # Convert to milliseconds
            "end_time": end_time.timestamp() * 1000,  # Convert to milliseconds
        }

        # Convert each channel data to list
        for channel_id in channel_ids:
            if channel_id in data_frame.columns:
                data_dict["channels"][channel_id] = data_frame[channel_id].tolist()

        return data_dict

    def get_channel_info(self, dataset_id=None, package_id=None):
        """
        Get detailed information about channels in a package

        Args:
            dataset_id (str, optional): Pennsieve dataset ID
            package_id (str, optional): Pennsieve package ID

        Returns:
            list: List of channel information dictionaries
        """
        metadata = self.metadata_service.get_metadata(dataset_id, package_id)
        return metadata.get("channels", [])

    def convert_time_to_microseconds(self, time_value, reference_start=None):
        """
        Convert a time value to microseconds

        Args:
            time_value (str/int/float): Time value to convert
            reference_start (int): Reference start time in microseconds

        Returns:
            int: Time in microseconds
        """
        # If already a number, assume it's microseconds
        if isinstance(time_value, (int, float)):
            return int(time_value)

        # If datetime object, convert to microseconds since epoch
        if isinstance(time_value, datetime.datetime):
            return int(time_value.timestamp() * 1000000)

        # If string, try to parse as relative time
        if isinstance(time_value, str):
            # Try to parse as a datetime
            try:
                dt = datetime.datetime.fromisoformat(time_value)
                return int(dt.timestamp() * 1000000)
            except ValueError:
                pass

            # Handle special cases like "first X seconds"
            if reference_start is not None:
                if time_value.startswith("first "):
                    try:
                        # Parse "first X seconds" format
                        parts = time_value.split()
                        if len(parts) >= 3 and parts[2].lower() in [
                            "seconds",
                            "second",
                            "sec",
                            "s",
                        ]:
                            seconds = float(parts[1])
                            return reference_start + int(seconds * 1000000)
                    except (ValueError, IndexError):
                        pass

        # Default fallback
        return int(time_value) if time_value else None
