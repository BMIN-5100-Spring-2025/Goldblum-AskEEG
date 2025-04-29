import os
import logging
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv
import time

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
            start_time (int/float): Time value
                             - With is_relative_time=True: seconds from start of recording
                             - With is_relative_time=False: absolute microseconds since Unix epoch
            end_time (int/float): Time value
                           - With is_relative_time=True: seconds from start of recording
                           - With is_relative_time=False: absolute microseconds since Unix epoch
            force_refresh (bool): Force refresh of cached data
            is_relative_time (bool): Whether the time values are relative to the start of recording.
                                    If True (default), times are in seconds from beginning of recording.
                                    If False, times are absolute microseconds since Unix epoch.

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

            # Get metadata
            metadata = self.metadata_service.get_metadata(dataset_id, package_id)

            # If no channel IDs provided, get all available channels
            if not channel_ids:
                channel_ids = [
                    channel["id"] for channel in metadata.get("channels", [])
                ]

            if not channel_ids:
                raise ValueError("No channels found for the specified package")

            # Get sampling rate from metadata
            sampling_rate = metadata.get("time_range", {}).get("sampling_rate")
            if sampling_rate:
                logger.info(f"Package sampling rate from metadata: {sampling_rate} Hz")

            # Log the request details
            logger.info(
                f"Retrieving data from Pennsieve: dataset={dataset_id}, "
                f"package={package_id}, channels={len(channel_ids)}, "
                f"start={start_time}, end={end_time}, "
                f"force_refresh={force_refresh}, is_relative_time={is_relative_time}"
            )

            # Get data from Pennsieve - always using the is_relative_time parameter as provided
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
            return self._process_data_frame(
                data_frame, channel_ids, dataset_id, package_id
            )

        except Exception as e:
            logger.error(f"Error retrieving data from Pennsieve: {e}")
            raise

    def _process_data_frame(
        self, data_frame, channel_ids, dataset_id=None, package_id=None
    ):
        """
        Process the data frame returned from Pennsieve

        Args:
            data_frame (pandas.DataFrame): DataFrame from Pennsieve
            channel_ids (list): List of channel IDs in the data
            dataset_id (str, optional): Pennsieve dataset ID
            package_id (str, optional): Pennsieve package ID

        Returns:
            dict: Processed data with additional information
        """
        # Get metadata sampling rate if available
        metadata_sampling_rate = None
        if dataset_id and package_id:
            try:
                metadata = self.metadata_service.get_metadata(dataset_id, package_id)
                metadata_sampling_rate = metadata.get("sampling_rate")
                if metadata_sampling_rate:
                    logger.info(
                        f"Using sampling rate from metadata: {metadata_sampling_rate} Hz"
                    )
            except Exception as e:
                logger.warning(f"Could not get sampling rate from metadata: {e}")

        # Check if data is empty
        if data_frame is None or data_frame.empty:
            logger.warning(
                f"Empty data frame returned for time range request: {len(channel_ids)} channels"
            )
            return {
                "data": None,
                "is_empty": True,
                "channels": channel_ids,
                "samples": 0,
                "duration_seconds": 0,
                "sampling_rate": metadata_sampling_rate,
            }

        # Get basic information
        start_time = data_frame.index[0]
        end_time = data_frame.index[-1]
        sample_count = len(data_frame)

        logger.info(
            f"Data frame received with {sample_count} samples across {len(data_frame.columns)} channels"
        )

        # Calculate duration_seconds, handle different types that might be returned
        try:
            # Try using timedelta's total_seconds method
            duration_seconds = (end_time - start_time).total_seconds()
        except AttributeError:
            # If we get a numpy.float64 instead of a timedelta, convert it manually
            if isinstance(end_time, np.datetime64) and isinstance(
                start_time, np.datetime64
            ):
                # Convert numpy datetime64 to seconds
                duration_seconds = (end_time - start_time) / np.timedelta64(1, "s")
            else:
                # As a fallback, try to calculate the difference directly
                duration_seconds = float(end_time) - float(start_time)
                if duration_seconds <= 0:
                    # If calculation fails or gives negative value, use sample count and metadata
                    logger.warning(
                        "Could not calculate proper duration, using sample count and metadata sampling rate"
                    )
                    # Use metadata sampling rate if available
                    if metadata_sampling_rate and metadata_sampling_rate > 0:
                        duration_seconds = sample_count / metadata_sampling_rate
                        logger.info(
                            f"Calculated duration using metadata sampling rate: {duration_seconds} sec"
                        )
                    else:
                        logger.warning(
                            "No valid sampling rate available, cannot determine duration"
                        )
                        duration_seconds = 0

        # Get sampling rate (samples per second) - use metadata if available, otherwise calculate
        if metadata_sampling_rate and metadata_sampling_rate > 0:
            sampling_rate = metadata_sampling_rate
            logger.info(f"Using sampling rate from metadata: {sampling_rate} Hz")
        elif sample_count > 1 and duration_seconds > 0:
            sampling_rate = sample_count / duration_seconds
            logger.info(f"Calculated sampling rate from data: {sampling_rate} Hz")
        else:
            sampling_rate = None
            logger.warning("Could not determine sampling rate")

        # Convert DataFrame to dictionary format suitable for JSON serialization
        data_dict = {
            "is_empty": False,
            "samples": sample_count,
            "duration_seconds": float(duration_seconds),  # Ensure it's a Python float
            "sampling_rate": (
                float(sampling_rate) if sampling_rate is not None else None
            ),
            "channels": {},
        }

        # Safely convert timestamps to milliseconds
        try:
            data_dict["times"] = data_frame.index.astype(np.int64) // 10**6
            data_dict["start_time"] = start_time.timestamp() * 1000
            data_dict["end_time"] = end_time.timestamp() * 1000
        except (AttributeError, TypeError):
            # If we can't convert directly, try an alternative approach
            logger.warning(
                "Could not convert timestamps directly, using index positions"
            )
            # Use sample indices as time placeholder (0-based)
            data_dict["times"] = list(range(sample_count))
            # Set approximate start/end times based on duration
            try:
                if hasattr(start_time, "timestamp"):
                    data_dict["start_time"] = start_time.timestamp() * 1000
                    data_dict["end_time"] = end_time.timestamp() * 1000
                else:
                    # Use epoch time as fallback
                    data_dict["start_time"] = 0
                    data_dict["end_time"] = duration_seconds * 1000
            except:
                data_dict["start_time"] = 0
                data_dict["end_time"] = duration_seconds * 1000

        # Convert each channel data to list
        for channel_id in channel_ids:
            if channel_id in data_frame.columns:
                # Convert numpy values to native Python types
                data_dict["channels"][channel_id] = [
                    float(x) for x in data_frame[channel_id].tolist()
                ]

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
