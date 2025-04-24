import os
import boto3
import glob
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("s3_operations")


class S3Operations:
    """
    Handle S3 operations for EEG data processing.
    """

    def __init__(self, bucket_name=None):
        """
        Initialize the S3 operations class.

        Args:
            bucket_name (str, optional): The name of the S3 bucket.
                                         If not provided, will use S3_BUCKET_NAME env var.
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
        if not self.bucket_name:
            logger.warning(
                "No S3 bucket name provided and S3_BUCKET_NAME env var not set"
            )

        # Initialize the S3 client
        self.s3_client = boto3.client("s3")

        # Check if we're running in Fargate or locally
        self.run_mode = os.getenv("RUN_MODE", "local")
        logger.info(f"Running in {self.run_mode} mode")

    def download_data(self, input_prefix, local_input_dir):
        """
        Download data from S3 to local directory.

        Args:
            input_prefix (str): The S3 key prefix to download files from
            local_input_dir (str): Local directory to download files to
        """
        if self.run_mode != "fargate":
            logger.info("Not running in Fargate mode, skipping S3 download")
            return

        if not self.bucket_name:
            logger.error("Cannot download from S3: no bucket name provided")
            return

        try:
            # Create input directory if it doesn't exist
            os.makedirs(local_input_dir, exist_ok=True)

            # List objects in the bucket with the given prefix
            logger.info(f"Listing objects in s3://{self.bucket_name}/{input_prefix}")
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=input_prefix
            )

            if "Contents" not in response:
                logger.warning(
                    f"No objects found in s3://{self.bucket_name}/{input_prefix}"
                )
                return

            # Download each file
            for obj in response["Contents"]:
                key = obj["Key"]
                filename = os.path.basename(key)
                local_file_path = os.path.join(local_input_dir, filename)

                logger.info(f"Downloading {key} to {local_file_path}")
                self.s3_client.download_file(
                    Bucket=self.bucket_name, Key=key, Filename=local_file_path
                )

            logger.info(f"Successfully downloaded files to {local_input_dir}")

        except Exception as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise

    def upload_results(self, local_output_dir, output_prefix):
        """
        Upload analysis results to S3.

        Args:
            local_output_dir (str): Local directory containing results
            output_prefix (str): The S3 key prefix to upload files to
        """
        if self.run_mode != "fargate":
            logger.info("Not running in Fargate mode, skipping S3 upload")
            return

        if not self.bucket_name:
            logger.error("Cannot upload to S3: no bucket name provided")
            return

        try:
            # Find all files in the output directory
            all_files = []
            for ext in ["*.png", "*.npz", "*.csv", "*.json"]:
                all_files.extend(glob.glob(os.path.join(local_output_dir, ext)))

            if not all_files:
                logger.warning(f"No result files found in {local_output_dir}")
                return

            # Upload each file
            for file_path in all_files:
                filename = os.path.basename(file_path)
                s3_key = f"{output_prefix}/{filename}"

                logger.info(
                    f"Uploading {file_path} to s3://{self.bucket_name}/{s3_key}"
                )
                self.s3_client.upload_file(
                    Filename=file_path, Bucket=self.bucket_name, Key=s3_key
                )

            logger.info(
                f"Successfully uploaded {len(all_files)} files to s3://{self.bucket_name}/{output_prefix}"
            )

        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise
