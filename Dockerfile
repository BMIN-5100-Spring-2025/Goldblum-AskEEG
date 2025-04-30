# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install required dependencies
RUN pip install --no-cache-dir numpy scipy matplotlib mne boto3 python-dotenv

# Copy application code and .env file
COPY app/ /app/
COPY .env /app/.env

# Create volume directories
RUN mkdir -p data/input data/output

# Default environment variables (will be overridden by .env if provided)
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output
ENV RUN_MODE=local
ENV S3_BUCKET_NAME=goldblum-askeeg
ENV SESSION_ID=""

# Set entrypoint to run synchrony analysis
CMD ["python", "-u", "eeg_synchrony.py"]