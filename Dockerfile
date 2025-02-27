# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir numpy scipy matplotlib mne

# Copy application code
COPY app/ /app/

# Create volume directories
RUN mkdir -p data/input data/output

# Default environment variables
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output

# Set entrypoint to run synchrony analysis
CMD ["python", "-u", "eeg_synchrony.py"]