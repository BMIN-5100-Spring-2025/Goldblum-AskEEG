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

# Install minimal dependencies (avoid extras we don't need)
RUN pip install --no-cache-dir numpy scipy matplotlib mne

# Copy synchrony script
COPY app/eeg_synchrony.py .

# Create volume directories
RUN mkdir -p /app/data/input /app/data/output

# Set entrypoint to run synchrony analysis
CMD ["python", "-u", "eeg_synchrony.py"]