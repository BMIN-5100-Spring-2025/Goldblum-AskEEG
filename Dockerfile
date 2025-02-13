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

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy YASA processing script
COPY app/yasa_from_edf.py .

# Create volume directories
RUN mkdir -p /app/data/input /app/data/output

# Set entrypoint to run YASA analysis
CMD ["python", "yasa_from_edf.py"] 