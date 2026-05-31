# Base image
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ResNet model
RUN python preload_model.py

# Create required directories
RUN mkdir -p data/uploads data/outputs

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Start backend first, then frontend
CMD bash -c "uvicorn app.api.main:app --host 0.0.0.0 --port 8000 & sleep 10 && streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"