# Use RunPod's optimized PyTorch image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Create cache directories with proper permissions
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/.cache/torch && \
    chmod -R 777 /app/.cache

# Pre-download and cache the model to reduce cold start time
RUN python -c "
import torch
from transformers import CLIPModel, CLIPProcessor
import os

print('Pre-loading CLIP model...')
cache_dir = '/app/.cache/huggingface'

# Download model and processor
model = CLIPModel.from_pretrained(
    'openai/clip-vit-base-patch32', 
    cache_dir=cache_dir,
    torch_dtype=torch.float16
)
processor = CLIPProcessor.from_pretrained(
    'openai/clip-vit-base-patch32', 
    cache_dir=cache_dir
)

print('Model pre-loaded successfully!')
"

# Copy application code
COPY handler.py .

# Create a simple health check script
RUN echo '#!/bin/bash\npython -c "from handler import health_check; import json; print(json.dumps(health_check()))"' > /app/health.sh && \
    chmod +x /app/health.sh

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD /app/health.sh || exit 1

# Expose port (for debugging/testing)
EXPOSE 8000

# Set the entrypoint
CMD ["python", "handler.py"]
