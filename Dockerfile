FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights during build to avoid cold loading at runtime
RUN python -c "from transformers import
