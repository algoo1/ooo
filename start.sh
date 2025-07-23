#!/bin/bash

# start.sh - سكريبت بدء التشغيل للإنتاج

# Set environment variables
export PYTHONPATH=/app
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p /app/logs
mkdir -p /dev/shm

echo "Starting SentenceTransformer API with Gunicorn..."

# تشغيل مع ملف الإعدادات
gunicorn --config gunicorn.conf.py app:app

# أو تشغيل مباشر مع المعاملات
# gunicorn --bind 0.0.0.0:8000 \
#          --workers 1 \
#          --threads 4 \
#          --timeout 300 \
#          --keep-alive 10 \
#          --max-requests 500 \
#          --max-requests-jitter 50 \
#          --preload \
#          --worker-tmp-dir /dev/shm \
#          --log-level info \
#          --access-logfile /app/logs/access.log \
#          --error-logfile /app/logs/error.log \
#          --capture-output \
#          app:app
