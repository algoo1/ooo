# gunicorn.conf.py
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
# للـ ML models استخدم worker واحد فقط عشان الـ GPU memory
workers = 1
worker_class = "gthread"
threads = 4
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# Timeout settings - مهم للـ ML inference
timeout = 300  # 5 minutes
keepalive = 10
graceful_timeout = 30

# Memory management
worker_tmp_dir = "/dev/shm"  # Use RAM for better performance

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "sentence_transformer_api"

# SSL (if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Restart workers after this many requests
max_requests = 500

# Add jitter to prevent all workers restarting at once
max_requests_jitter = 50

# Worker timeout for graceful restart
timeout = 120

# Environment
raw_env = [
    "PYTHONPATH=/app",
    "CUDA_VISIBLE_DEVICES=0"  # استخدم GPU الأولى فقط
]
