FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# بيئة وكاش
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV CLIP_MODEL_ID=openai/clip-vit-base-patch32
ENV PRELOAD_MODEL=1

WORKDIR /app

# أدوات نظام خفيفة + تنظيف
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# تثبيت باكجات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# كاش الموديل (اختياري لسرعة الـ build)
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch && \
    python -c "from transformers import CLIPModel, CLIPProcessor; \
               CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface'); \
               CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface')"

# نسخ الكود
COPY app.py .

EXPOSE 8000

# تشغيل Uvicorn مباشرة
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
