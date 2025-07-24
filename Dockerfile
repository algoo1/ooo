# استخدام Python image محسن مع PyTorch
FROM python:3.10-slim

# تحديد متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# إنشاء مجلد العمل
WORKDIR /app

# تثبيت system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# إنشاء مجلدات الـ cache
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/.cache/torch

# تحميل النموذج مسبقاً لتسريع cold start
RUN python -c "from transformers import CLIPModel, CLIPProcessor; import os; os.makedirs('/app/.cache/huggingface', exist_ok=True); CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface')"

# نسخ الكود
COPY . .

# تحديد المنفذ
EXPOSE 8000

# تشغيل التطبيق مع تحسينات الأداء
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120", "--timeout-graceful-shutdown", "30"]
