FROM python:3.10-slim

# متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# تثبيت المتطلبات النظام
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# نسخ وتثبيت المتطلبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# إنشاء مجلد cache
RUN mkdir -p /app/.cache/huggingface

# تحميل النموذج في build time
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface')"

# نسخ الكود
COPY app.py .

EXPOSE 8000

# تشغيل مع إعدادات محسنة
CMD ["python", "app.py"]
