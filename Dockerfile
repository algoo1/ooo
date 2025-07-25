FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# متغيرات البيئة (للكاش وتحسين الأداء)
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# تثبيت المتطلبات النظام (لو احتجت curl أو أدوات أخرى)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# نسخ وتثبيت المتطلبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# إنشاء مجلد cache
RUN mkdir -p /app/.cache/huggingface

# تحميل النموذج في build time وتحديد الكاش
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface')"

# نسخ الكود الأساسي
COPY handler.py .

# فتح البورت (لو التطبيق يحتاج ذلك)
EXPOSE 8000

# أمر التشغيل
CMD ["python", "handler.py"]
