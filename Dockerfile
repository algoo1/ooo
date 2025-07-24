# استخدام Python image محسن
FROM python:3.10-slim

# تحديد متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# إنشاء مجلد العمل
WORKDIR /app

# تثبيت system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# إنشاء مجلد الـ cache
RUN mkdir -p /root/.cache/huggingface

# نسخ الكود
COPY . .

# تحديد المنفذ
EXPOSE 8000

# تشغيل التطبيق مع تحسينات الأداء
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "65"]
