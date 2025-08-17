FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# إنشاء مجلد الكاش وتحميل النموذج مسبقًا
RUN mkdir -p /app/.cache/huggingface
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/.cache/huggingface')"

COPY handler.py .

EXPOSE 8000

CMD ["python", "handler.py"]
