FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# نسخ المتطلبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# تحميل النموذج مسبقاً
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# نسخ الكود
COPY handler.py .

CMD ["python", "handler.py"]
