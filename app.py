import os
import time
import torch
import requests
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from transformers import CLIPProcessor, CLIPModel
import logging

# تكوين الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CLIP Image Embedding API")

# متغيرات عامة للنموذج (يتم تحميلها مرة واحدة)
model = None
processor = None

def load_model():
    """تحميل النموذج مرة واحدة عند بدء التطبيق"""
    global model, processor
    
    start_time = time.time()
    logger.info("بدء تحميل نموذج CLIP...")
    
    try:
        # استخدام cache_dir لضمان حفظ النموذج
        cache_dir = "/root/.cache/huggingface"
        os.makedirs(cache_dir, exist_ok=True)
        
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        
        # نقل النموذج للـ GPU إذا متوفر
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("تم نقل النموذج للـ GPU")
        
        load_time = time.time() - start_time
        logger.info(f"تم تحميل النموذج بنجاح في {load_time:.2f} ثانية")
        
    except Exception as e:
        logger.error(f"خطأ في تحميل النموذج: {str(e)}")
        raise e

# تحميل النموذج عند بدء التطبيق
load_model()

@app.get("/health")
async def health_check():
    """فحص حالة الخدمة"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": time.time()
    }

@app.post("/")
async def embed_image(request_data: dict):
    """استخراج الـ embedding من الصورة"""
    start_time = time.time()
    
    try:
        # التحقق من وجود النموذج
        if model is None or processor is None:
            raise HTTPException(status_code=500, detail="النموذج غير محمل")
        
        # استخراج رابط الصورة
        input_data = request_data.get("input", {})
        image_url = input_data.get("image_url")
        
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url مطلوب")
        
        logger.info(f"معالجة الصورة من: {image_url}")
        
        # تحميل الصورة مع timeout
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"خطأ في تحميل الصورة: {str(e)}")
        
        # معالجة الصورة
        inputs = processor(images=image, return_tensors="pt")
        
        # نقل البيانات للـ GPU إذا متوفر
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # استخراج الـ features
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            
            # تحويل لـ CPU إذا كان على GPU
            if features.is_cuda:
                features = features.cpu()
            
            embedding = features[0].tolist()
        
        processing_time = time.time() - start_time
        logger.info(f"تم استخراج الـ embedding في {processing_time:.2f} ثانية")
        
        return {
            "embedding": embedding,
            "processing_time": processing_time,
            "embedding_size": len(embedding)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ غير متوقع: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """أحداث بدء التطبيق"""
    logger.info("تم بدء تشغيل الخدمة")

@app.on_event("shutdown") 
async def shutdown_event():
    """أحداث إيقاف التطبيق"""
    logger.info("تم إيقاف الخدمة")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
