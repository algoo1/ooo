import os
import time
import torch
import requests
import logging
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from transformers import CLIPProcessor, CLIPModel

# إعداد الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء التطبيق
app = FastAPI(title="CLIP Embedding API")

# متغيرات النموذج
model = None
processor = None
model_ready = False

def initialize_model():
    """تحميل النموذج بشكل آمن"""
    global model, processor, model_ready
    
    if model_ready:
        return True
    
    try:
        logger.info("🚀 Loading CLIP model...")
        start_time = time.time()
        
        # تحديد cache directory
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/app/.cache/huggingface')
        os.makedirs(cache_dir, exist_ok=True)
        
        # تحميل النموذج
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        
        # نقل للـ GPU
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ Model loaded on GPU")
        else:
            logger.info("✅ Model loaded on CPU")
        
        model_ready = True
        load_time = time.time() - start_time
        logger.info(f"✅ Model ready in {load_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        model_ready = False
        return False

@app.on_event("startup")
async def startup():
    """تهيئة التطبيق"""
    logger.info("🔧 Starting application...")
    
    # محاولة تحميل النموذج
    success = initialize_model()
    if success:
        logger.info("🎉 Application ready!")
    else:
        logger.error("💥 Startup failed!")

@app.get("/")
async def root():
    """الصفحة الرئيسية - اختبار سريع"""
    return {
        "status": "running",
        "model_ready": model_ready,
        "gpu_available": torch.cuda.is_available(),
        "message": "CLIP Embedding API is ready!"
    }

@app.get("/health")
async def health():
    """فحص الصحة"""
    return {
        "healthy": model_ready,
        "model_loaded": model is not None,
        "gpu": torch.cuda.is_available(),
        "timestamp": int(time.time())
    }

@app.post("/")
async def embed_image(request: Request):
    """استخراج embedding من الصورة"""
    try:
        # التحقق من جاهزية النموذج
        if not model_ready:
            logger.info("🔄 Model not ready, initializing...")
            if not initialize_model():
                raise HTTPException(status_code=503, detail="Model not available")
        
        # قراءة الطلب
        data = await request.json()
        image_url = data.get("input", {}).get("image_url")
        
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url required")
        
        logger.info(f"🖼️ Processing: {image_url}")
        start_time = time.time()
        
        # تحميل الصورة
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image loading failed: {str(e)}")
        
        # معالجة الصورة
        inputs = processor(images=image, return_tensors="pt")
        
        if torch.cuda.is_available() and model.device.type == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # استخراج الـ embedding
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if features.is_cuda:
                features = features.cpu()
            embedding = features[0].tolist()
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Done in {processing_time:.2f}s")
        
        return {
            "embedding": embedding,
            "processing_time": round(processing_time, 3),
            "embedding_size": len(embedding)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# تشغيل مع uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # تحميل النموذج قبل بدء الخادم
    initialize_model()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=30
    )
