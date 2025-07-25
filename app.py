import os
import time
import torch
import requests
import logging
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from transformers import CLIPProcessor, CLIPModel
from contextlib import asynccontextmanager

# إعداد الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# متغيرات النموذج العامة
model = None
processor = None
model_ready = False

def initialize_model():
    """تحميل النموذج بشكل آمن"""
    global model, processor, model_ready
    
    if model_ready:
        logger.info("✅ Model already loaded")
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events - بدل on_event المهجور"""
    # Startup
    logger.info("🔧 Starting application...")
    success = initialize_model()
    if success:
        logger.info("🎉 Application ready!")
    else:
        logger.error("💥 Startup failed!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")

# إنشاء التطبيق مع lifespan
app = FastAPI(
    title="CLIP Embedding API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """الصفحة الرئيسية - اختبار سريع"""
    return {
        "status": "running",
        "model_ready": model_ready,
        "gpu_available": torch.cuda.is_available(),
        "message": "CLIP Embedding API is ready!",
        "timestamp": int(time.time())
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
        logger.info("📥 Received POST request")
        
        # التحقق من جاهزية النموذج
        if not model_ready:
            logger.info("🔄 Model not ready, initializing...")
            if not initialize_model():
                raise HTTPException(status_code=503, detail="Model not available")
        
        # قراءة الطلب
        try:
            data = await request.json()
            logger.info(f"📋 Request data: {data}")
        except Exception as e:
            logger.error(f"❌ JSON parsing error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON")
        
        # استخراج image_url
        input_data = data.get("input", {})
        image_url = input_data.get("image_url")
        
        if not image_url:
            logger.error("❌ No image_url provided")
            raise HTTPException(status_code=400, detail="image_url required in input")
        
        logger.info(f"🖼️ Processing image: {image_url}")
        start_time = time.time()
        
        # تحميل الصورة
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            logger.info(f"✅ Image loaded: {image.size}")
        except Exception as e:
            logger.error(f"❌ Image loading failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Image loading failed: {str(e)}")
        
        # معالجة الصورة
        try:
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
            logger.info(f"✅ Processing completed in {processing_time:.2f}s")
            
            return {
                "embedding": embedding,
                "processing_time": round(processing_time, 3),
                "embedding_size": len(embedding),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Model processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/embed")
async def embed_alias(request: Request):
    """نسخة بديلة من الـ endpoint"""
    return await embed_image(request)

# تشغيل مع uvicorn
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🔥 Starting server directly...")
    
    # تحميل النموذج قبل بدء الخادم
    initialize_model()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
