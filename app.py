import os
import time
import torch
import requests
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from transformers import CLIPProcessor, CLIPModel
import logging
import asyncio

# تكوين الـ logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="CLIP Image Embedding API")

# متغيرات عامة للنموذج (يتم تحميلها مرة واحدة)
model = None
processor = None
model_loaded = False

def load_model():
    """تحميل النموذج مع تحسينات للسرعة"""
    global model, processor, model_loaded
    
    if model_loaded:
        logger.info("✅ النموذج محمل مسبقاً")
        return True
    
    start_time = time.time()
    logger.info("🚀 بدء تحميل نموذج CLIP...")
    
    try:
        # تحديد مجلد الـ cache
        cache_dir = "/app/.cache/huggingface"
        os.makedirs(cache_dir, exist_ok=True)
        
        # تحميل النموذج مع تحسينات
        logger.info("📥 تحميل النموذج...")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        logger.info("📥 تحميل المعالج...")
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        
        # نقل للـ GPU إذا متوفر
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("🎮 تم نقل النموذج للـ GPU")
        else:
            logger.info("💻 تشغيل على CPU")
        
        model_loaded = True
        load_time = time.time() - start_time
        logger.info(f"✅ تم تحميل النموذج في {load_time:.2f} ثانية")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
        model_loaded = False
        return False

@app.on_event("startup")
async def startup_event():
    """تحميل النموذج عند بداية التطبيق"""
    logger.info("🔧 بدء تهيئة التطبيق...")
    
    # تحميل النموذج في background task
    success = load_model()
    
    if success:
        logger.info("🎉 التطبيق جاهز للاستخدام!")
    else:
        logger.error("💥 فشل في تهيئة التطبيق")

@app.get("/")
async def root():
    """الصفحة الرئيسية"""
    return {
        "status": "running",
        "message": "CLIP Embedding API",
        "model_loaded": model_loaded,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/health")
async def health_check():
    """فحص حالة الخدمة"""
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A",
        "timestamp": time.time()
    }

@app.post("/")
async def embed_image(request: Request):
    """استخراج الـ embedding من الصورة"""
    start_time = time.time()
    
    try:
        # التحقق من تحميل النموذج
        if not model_loaded or model is None or processor is None:
            # محاولة تحميل النموذج إذا لم يكن محملاً
            logger.info("🔄 محاولة تحميل النموذج...")
            if not load_model():
                raise HTTPException(status_code=503, detail="النموذج غير متاح حالياً")
        
        # قراءة البيانات
        try:
            request_data = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"خطأ في قراءة البيانات: {str(e)}")
        
        # استخراج رابط الصورة
        input_data = request_data.get("input", {})
        image_url = input_data.get("image_url")
        
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url مطلوب في input")
        
        logger.info(f"🔄 معالجة الصورة: {image_url}")
        
        # تحميل الصورة مع timeout
        try:
            response = requests.get(image_url, timeout=15, stream=True)
            response.raise_for_status()
            
            # التحقق من نوع المحتوى
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="الرابط لا يحتوي على صورة صالحة")
            
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=408, detail="انتهت مهلة تحميل الصورة")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"خطأ في تحميل الصورة: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"خطأ في معالجة الصورة: {str(e)}")
        
        # معالجة الصورة
        try:
            inputs = processor(images=image, return_tensors="pt")
            
            # نقل البيانات للـ GPU إذا متوفر
            if torch.cuda.is_available() and model.device.type == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # استخراج الـ features
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                
                # تحويل لـ CPU إذا كان على GPU
                if features.is_cuda:
                    features = features.cpu()
                
                embedding = features[0].tolist()
            
        except Exception as e:
            logger.error(f"خطأ في استخراج الـ embedding: {str(e)}")
            raise HTTPException(status_code=500, detail=f"خطأ في معالجة الصورة: {str(e)}")
        
        processing_time = time.time() - start_time
        logger.info(f"✅ تم استخراج الـ embedding في {processing_time:.2f} ثانية")
        
        return {
            "embedding": embedding,
            "processing_time": round(processing_time, 3),
            "embedding_size": len(embedding),
            "gpu_used": torch.cuda.is_available() and model.device.type == 'cuda'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ خطأ غير متوقع: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")

@app.post("/embed")
async def embed_image_alt(request: Request):
    """نسخة بديلة من endpoint الـ embedding"""
    return await embed_image(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
