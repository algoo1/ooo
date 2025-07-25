import runpod
import torch
import requests
import logging
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time
import os

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# متغيرات النموذج
model = None
processor = None
model_loaded = False

def load_model():
    """تحميل النموذج"""
    global model, processor, model_loaded
    
    if model_loaded:
        return True
    
    try:
        logger.info("🚀 Loading CLIP model...")
        start_time = time.time()
        
        cache_dir = "/runpod-volume"
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
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ Model loaded on GPU")
        
        model_loaded = True
        load_time = time.time() - start_time
        logger.info(f"✅ Model ready in {load_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        return False

def handler(job):
    """معالج RunPod"""
    try:
        logger.info(f"📥 Received job: {job}")
        
        # تحميل النموذج إذا لم يكن محملاً
        if not load_model():
            return {"error": "Model loading failed"}
        
        # استخراج البيانات
        job_input = job.get("input", {})
        image_url = job_input.get("image_url")
        
        if not image_url:
            logger.error("❌ No image_url provided in input. Example: {'input': {'image_url': 'https://...'}}")
            return {"error": "image_url required in input. Example: {'input': {'image_url': 'https://...'}}"}
        
        logger.info(f"🖼️ Processing: {image_url}")
        start_time = time.time()
        
        # تحميل الصورة
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return {"error": f"Image loading failed: {str(e)}"}
        
        # معالجة الصورة
        inputs = processor(images=image, return_tensors="pt")
        
        if torch.cuda.is_available() and model.device.type == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # استخراج embedding
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
            "embedding_size": len(embedding),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        return {"error": str(e)}

# تحميل النموذج عند البدء
logger.info("🔧 Initializing...")
load_model()

# بدء RunPod
runpod.serverless.start({"handler": handler})
