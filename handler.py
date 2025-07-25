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
    """معالج RunPod يدعم الصورة والنص"""
    try:
        logger.info(f"📥 Received job: {job}")
        
        # تحميل النموذج إذا لم يكن محملاً
        if not load_model():
            return {"error": "Model loading failed"}
        
        # استخراج البيانات
        job_input = job.get("input", {})
        image_url = job_input.get("image_url")
        text = job_input.get("text") or job_input.get("prompt")
        
        results = {}
        start_time = time.time()
        
        # معالجة الصورة إذا وجدت
        if image_url:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; MyApp/1.0; +https://example.com/)"
                }
                response = requests.get(image_url, headers=headers, timeout=15)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                results["image_error"] = f"Image loading failed: {str(e)}"
            else:
                try:
                    inputs_image = processor(images=image, return_tensors="pt")
                    if torch.cuda.is_available() and model.device.type == 'cuda':
                        inputs_image = {k: v.cuda() for k, v in inputs_image.items()}
                    with torch.no_grad():
                        features_image = model.get_image_features(**inputs_image)
                        if features_image.is_cuda:
                            features_image = features_image.cpu()
                        embedding_image = features_image[0].tolist()
                    results["image_embedding"] = embedding_image
                    results["image_embedding_size"] = len(embedding_image)
                except Exception as e:
                    results["image_error"] = f"Image embedding failed: {str(e)}"
        
        # معالجة النص إذا وجد
        if text:
            try:
                inputs_text = processor(text=[text], return_tensors="pt")
                if torch.cuda.is_available() and model.device.type == 'cuda':
                    inputs_text = {k: v.cuda() for k, v in inputs_text.items()}
                with torch.no_grad():
                    features_text = model.get_text_features(**inputs_text)
                    if features_text.is_cuda:
                        features_text = features_text.cpu()
                    embedding_text = features_text[0].tolist()
                results["text_embedding"] = embedding_text
                results["text_embedding_size"] = len(embedding_text)
            except Exception as e:
                results["text_error"] = f"Text embedding failed: {str(e)}"
        
        processing_time = time.time() - start_time
        results["processing_time"] = round(processing_time, 3)
        results["status"] = "success" if ("image_embedding" in results or "text_embedding" in results) else "failed"
        
        if not ("image_embedding" in results or "text_embedding" in results):
            results["error"] = "No valid image_url or text provided in input. Example: {'input': {'image_url': '...', 'text': '...'}}"
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        return {"error": str(e)}

# تحميل النموذج عند البدء
logger.info("🔧 Initializing...")
load_model()

# بدء RunPod
runpod.serverless.start({"handler": handler})
