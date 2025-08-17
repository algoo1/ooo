import runpod
import torch
import requests
import logging
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time
import os

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# المسار الموحّد للنموذج (يجب أن يطابق Dockerfile)
CACHE_DIR = "/app/.cache/huggingface"

model = None
processor = None

def load_model():
    """تحميل النموذج مرة واحدة عند التشغيل"""
    global model, processor

    if model is not None:
        return True

    try:
        logger.info(f"🚀 بدء تحميل النموذج من {CACHE_DIR}")
        
        # تحميل النموذج والمعالج
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)

        if torch.cuda.is_available():
            model = model.cuda()
            model = model.half()  # تقليل الدقة لتوفير الذاكرة
            logger.info("✅ النموذج تم تحميله على GPU")
        else:
            logger.info("✅ النموذج تم تحميله على CPU")

        logger.info("🎉 النموذج جاهز للعمل")
        return True

    except Exception as e:
        logger.error(f"❌ فشل تحميل النموذج: {str(e)}")
        return False


def handler(job):
    """المعالج الرئيسي لطلبات RunPod"""
    try:
        job_input = job.get("input", {})
        image_url = job_input.get("image_url")
        text = job_input.get("text") or job_input.get("prompt")

        if not image_url and not text:
            return {
                "error": "يجب توفير image_url أو text على الأقل",
                "status": "failed"
            }

        # تحميل النموذج إذا لم يتم تحميله
        if not load_model():
            return {"error": "فشل في تحميل النموذج"}

        results = {}
        start_time = time.time()

        # معالجة الصورة
        if image_url:
            try:
                headers = {"User-Agent": "MyApp/1.0"}
                response = requests.get(image_url, headers=headers, timeout=15)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")

                inputs = processor(images=image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    if image_features.is_cuda:
                        image_features = image_features.cpu()
                    results["image_embedding"] = image_features[0].tolist()
                    results["image_embedding_size"] = image_features.shape[-1]

            except Exception as e:
                results["image_error"] = str(e)

        # معالجة النص
        if text:
            try:
                inputs = processor(text=[text], return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = model.get_text_features(**inputs)
                    if text_features.is_cuda:
                        text_features = text_features.cpu()
                    results["text_embedding"] = text_features[0].tolist()
                    results["text_embedding_size"] = text_features.shape[-1]

            except Exception as e:
                results["text_error"] = str(e)

        # وقت المعالجة
        results["processing_time"] = round(time.time() - start_time, 3)
        results["status"] = "success"

        return results

    except Exception as e:
        logger.error(f"❌ خطأ عام: {str(e)}")
        return {"error": str(e)}


# بدء الخدمة
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
