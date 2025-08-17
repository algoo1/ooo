import runpod
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
processor = None

def load_model():
    global model, processor
    try:
        logger.info("Loading CLIP model...")
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
            
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

def handler(job):
    try:
        # Load model if needed
        if model is None:
            if not load_model():
                return {"error": "Model loading failed"}
        
        # Get input
        job_input = job.get("input", {})
        image_url = job_input.get("image_url")
        text = job_input.get("text")
        
        if not image_url and not text:
            return {
                "error": "Please provide 'image_url' or 'text' in input",
                "example": {
                    "input": {
                        "text": "a cat",
                        "image_url": "https://example.com/image.jpg"
                    }
                }
            }
        
        results = {}
        
        # Process text
        if text:
            try:
                inputs = processor(text=[text], return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = model.get_text_features(**inputs)
                    if features.is_cuda:
                        features = features.cpu()
                    
                results["text_embedding"] = features[0].tolist()
            except Exception as e:
                results["text_error"] = str(e)
        
        # Process image
        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                inputs = processor(images=image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                    if features.is_cuda:
                        features = features.cpu()
                        
                results["image_embedding"] = features[0].tolist()
            except Exception as e:
                results["image_error"] = str(e)
        
        return results
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}

# Pre-load model
logger.info("Initializing...")
load_model()

# Start RunPod
runpod.serverless.start({"handler": handler})
