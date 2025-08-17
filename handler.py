import runpod
import torch
import requests
import logging
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time
import os
import json
from typing import Dict, Any, Optional

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model variables
model = None
processor = None
model_loaded = False

def load_model():
    """Load CLIP model with enhanced error handling"""
    global model, processor, model_loaded
    
    if model_loaded:
        return True
    
    try:
        logger.info("🚀 Loading CLIP model...")
        start_time = time.time()
        
        # Use multiple cache directories as fallback
        cache_dirs = ["/runpod-volume", "/app/.cache/huggingface", "./cache"]
        cache_dir = None
        
        for dir_path in cache_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(dir_path, "test_write")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                cache_dir = dir_path
                break
            except Exception as e:
                logger.warning(f"Cache directory {dir_path} not writable: {e}")
                continue
        
        if not cache_dir:
            logger.warning("No writable cache directory found, using default")
            cache_dir = None
        
        # Load model with appropriate dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info(f"Loading model to {device} with dtype {dtype}")
        
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            torch_dtype=dtype
        )
        
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        
        if device == "cuda":
            model = model.to(device)
            logger.info("✅ Model loaded on GPU")
        else:
            logger.info("✅ Model loaded on CPU")
        
        model_loaded = True
        load_time = time.time() - start_time
        logger.info(f"✅ Model ready in {load_time:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        return False

def validate_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize input data"""
    result = {
        "image_url": None,
        "text": None,
        "errors": []
    }
    
    # Extract image URL
    image_url = job_input.get("image_url") or job_input.get("image") or job_input.get("url")
    if image_url:
        if isinstance(image_url, str) and image_url.strip():
            if image_url.startswith(("http://", "https://")):
                result["image_url"] = image_url.strip()
            else:
                result["errors"].append("Image URL must start with http:// or https://")
        else:
            result["errors"].append("Image URL must be a non-empty string")
    
    # Extract text
    text = job_input.get("text") or job_input.get("prompt") or job_input.get("query")
    if text:
        if isinstance(text, str) and text.strip():
            result["text"] = text.strip()
        else:
            result["errors"].append("Text must be a non-empty string")
    
    # Check if at least one input is provided
    if not result["image_url"] and not result["text"]:
        result["errors"].append("At least one of 'image_url' or 'text' must be provided")
    
    return result

def download_image(image_url: str, timeout: int = 15, max_size_mb: int = 10) -> Optional[Image.Image]:
    """Download and validate image with better error handling"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; CLIP-API/1.0)",
            "Accept": "image/*"
        }
        
        logger.info(f"Downloading image from: {image_url}")
        response = requests.get(
            image_url, 
            headers=headers, 
            timeout=timeout,
            stream=True
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            raise ValueError(f"Invalid content type: {content_type}")
        
        # Check file size
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"Image too large: {size_mb:.2f}MB > {max_size_mb}MB")
        
        # Download content
        content = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content.write(chunk)
                if content.tell() > max_size_mb * 1024 * 1024:
                    raise ValueError(f"Image too large during download")
        
        content.seek(0)
        image = Image.open(content)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Validate image dimensions
        width, height = image.size
        if width < 10 or height < 10:
            raise ValueError(f"Image too small: {width}x{height}")
        if width > 4096 or height > 4096:
            logger.warning(f"Large image: {width}x{height}, processing may be slow")
        
        logger.info(f"✅ Image loaded successfully: {width}x{height}")
        return image
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        raise Exception(f"Image processing error: {str(e)}")

def get_image_embedding(image: Image.Image) -> list:
    """Generate image embedding with error handling"""
    try:
        inputs = processor(images=image, return_tensors="pt")
        
        # Move to device if using GPU
        if torch.cuda.is_available() and model.device.type == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            
            # Move back to CPU if needed
            if features.is_cuda:
                features = features.cpu()
            
            embedding = features[0].tolist()
        
        logger.info(f"✅ Image embedding generated: {len(embedding)} dimensions")
        return embedding
        
    except Exception as e:
        raise Exception(f"Image embedding generation failed: {str(e)}")

def get_text_embedding(text: str) -> list:
    """Generate text embedding with error handling"""
    try:
        inputs = processor(text=[text], return_tensors="pt", truncation=True, max_length=77)
        
        # Move to device if using GPU
        if torch.cuda.is_available() and model.device.type == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            
            # Move back to CPU if needed
            if features.is_cuda:
                features = features.cpu()
            
            embedding = features[0].tolist()
        
        logger.info(f"✅ Text embedding generated: {len(embedding)} dimensions")
        return embedding
        
    except Exception as e:
        raise Exception(f"Text embedding generation failed: {str(e)}")

def handler(job):
    """Enhanced RunPod handler with comprehensive error handling"""
    start_time = time.time()
    
    try:
        logger.info(f"📥 Received job: {type(job)} - {str(job)[:200]}...")
        
        # Load model if not already loaded
        if not load_model():
            return {
                "error": "Model loading failed",
                "status": "error",
                "processing_time": round(time.time() - start_time, 3)
            }
        
        # Extract and validate input
        job_input = job.get("input", {})
        if not job_input:
            return {
                "error": "No input provided. Expected format: {'input': {'image_url': '...', 'text': '...'}}",
                "status": "error",
                "processing_time": round(time.time() - start_time, 3)
            }
        
        validation = validate_input(job_input)
        if validation["errors"]:
            return {
                "error": "; ".join(validation["errors"]),
                "status": "error",
                "processing_time": round(time.time() - start_time, 3)
            }
        
        results = {
            "status": "success",
            "embeddings": {}
        }
        
        # Process image if provided
        if validation["image_url"]:
            try:
                image = download_image(validation["image_url"])
                image_embedding = get_image_embedding(image)
                results["embeddings"]["image"] = {
                    "embedding": image_embedding,
                    "size": len(image_embedding),
                    "dimensions": image.size
                }
            except Exception as e:
                results["embeddings"]["image"] = {"error": str(e)}
                logger.error(f"Image processing failed: {e}")
        
        # Process text if provided
        if validation["text"]:
            try:
                text_embedding = get_text_embedding(validation["text"])
                results["embeddings"]["text"] = {
                    "embedding": text_embedding,
                    "size": len(text_embedding),
                    "input_length": len(validation["text"])
                }
            except Exception as e:
                results["embeddings"]["text"] = {"error": str(e)}
                logger.error(f"Text processing failed: {e}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results["processing_time"] = round(processing_time, 3)
        
        # Check if any embeddings were successfully generated
        successful_embeddings = [
            k for k, v in results["embeddings"].items() 
            if isinstance(v, dict) and "embedding" in v
        ]
        
        if not successful_embeddings:
            results["status"] = "error"
            results["error"] = "No embeddings could be generated"
        
        logger.info(f"✅ Job completed in {processing_time:.3f}s - Status: {results['status']}")
        return results
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Handler error: {str(e)}")
        return {
            "error": f"Handler error: {str(e)}",
            "status": "error",
            "processing_time": round(processing_time, 3)
        }

# Health check function for debugging
def health_check():
    """Health check function"""
    try:
        if not load_model():
            return {"status": "unhealthy", "reason": "model_not_loaded"}
        
        # Test with simple text
        test_result = get_text_embedding("test")
        
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "device": str(model.device) if model else "unknown",
            "embedding_size": len(test_result) if test_result else 0
        }
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

# Initialize model on startup
logger.info("🔧 Initializing CLIP API...")
load_model()

# Start RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
