import os
import time
import logging
from io import BytesIO
from typing import Optional, Dict, Any

import torch
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyHttpUrl, ValidationError
from transformers import CLIPProcessor, CLIPModel

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clip-embedding-api")

# -------------------- FastAPI --------------------
app = FastAPI(title="CLIP Embedding API", version="1.0.0")

# -------------------- Globals --------------------
model: Optional[CLIPModel] = None
processor: Optional[CLIPProcessor] = None
model_loaded: bool = False

CACHE_DIR = os.getenv("HF_HOME", "/app/.cache/huggingface")
TORCH_HOME = os.getenv("TORCH_HOME", "/app/.cache/torch")
MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TORCH_HOME, exist_ok=True)

# -------------------- Schemas --------------------
class InputInner(BaseModel):
    image_url: Optional[AnyHttpUrl] = None
    text: Optional[str] = None

class EmbedRequest(BaseModel):
    """
    يدعم شكلين:
    1) {"image_url": "...", "text": "..."}
    2) {"input": {"image_url": "...", "text": "..."}}
    """
    image_url: Optional[AnyHttpUrl] = None
    text: Optional[str] = None
    input: Optional[InputInner] = None

    def normalized(self) -> InputInner:
        if self.input is not None:
            return self.input
        return InputInner(image_url=self.image_url, text=self.text)

# -------------------- Utils --------------------
def load_model() -> None:
    global model, processor, model_loaded
    if model_loaded:
        return

    try:
        logger.info("🚀 Loading CLIP model '%s' ...", MODEL_ID)
        t0 = time.time()

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = CLIPModel.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=dtype,
        )
        processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ Model loaded on GPU")
        else:
            logger.info("✅ Model loaded on CPU")

        model_loaded = True
        logger.info("✅ Model ready in %.2fs", time.time() - t0)
    except Exception as e:
        logger.exception("❌ Failed to load model: %s", e)
        raise

def _download_image(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ClipAPI/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

# -------------------- Startup --------------------
@app.on_event("startup")
def _startup():
    preload = os.getenv("PRELOAD_MODEL", "0") == "1"
    if preload:
        load_model()

# -------------------- Endpoints --------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_id": MODEL_ID,
        "cache_dir": CACHE_DIR,
    }

@app.post("/embed")
def embed(req: EmbedRequest) -> Dict[str, Any]:
    try:
        data = req.normalized()
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if data.image_url is None and (data.text is None or data.text.strip() == ""):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: image_url or text.",
        )

    load_model()

    results: Dict[str, Any] = {}
    t0 = time.time()

    # Image path
    if data.image_url:
        try:
            image = _download_image(str(data.image_url))
            inputs_image = processor(images=image, return_tensors="pt")
            if torch.cuda.is_available() and model is not None and next(model.parameters()).is_cuda:
                inputs_image = {k: v.cuda() for k, v in inputs_image.items()}
            with torch.no_grad():
                features_image = model.get_image_features(**inputs_image)
                if features_image.is_cuda:
                    features_image = features_image.cpu()
                emb = features_image[0].tolist()
            results["image_embedding"] = emb
            results["image_embedding_size"] = len(emb)
        except Exception as e:
            results["image_error"] = f"Image processing failed: {e}"

    # Text path
    if data.text and data.text.strip():
        try:
            inputs_text = processor(text=[data.text], return_tensors="pt")
            if torch.cuda.is_available() and model is not None and next(model.parameters()).is_cuda:
                inputs_text = {k: v.cuda() for k, v in inputs_text.items()}
            with torch.no_grad():
                features_text = model.get_text_features(**inputs_text)
                if features_text.is_cuda:
                    features_text = features_text.cpu()
                emb = features_text[0].tolist()
            results["text_embedding"] = emb
            results["text_embedding_size"] = len(emb)
        except Exception as e:
            results["text_error"] = f"Text processing failed: {e}"

    results["processing_time"] = round(time.time() - t0, 3)
    results["status"] = "success" if ("image_embedding" in results or "text_embedding" in results) else "failed"

    return results
