from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import requests
from io import BytesIO

app = FastAPI()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class ImageInput(BaseModel):
    image_url: str

@app.post("/embed-image")
def embed_image(input: ImageInput):
    image = Image.open(BytesIO(requests.get(input.image_url).content)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return {"embedding": features[0].tolist()}
