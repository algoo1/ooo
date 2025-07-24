from fastapi import FastAPI
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import requests
import io

app = FastAPI()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class Input(BaseModel):
    image_url: str

@app.post("/")
async def embed_image(input: Input):
    image_data = requests.get(input.image_url).content
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return {"embedding": features[0].tolist()}
