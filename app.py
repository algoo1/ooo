from fastapi import FastAPI, UploadFile, File
from jina_clip import JinaCLIP
from PIL import Image
import io

model = JinaCLIP('ViT-B/32')  # أسرع نسخة
app = FastAPI()

@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = model.encode_image(image).tolist()
    return {"embedding": embedding}
