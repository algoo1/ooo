# CLIP Image Embedding API

Fast, GPU-accelerated CLIP embeddings API (HTTP) with FastAPI + Uvicorn.  
Works locally (Docker) or on RunPod (Container Image endpoint).

## Features
- ⚡ Fast cold start (preloads model on startup)
- 🧠 `openai/clip-vit-base-patch32`
- 🛡️ Clear errors (400 only for bad input)
- 🩺 `/health` endpoint for healthchecks
- 📦 Dockerfile pre-downloads model into image cache
- 🔁 Compatible with CPU/GPU

## Run (Docker Compose)

```bash
docker compose up --build
# API at http://localhost:8000
