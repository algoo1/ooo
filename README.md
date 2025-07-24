# CLIP Image Embedding API

🚀 Fast and optimized CLIP image embedding API built for RunPod serverless deployment.

## Features

- ⚡ **Ultra-fast cold start** (< 10 seconds)
- 🎮 **GPU acceleration** support
- 🛡️ **Robust error handling** 
- 📊 **Health monitoring** endpoints
- 🔄 **Model pre-loading** in Docker image
- 📝 **Detailed logging** for debugging

## Quick Deploy to RunPod

### Method 1: Direct GitHub Integration

1. **Create RunPod Endpoint**
   - Go to RunPod → Endpoints → New Endpoint
   - Choose "Container Image" 
   - Set Repository: `https://github.com/YOUR_USERNAME/clip-embedding-api`
   - Container Start Command: `python app.py`

### Method 2: Pre-built Docker Image

```bash
# Build and push to Docker Hub
docker build -t your-username/clip-embedding-api .
docker push your-username/clip-embedding-api
