from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the model (loaded once)
model = None

def load_model():
    """Load the model once and keep it in memory"""
    global model
    if model is None:
        logger.info("Loading SentenceTransformer model...")
        start_time = time.time()
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    return model

# Load model on startup
load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/encode', methods=['POST'])
def encode_sentences():
    """Encode sentences to embeddings"""
    try:
        data = request.get_json()
        
        if not data or 'sentences' not in data:
            return jsonify({"error": "Missing 'sentences' in request body"}), 400
        
        sentences = data['sentences']
        if not isinstance(sentences, list):
            sentences = [sentences]
        
        start_time = time.time()
        
        # Get embeddings
        embeddings = model.encode(sentences)
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "embeddings": embeddings_list,
            "processing_time": f"{processing_time:.3f}s",
            "num_sentences": len(sentences)
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/similarity', methods=['POST'])
def calculate_similarity():
    """Calculate similarity between sentences"""
    try:
        data = request.get_json()
        
        if not data or 'sentences1' not in data or 'sentences2' not in data:
            return jsonify({"error": "Missing 'sentences1' or 'sentences2' in request body"}), 400
        
        sentences1 = data['sentences1']
        sentences2 = data['sentences2']
        
        if not isinstance(sentences1, list):
            sentences1 = [sentences1]
        if not isinstance(sentences2, list):
            sentences2 = [sentences2]
        
        start_time = time.time()
        
        # Get embeddings
        embeddings1 = model.encode(sentences1)
        embeddings2 = model.encode(sentences2)
        
        # Calculate cosine similarity
        similarities = model.similarity(embeddings1, embeddings2)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "similarities": similarities.tolist(),
            "processing_time": f"{processing_time:.3f}s"
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "SentenceTransformers API is running",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "endpoints": {
            "/encode": "POST - Encode sentences to embeddings",
            "/similarity": "POST - Calculate similarity between sentences",
            "/health": "GET - Health check"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
