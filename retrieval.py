# retrieval.py - Query Embedding & Search
# Uses Faiss index (memory-mapped) and Gemini API for query embeddings
import faiss
import numpy as np
import pickle
import time
import google.generativeai as genai
from config import TOP_K, GEMINI_API_KEY

# Configure and initialize Google GenAI client
genai.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# Load Faiss index (memory-mapped to avoid loading all data into RAM)
try:
    index = faiss.read_index('db/index.faiss', faiss.IO_FLAG_MMAP)
except Exception:
    index = faiss.read_index('db/index.faiss')

# Load chunk metadata
with open('db/chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

def get_query_embedding(query):
    """Get query embedding from Gemini API with retry logic."""
    for attempt in range(3):
        try:
            result = client.models.embed_content(
                model="models/text-embedding-004",
                content=query
            )
            # Extract embedding values
            if hasattr(result, 'embedding'):
                return result.embedding.values
            return result['embedding']
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                raise RuntimeError(f"Embedding API failed: {e}")

def retrieve_top_k(query, top_k=TOP_K):
    """Get the query embedding using Gemini and retrieve top-k chunks relevant to the query using Faiss search."""
    # Get query embedding from Gemini
    embedding = get_query_embedding(query)
    
    # Prepare query vector and normalize for inner-product search
    query_vec = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm
    
    # Search in Faiss index
    D, I = index.search(np.expand_dims(query_vec, axis=0), top_k)
    
    # Return top chunks with scores
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx != -1 and idx < len(chunks):
            results.append((float(score), chunks[idx]))
    
    return results
