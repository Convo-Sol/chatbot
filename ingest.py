# ingest.py - Offline Ingestion
# Run this locally to build the Faiss index. DO NOT run on Render.
import sys
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from utils import read_text_files, chunk_text
from config import CHUNK_SIZE, CHUNK_OVERLAP

def build_db():
    """Build Faiss index offline with local sentence-transformers model."""
    # Read and chunk documents
    files = read_text_files("data")
    chunks = []
    
    for file in files:
        text = file["text"]
        fname = file["filename"]
        chunk_list = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        
        for idx, (chunk_text_, start, end) in enumerate(chunk_list):
            chunks.append({
                "text": chunk_text_,
                "filename": fname,
                "chunk_index": idx
            })
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Load embedding model offline
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract text for embedding
    chunk_texts = [c["text"] for c in chunks]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(chunk_texts, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)  # Use float32 to save memory
    
    # Build Faiss index (Inner-Product for cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Save index and metadata
    os.makedirs('db', exist_ok=True)
    faiss.write_index(index, 'db/index.faiss')
    
    with open('db/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Ingestion complete. Saved db/index.faiss and db/chunks.pkl")
    print(f"Total chunks: {len(chunks)}, Embedding dimension: {dim}")

if __name__ == "__main__":
    build_db()
