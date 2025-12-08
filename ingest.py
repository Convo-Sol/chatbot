# ingest.py - Offline Ingestion
# Run this locally to build the ChromaDB collection.
import sys
import os
from sentence_transformers import SentenceTransformer
import chromadb
from utils import read_text_files, chunk_text
from config import CHUNK_SIZE, CHUNK_OVERLAP

def build_db():
    """Build ChromaDB collection with local sentence-transformers model."""
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
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="db")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name="documents")
    except:
        pass
    
    # Create collection with sentence-transformers embedding function
    collection = client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Prepare data for ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"filename": c["filename"], "chunk_index": c["chunk_index"]} for c in chunks]
    
    # Load embedding model and generate embeddings
    print("Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.tolist()
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    
    print(f"Ingestion complete. ChromaDB collection saved in db/")
    print(f"Total chunks: {len(chunks)}")

if __name__ == "__main__":
    build_db()
