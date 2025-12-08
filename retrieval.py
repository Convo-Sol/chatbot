# retrieval.py - Query Embedding & Search
# Uses ChromaDB and Gemini API for query embeddings
import time
import google.generativeai as genai
import chromadb
from config import TOP_K, GEMINI_API_KEY

# Configure Google GenAI
genai.configure(api_key=GEMINI_API_KEY)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="db")
collection = client.get_collection(name="documents")

def get_query_embedding(query):
    """Get query embedding from Gemini API with retry logic."""
    for attempt in range(3):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query
            )
            # Extract embedding values
            if hasattr(result, 'embedding'):
                return result['embedding']
            return result['embedding']
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                raise RuntimeError(f"Embedding API failed: {e}")

def retrieve_top_k(query, top_k=TOP_K):
    """Get the query embedding using Gemini and retrieve top-k chunks relevant to the query using ChromaDB."""
    # Get query embedding from Gemini
    embedding = get_query_embedding(query)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    
    # Format results to match original return format
    formatted_results = []
    if results['documents'] and len(results['documents']) > 0:
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i] if results['distances'] else 0
            # Convert distance to similarity score (ChromaDB returns distances, lower is better)
            score = 1 - distance
            chunk = {
                'text': results['documents'][0][i],
                'filename': results['metadatas'][0][i].get('filename', ''),
                'chunk_index': results['metadatas'][0][i].get('chunk_index', 0)
            }
            formatted_results.append((float(score), chunk))
    
    return formatted_results
