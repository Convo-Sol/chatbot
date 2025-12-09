# retrieval.py - Query Embedding & Search
# Uses ChromaDB and local sentence-transformers for query embeddings (consistent with ingest.py)
import chromadb
from embeddings import get_embedding
from config import TOP_K

# Lazy-loaded ChromaDB client and collection
_client = None
_collection = None

def _get_collection():
    """Lazy load ChromaDB collection to avoid blocking startup."""
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path="db")
    if _collection is None:
        try:
            _collection = _client.get_collection(name="documents")
        except Exception as e:
            raise RuntimeError(f"Failed to load ChromaDB collection: {e}")
    return _collection

def retrieve_top_k(query, top_k=TOP_K):
    """Get the query embedding using local model and retrieve top-k chunks relevant to the query using ChromaDB."""
    # Get query embedding from local model
    embedding = get_embedding(query, task_type="retrieval_query")

    # Query ChromaDB
    collection = _get_collection()
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
