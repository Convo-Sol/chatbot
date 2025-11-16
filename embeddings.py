# embeddings.py
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# Load the model once at module level (lazy loading)
_model = None

def _get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_embedding(text: str, model: str = None, task_type: str = "retrieval_document"):
    """
    Generate embedding for text using local sentence-transformers model.
    Returns vector as list[float].
    
    Args:
        text: Text to embed
        model: Model name (ignored, uses config model)
        task_type: "retrieval_document" for documents, "retrieval_query" for queries (both use same model)
    """
    try:
        embedding_model = _get_model()
        # sentence-transformers handles both documents and queries the same way
        embedding = embedding_model.encode(text, convert_to_numpy=True).tolist()
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to get embedding: {e}")
