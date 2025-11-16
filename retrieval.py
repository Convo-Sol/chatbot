# retrieval.py
import numpy as np
from utils import load_db, save_db
from config import DB_PATH, TOP_K

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class InMemoryVectorStore:
    """
    DB is a list of dicts:
    {
        "id": str,
        "filename": str,
        "chunk_index": int,
        "text": str,
        "embedding": list[float]
    }
    """
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.docs = load_db(db_path)

    def add(self, record):
        self.docs.append(record)

    def persist(self):
        save_db(self.docs, self.db_path)

    def search(self, query_embedding, top_k=TOP_K):
        # compute cosine for each doc
        scored = []
        for doc in self.docs:
            score = cosine_similarity(query_embedding, doc["embedding"])
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
