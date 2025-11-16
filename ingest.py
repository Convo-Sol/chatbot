# ingest.py
import uuid
from tqdm import tqdm
from embeddings import get_embedding
from utils import read_text_files, chunk_text
from retrieval import InMemoryVectorStore

from config import CHUNK_SIZE, CHUNK_OVERLAP, DB_PATH

def build_db():
    files = read_text_files("data")
    store = InMemoryVectorStore(DB_PATH)

    # Optionally clear existing DB: uncomment if you want a fresh DB
    # store.docs = []

    next_id = len(store.docs)
    for file in files:
        text = file["text"]
        fname = file["filename"]
        chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for idx, (chunk_text_, start, end) in enumerate(tqdm(chunks, desc=f"Embedding {fname}")):
            emb = get_embedding(chunk_text_)
            rec = {
                "id": f"{fname}::{idx}",
                "filename": fname,
                "chunk_index": idx,
                "text": chunk_text_,
                "start": start,
                "end": end,
                "embedding": emb
            }
            store.add(rec)

    store.persist()
    print(f"DB saved to {DB_PATH}. Total chunks: {len(store.docs)}")

if __name__ == "__main__":
    build_db()
