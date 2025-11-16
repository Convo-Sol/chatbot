# utils.py
import os
import re
import pickle

def read_text_files(folder="data"):
    files = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                files.append({"filename": fname, "text": f.read()})
    return files

def chunk_text(text, size=800, overlap=200):
    """
    Chunk by characters with overlap (simple, robust).
    Yields list of (chunk_text, start_index, end_index)
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + size
        chunk = text[start: end]
        chunks.append((chunk, start, min(end, length)))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def save_db(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_db(path):
    import pickle, os
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)
