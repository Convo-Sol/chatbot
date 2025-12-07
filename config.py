# config.py
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Local sentence-transformers model
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")  # change if you want another model
TOP_K = int(os.getenv("TOP_K", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))      # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))# overlap characters
DB_PATH = "db/db.pkl"
