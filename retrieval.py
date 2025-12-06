# retrieval.py - Query Embedding & Search
# Uses Faiss index (memory-mapped) and Gemini API for query embeddings
import faiss
import numpy as np
import pickle
import time
from google import genai
from config import TOP_K

# Initialize Google GenAI client
client = genai.Client()

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
    "."""
tsurn resulet 
    r   t(idx)]))
ks[ine), chun(scorfloatts.append((      resul):
      chunksd idx < len( -1 andx !=        if i], I[0]):
zip(D[0 in dxre, i
    for sco[]ults =  ress
    with scorekschunop n t Retur 
    #    top_k)
c, axis=0),ms(query_vep.expand_di(nx.searchindeI = ex
    D, in Faiss indSearch   
    #   = norm
vec /    query_m > 0:
        if norquery_vec)
orm(np.linalg.n   norm = oat32)
 p.fl=ndtypembedding, .array(enp_vec = 
    queryct searchoduor inner-prnormalize for and re vect Prepa 
    #
   g(query)eddiny_embet_quering = g  embeddmini
  ing from Ge embeddet query  # G   searchs and Faissi embeddingng Geminusiery he qulevant to ts retop-k chunk"Get "
