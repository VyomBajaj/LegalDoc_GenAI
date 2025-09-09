# modules/embedding_store.py
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import uuid
import numpy as np

from pinecone import Pinecone  # ServerlessSpec not needed as index already exists

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

INDEX_DIR = Path("vectorstore/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "doc-embeddings")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
USE_PINECONE = bool(PINECONE_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY) if USE_PINECONE else None

# --- Caching the embedding model ---
_EMBEDDING_MODEL_INSTANCE = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_embedding_model():
    """Return cached embedding model instance."""
    return _EMBEDDING_MODEL_INSTANCE

# ---- Existing FAISS helpers ----
def build_faiss_from_chunks(chunks, metadatas=None, index_path: str = str(INDEX_DIR / "faiss_index")):
    embedding = get_embedding_model()
    docs = []
    for i, chunk in enumerate(chunks):
        meta = metadatas[i] if metadatas and i < len(metadatas) else {"chunk_id": i}
        docs.append(Document(page_content=chunk, metadata=meta))
    db = FAISS.from_documents(docs, embedding)
    db.save_local(index_path)
    return db

def load_faiss(index_path: str = str(INDEX_DIR / "faiss_index")):
    """Load the FAISS index."""
    embedding = get_embedding_model()
    idx_path = Path(index_path)
    if not idx_path.exists():
        raise FileNotFoundError(f"No index at {index_path}")
    db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    return db

# ---- Pinecone helpers ----
def init_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    if not USE_PINECONE:
        raise RuntimeError("Pinecone is not configured. Set PINECONE_API_KEY in .env")
    return pc.Index(index_name)

def _normalize_vector(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()

def upsert_chunks_to_pinecone(chunks, metadatas=None, index_name: str = PINECONE_INDEX_NAME, namespace: str = None, batch_size: int = 100):
    if not USE_PINECONE:
        raise RuntimeError("Pinecone not enabled in environment variables.")

    embedder = get_embedding_model()
    embeddings = embedder.embed_documents(chunks)

    idx = init_pinecone_index(index_name=index_name)

    vectors_to_upsert = []
    returned_ids = []
    for i, emb in enumerate(embeddings):
        chunk_id = str(uuid.uuid4())
        if metadatas and i < len(metadatas):
            chunk_id = str(metadatas[i].get("chunk_id", chunk_id))

        metadata = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
        metadata["text"] = chunks[i]
        metadata["conversation_id"] = namespace

        vec = _normalize_vector(emb)
        vectors_to_upsert.append((chunk_id, vec, metadata))
        returned_ids.append(chunk_id)

        if len(vectors_to_upsert) >= batch_size:
            idx.upsert(vectors=vectors_to_upsert, namespace=namespace)
            vectors_to_upsert = []

    if vectors_to_upsert:
        idx.upsert(vectors=vectors_to_upsert, namespace=namespace)

    return returned_ids

def query_pinecone(query: str, top_k: int = 4, index_name: str = PINECONE_INDEX_NAME, namespace: str = None):
    if not USE_PINECONE:
        raise RuntimeError("Pinecone not enabled in environment variables.")

    embedder = get_embedding_model()
    qvec = embedder.embed_query(query) if hasattr(embedder, "embed_query") else embedder.embed_documents([query])[0]
    qvec = _normalize_vector(qvec)

    idx = init_pinecone_index(index_name=index_name)
    res = idx.query(vector=qvec, top_k=top_k, include_metadata=True, namespace=namespace)
    matches = res.matches

    docs = []
    for m in matches:
        metadata = m.metadata or {}
        text = metadata.get("text", "")
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def delete_namespace_from_pinecone(index_name: str = PINECONE_INDEX_NAME, namespace: str = None):
    if not USE_PINECONE:
        raise RuntimeError("Pinecone not enabled.")
    idx = init_pinecone_index(index_name=index_name)
    if namespace:
        idx.delete(delete_all=True, namespace=namespace)
