"""
Vector store — FAISS with JSON sidecar for metadata.

Why FAISS instead of ChromaDB?
  ChromaDB requires building chroma-hnswlib from C++ source on Windows, which
  demands Visual Studio Build Tools. FAISS ships pre-built Python wheels on
  every major platform (Windows/Mac/Linux, py3.8–3.12) via PyPI — zero compile
  step, no extra system dependencies.

Architecture:
  - One FAISS IndexFlatIP index per bot (inner product on L2-normalised vectors
    = cosine similarity, but faster than IndexFlatL2)
  - A companion JSON file stores chunk texts + metadata in insertion order
  - Both files live under data/bots/{bot_id}/ — structural per-bot isolation

Trade-off vs ChromaDB:
  FAISS IndexFlatIP does a brute-force scan, which is O(n). For knowledge bases
  up to ~50k chunks (most documents), this is fast enough (<10ms). If you needed
  millions of chunks, you'd switch to IndexIVFFlat (approximate, faster).
"""

import os
import json
import numpy as np
from typing import List, Dict, Any

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

_BOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bots")
os.makedirs(_BOTS_DIR, exist_ok=True)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size


def _bot_dir(bot_id: str) -> str:
    path = os.path.join(_BOTS_DIR, bot_id)
    os.makedirs(path, exist_ok=True)
    return path


def _index_path(bot_id: str) -> str:
    return os.path.join(_bot_dir(bot_id), "index.faiss")


def _meta_path(bot_id: str) -> str:
    return os.path.join(_bot_dir(bot_id), "chunks.json")


def _load_index(bot_id: str):
    if not _FAISS_OK:
        raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu")
    path = _index_path(bot_id)
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)


def _normalise(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise so inner product == cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return vecs / norms


def upsert_chunks(
    bot_id: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    """Store chunk texts + L2-normalised embeddings for a bot."""
    if not _FAISS_OK:
        raise RuntimeError("faiss-cpu is not installed.")

    vecs = np.array(embeddings, dtype=np.float32)
    vecs = _normalise(vecs)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vecs)
    faiss.write_index(index, _index_path(bot_id))

    # Save chunk texts + metadata alongside the index
    with open(_meta_path(bot_id), "w", encoding="utf-8") as f:
        json.dump([{"text": c["text"], "metadata": c["metadata"]} for c in chunks], f)


def query_chunks(
    bot_id: str,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks by cosine similarity.
    Returns list of {text, metadata, distance} where distance is 1 - cosine_sim
    (lower = more similar, consistent with ChromaDB's convention).
    """
    index = _load_index(bot_id)
    if index is None or index.ntotal == 0:
        return []

    with open(_meta_path(bot_id), "r", encoding="utf-8") as f:
        chunk_store = json.load(f)

    q = np.array([query_embedding], dtype=np.float32)
    q = _normalise(q)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(q, k)  # scores = cosine similarities

    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunk_store[idx]
        # Convert cosine similarity to distance (1 - sim) so higher = less relevant
        hits.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "distance": float(1.0 - score),
        })

    return hits


def bot_exists(bot_id: str) -> bool:
    """Check whether a bot index exists and has at least one vector."""
    path = _index_path(bot_id)
    if not os.path.exists(path):
        return False
    if not _FAISS_OK:
        return True  # can't verify count, assume yes
    index = faiss.read_index(path)
    return index.ntotal > 0


def delete_bot(bot_id: str) -> None:
    """Remove all stored data for a bot."""
    import shutil
    bot_path = os.path.join(_BOTS_DIR, bot_id)
    if os.path.exists(bot_path):
        shutil.rmtree(bot_path)
