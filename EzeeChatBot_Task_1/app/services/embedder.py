"""
Embedder — thin singleton wrapper around sentence-transformers.

We use `all-MiniLM-L6-v2` because:
  - 384-dim vectors: small enough to be fast at query time, rich enough for semantic retrieval
  - Runs fully offline — no API key, no rate limits, no cost
  - Widely benchmarked; beats much larger models on most retrieval tasks

The model is loaded once at startup and reused for every embed call to avoid
paying the 1–2s model load cost on every request.
"""

from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model exactly once (cached by lru_cache)."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a list of embedding vectors for the given texts."""
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
