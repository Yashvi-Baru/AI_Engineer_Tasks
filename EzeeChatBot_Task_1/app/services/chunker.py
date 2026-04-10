"""
Semantic sentence-boundary chunker.

Strategy
--------
Split raw text into sentences using NLTK (handles abbreviations, punctuation,
multi-sentence edge cases). Then greedily pack sentences into chunks that stay
under a target character budget (~1600 chars ≈ 400 tokens at ~4 chars/token).
A one-sentence overlap between consecutive chunks preserves context that would
otherwise be lost right at a chunk boundary.

Why not character splits?
  A naive 500-char cut lands mid-sentence, mid-idea. Retrieval then returns
  fragments with no subject or dangling predicates — garbage in, garbage out.
  Sentence-boundary chunking keeps each chunk semantically self-contained.

Why overlap?
  If an answer spans two sentences that straddle a chunk boundary, overlap
  ensures at least one chunk contains both sentences.
"""

import re
from typing import List, Dict, Any


# Try NLTK for high-quality sentence splitting; fall back gracefully
try:
    import nltk
    # Download quietly — only runs once, subsequent calls are no-ops
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    _USE_NLTK = True
except Exception:
    _USE_NLTK = False


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK or a regex fallback."""
    if _USE_NLTK:
        return sent_tokenize(text)

    # Regex fallback: split on '. ', '! ', '? ' followed by a capital letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str = "unknown",
    target_chars: int = 1600,
    overlap_sentences: int = 1,
) -> List[Dict[str, Any]]:
    """
    Break `text` into overlapping, sentence-aligned chunks.

    Each returned chunk dict has:
      - text: the chunk content
      - metadata: source, chunk_index, char_start, char_end, sentence_count
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Dict[str, Any]] = []
    current_sentences: List[str] = []
    current_len = 0
    char_cursor = 0

    def flush(sentences_batch: List[str], start: int) -> None:
        body = " ".join(sentences_batch)
        end = start + len(body)
        chunks.append(
            {
                "text": body,
                "metadata": {
                    "source": source,
                    "chunk_index": len(chunks),
                    "char_start": start,
                    "char_end": end,
                    "sentence_count": len(sentences_batch),
                },
            }
        )

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sent_len = len(sentence) + 1  # +1 for the joining space

        if current_sentences and current_len + sent_len > target_chars:
            # Flush current batch
            flush(current_sentences, char_cursor)

            # Advance char_cursor to end of flushed chunk
            flushed_text = " ".join(current_sentences)
            char_cursor += len(flushed_text) + 1

            # Keep `overlap_sentences` sentences as context seed for next chunk
            carry = current_sentences[-overlap_sentences:] if overlap_sentences else []
            current_sentences = carry
            current_len = sum(len(s) + 1 for s in carry)

            # Don't advance i — re-evaluate current sentence in new context
            continue

        current_sentences.append(sentence)
        current_len += sent_len
        i += 1

    # Flush remaining sentences
    if current_sentences:
        flush(current_sentences, char_cursor)

    return chunks
