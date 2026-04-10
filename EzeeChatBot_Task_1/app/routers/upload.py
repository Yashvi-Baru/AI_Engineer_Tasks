"""
POST /upload

Accepts a text payload or a URL, ingests and chunks the content,
generates embeddings, and stores them in a per-bot ChromaDB collection.
Returns a bot_id you can use to chat against this knowledge base.
"""

import uuid
import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import UploadRequest, UploadResponse
from app.services.ingestion import fetch_url, clean_text
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts
from app.services import vector_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload(payload: UploadRequest) -> UploadResponse:
    bot_id = str(uuid.uuid4())

    # --- 1. Fetch / receive raw content ---
    if payload.url:
        source_label = payload.source_name or payload.url
        try:
            raw_text = await fetch_url(payload.url)
        except Exception as exc:
            logger.error("URL fetch failed: %s", exc)
            raise HTTPException(status_code=422, detail=f"Could not fetch URL: {exc}")
        source_type = "url"
    else:
        raw_text = payload.text
        source_label = payload.source_name or "raw_text"
        source_type = "text"

    # --- 2. Clean ---
    cleaned = clean_text(raw_text)
    if len(cleaned) < 50:
        raise HTTPException(
            status_code=422,
            detail="The content is too short to build a useful knowledge base (< 50 chars after cleaning).",
        )

    # --- 3. Chunk ---
    chunks = chunk_text(cleaned, source=source_label)
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks could be extracted from the content.")

    logger.info("Bot %s: %d chunks from %s", bot_id, len(chunks), source_label)

    # --- 4. Embed ---
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # --- 5. Store ---
    vector_store.upsert_chunks(bot_id, chunks, embeddings)

    return UploadResponse(
        bot_id=bot_id,
        chunks_stored=len(chunks),
        source=source_type,
    )
