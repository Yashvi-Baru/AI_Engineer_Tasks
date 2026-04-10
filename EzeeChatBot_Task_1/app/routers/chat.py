"""
POST /chat

Accepts { bot_id, user_message, conversation_history }.

Pipeline:
  1. Embed the user's question
  2. Retrieve top-5 most similar chunks from the bot's knowledge base
  3. If no chunks are close enough, return fallback immediately (no LLM call)
  4. Build a grounding-only system prompt and pipe it to GPT-4o-mini
  5. Stream response tokens back as Server-Sent Events (SSE)
  6. Detect the EZEE_NO_ANSWER sentinel and swap it for a friendly fallback
  7. Persist latency + cost + unanswered flag to SQLite

Streaming format:
  data: {"token": "Hello"}
  data: {"token": " world"}
  data: [DONE]
"""

import json
import time
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest
from app.services.embedder import embed_query
from app.services import vector_store
from app.services.llm import (
    stream_chat,
    estimate_cost,
    NO_ANSWER_SENTINEL,
    FALLBACK_MESSAGE,
)
from app.db.database import record_chat_event

logger = logging.getLogger(__name__)
router = APIRouter()

# Cosine distance threshold (1 - cosine_similarity).
# distance=0 means identical, distance=1 means orthogonal, >1 means opposite.
# 0.50 cuts off anything with < 50% cosine similarity — reliably filters noise.
RELEVANCE_THRESHOLD = 0.50


@router.post("/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    if not vector_store.bot_exists(payload.bot_id):
        raise HTTPException(
            status_code=404,
            detail=f"No knowledge base found for bot_id '{payload.bot_id}'. Upload a document first.",
        )

    start_time = time.perf_counter()

    # --- 1. Embed query ---
    query_vec = embed_query(payload.user_message)

    # --- 2. Retrieve relevant chunks ---
    hits = vector_store.query_chunks(payload.bot_id, query_vec, top_k=5)

    # --- 3. Check relevance — if nothing is close enough, short-circuit ---
    relevant = [h for h in hits if h["distance"] <= RELEVANCE_THRESHOLD]
    if not relevant:
        async def fallback_stream():
            # Still log this as an unanswered event
            latency = (time.perf_counter() - start_time) * 1000
            await record_chat_event(
                bot_id=payload.bot_id,
                latency_ms=latency,
                cost_usd=0.0,
                was_unanswered=True,
            )
            yield f"data: {json.dumps({'token': FALLBACK_MESSAGE})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fallback_stream(), media_type="text/event-stream")

    # --- 4 & 5. Stream LLM response ---
    history = [{"role": m.role, "content": m.content} for m in (payload.conversation_history or [])]

    return StreamingResponse(
        _stream_and_record(
            bot_id=payload.bot_id,
            context_chunks=relevant,
            conversation_history=history,
            user_message=payload.user_message,
            start_time=start_time,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if deployed behind it
        },
    )


async def _stream_and_record(
    bot_id: str,
    context_chunks: list,
    conversation_history: list,
    user_message: str,
    start_time: float,
) -> AsyncGenerator[str, None]:
    """
    Internal generator: streams tokens to the client, detects the no-answer
    sentinel, and records analytics when the stream ends.
    """
    accumulated = []
    usage = None
    was_unanswered = False

    try:
        sentinel_triggered = False
        async for token, usage_data in stream_chat(context_chunks, conversation_history, user_message):
            if usage_data is not None:
                usage = usage_data
                continue  # usage chunk carries no token content

            if sentinel_triggered:
                # Discard any tokens that arrive after the sentinel
                continue

            accumulated.append(token)
            full_so_far = "".join(accumulated)

            if NO_ANSWER_SENTINEL in full_so_far:
                # Swap sentinel for friendly fallback — never let the model's
                # raw sentinel string reach the client
                sentinel_triggered = True
                was_unanswered = True
                yield f"data: {json.dumps({'token': FALLBACK_MESSAGE})}\n\n"
                continue

            yield f"data: {json.dumps({'token': token})}\n\n"

    except Exception as exc:
        logger.error("LLM stream error: %s", exc)
        error_msg = "Something went wrong while generating a response. Please try again."
        yield f"data: {json.dumps({'token': error_msg})}\n\n"

    yield "data: [DONE]\n\n"

    # --- 6. Record analytics ---
    latency_ms = (time.perf_counter() - start_time) * 1000
    cost = 0.0
    if usage:
        cost = estimate_cost(usage["prompt_tokens"], usage["completion_tokens"])

    try:
        await record_chat_event(
            bot_id=bot_id,
            latency_ms=latency_ms,
            cost_usd=cost,
            was_unanswered=was_unanswered,
        )
    except Exception as exc:
        logger.warning("Failed to record chat event: %s", exc)
