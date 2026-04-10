"""
GET /stats/{bot_id}

Returns aggregate analytics for a bot:
  - total_messages: how many /chat calls have been served
  - avg_latency_ms: mean time from request received to stream complete
  - estimated_cost_usd: sum of LLM token costs at GPT-4o-mini rates
  - unanswered_questions: chats where the bot said it couldn't find the answer
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import StatsResponse
from app.db.database import get_stats
from app.services import vector_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats/{bot_id}", response_model=StatsResponse)
async def stats(bot_id: str) -> StatsResponse:
    if not vector_store.bot_exists(bot_id):
        raise HTTPException(status_code=404, detail=f"No knowledge base found for bot_id '{bot_id}'.")

    data = await get_stats(bot_id)
    return StatsResponse(**data)
