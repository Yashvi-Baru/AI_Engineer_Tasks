from pydantic import BaseModel, AnyHttpUrl, field_validator
from typing import Optional, List
import uuid


class UploadRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    source_name: Optional[str] = None

    @field_validator("url", "text", mode="before")
    @classmethod
    def at_least_one(cls, v):
        return v

    def model_post_init(self, __context):
        if not self.text and not self.url:
            raise ValueError("Provide either 'text' or 'url' — one of them must be present.")


class UploadResponse(BaseModel):
    bot_id: str
    chunks_stored: int
    source: str


class ConversationMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    bot_id: str
    user_message: str
    conversation_history: Optional[List[ConversationMessage]] = []


class StatsResponse(BaseModel):
    bot_id: str
    total_messages: int
    avg_latency_ms: float
    estimated_cost_usd: float
    unanswered_questions: int
