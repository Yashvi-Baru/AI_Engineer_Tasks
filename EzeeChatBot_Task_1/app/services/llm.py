"""
LLM service — OpenAI streaming chat with hard grounding constraints.

Hallucination handling:
  The system prompt draws a clear line: answer ONLY from the context I give you.
  If the context doesn't contain the answer, output the exact string EZEE_NO_ANSWER.
  The chat router watches for this sentinel and replaces it with a friendly
  fallback message before sending it to the user — while also incrementing the
  unanswered_questions stat so we can measure knowledge-base coverage over time.

Token cost estimation:
  We track prompt + completion tokens from the usage field of the final chunk
  and price them at GPT-4o-mini rates (Apr 2025):
    $0.150 / 1M input tokens, $0.600 / 1M output tokens
"""

import os
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

from openai import AsyncOpenAI

MODEL = "gpt-4o-mini"

# Pricing as of Apr 2025 (USD per token)
COST_PER_INPUT_TOKEN = 0.150 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.600 / 1_000_000

# Sentinel the model outputs when it can't answer from context
NO_ANSWER_SENTINEL = "EZEE_NO_ANSWER"

FALLBACK_MESSAGE = (
    "I couldn't find anything in the uploaded document that answers your question. "
    "Could you try rephrasing, or check if this topic is covered in the knowledge base?"
)

_SYSTEM_TEMPLATE = """You are EzeeChatBot — a focused assistant that answers questions ONLY based on the context provided below.

Rules you must follow without exception:
1. Use ONLY the information in the [CONTEXT] section to answer.
2. Do NOT use any external knowledge, assumptions, or training data to fill gaps.
3. If the context does not contain the answer, respond with exactly: {sentinel}
4. Do not apologise or explain why you can't answer — just output {sentinel}.
5. Keep answers concise and factual. Quote or paraphrase the context directly.

[CONTEXT]
{context}
[END CONTEXT]
"""


def _build_messages(
    context_chunks: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    system_prompt = _SYSTEM_TEMPLATE.format(
        context=context_text,
        sentinel=NO_ANSWER_SENTINEL,
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Inject last N turns of conversation (cap at 10 to control token spend)
    for turn in conversation_history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_message})
    return messages


async def stream_chat(
    context_chunks: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    user_message: str,
) -> AsyncGenerator[Tuple[str, Optional[Dict[str, int]]], None]:
    """
    Stream tokens from OpenAI.

    Yields:
      (token_str, None)          — during streaming
      ("", usage_dict)           — as the final item, with token counts
    """
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = _build_messages(context_chunks, conversation_history, user_message)

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        temperature=0.2,  # Low temp = more faithful to context
        max_tokens=1024,
    )

    usage_data: Optional[Dict[str, int]] = None

    async for chunk in stream:
        # Token usage arrives in the final chunk
        if chunk.usage:
            usage_data = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
            }

        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content, None

    yield "", usage_data


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * COST_PER_INPUT_TOKEN
        + completion_tokens * COST_PER_OUTPUT_TOKEN
    )
