"""
SQLite persistence for per-bot analytics.

Schema:
  bot_stats: one row per bot recording aggregate counters
  chat_events: one row per chat call recording raw data for latency averaging

We use aiosqlite for non-blocking async access so we never stall the event loop.
"""

import os
import aiosqlite

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "analytics.db")
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)


async def init_db() -> None:
    """Create tables if they don't exist. Called at app startup."""
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bot_stats (
                bot_id          TEXT PRIMARY KEY,
                total_messages  INTEGER DEFAULT 0,
                total_cost_usd  REAL    DEFAULT 0.0,
                unanswered      INTEGER DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id          TEXT    NOT NULL,
                latency_ms      REAL    NOT NULL,
                cost_usd        REAL    NOT NULL DEFAULT 0.0,
                was_unanswered  INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT    DEFAULT (datetime('now'))
            )
        """)
        await db.commit()


async def record_chat_event(
    bot_id: str,
    latency_ms: float,
    cost_usd: float,
    was_unanswered: bool,
) -> None:
    """Append one chat event and update the aggregate stats row atomically."""
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO chat_events (bot_id, latency_ms, cost_usd, was_unanswered)
            VALUES (?, ?, ?, ?)
            """,
            (bot_id, latency_ms, cost_usd, int(was_unanswered)),
        )

        # Upsert the aggregate row
        await db.execute(
            """
            INSERT INTO bot_stats (bot_id, total_messages, total_cost_usd, unanswered)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(bot_id) DO UPDATE SET
                total_messages = total_messages + 1,
                total_cost_usd = total_cost_usd + excluded.total_cost_usd,
                unanswered     = unanswered     + excluded.unanswered
            """,
            (bot_id, cost_usd, int(was_unanswered)),
        )
        await db.commit()


async def get_stats(bot_id: str) -> dict:
    """
    Return stats for a bot. avg_latency_ms is computed from raw events
    rather than stored as a running average to avoid floating-point drift.
    """
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        row = await db.execute_fetchall(
            "SELECT * FROM bot_stats WHERE bot_id = ?", (bot_id,)
        )
        agg = dict(row[0]) if row else {
            "total_messages": 0,
            "total_cost_usd": 0.0,
            "unanswered": 0,
        }

        latency_row = await db.execute_fetchall(
            "SELECT AVG(latency_ms) AS avg FROM chat_events WHERE bot_id = ?",
            (bot_id,),
        )
        avg_latency = latency_row[0]["avg"] or 0.0

    return {
        "bot_id": bot_id,
        "total_messages": agg["total_messages"],
        "avg_latency_ms": round(avg_latency, 2),
        "estimated_cost_usd": round(agg["total_cost_usd"], 6),
        "unanswered_questions": agg["unanswered"],
    }
