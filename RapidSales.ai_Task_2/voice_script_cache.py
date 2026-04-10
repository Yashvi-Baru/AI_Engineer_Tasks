# Production caching layer for RapidSales.ai voice scripts
# Key: industry + product_category | TTL: 72h | Fallback: Llama → GPT-4o-mini

import redis
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ScriptSource(Enum):
    CACHE     = "cache"
    LLAMA     = "llama_groq"
    OPENAI    = "openai_fallback"
    EMERGENCY = "emergency_template"

class CacheConfig:
    TTL_SECONDS     = 72 * 3600   # 72 hours — scripts stable, refresh weekly
    VERSION         = "v1"          # bump to invalidate all cached scripts
    MAX_KEY_LENGTH  = 200
    LOCK_TTL        = 30            # prevent thundering herd on cache miss

@dataclass
class VoiceScript:
    script:        str
    industry:      str
    product_cat:   str
    source:        str
    generated_at:  str
    expires_at:    str
    token_count:   int
    version:       str = CacheConfig.VERSION

def build_cache_key(industry: str, product_category: str) -> str:
    """Normalize inputs and build a stable, version-prefixed cache key."""
    norm_industry = industry.lower().strip().replace(" ", "_")[:50]
    norm_product  = product_category.lower().strip().replace(" ", "_")[:50]
    key = f"voice:script:{CacheConfig.VERSION}:{norm_industry}:{norm_product}"
    # If key exceeds Redis limit, hash the variable part
    if len(key) > CacheConfig.MAX_KEY_LENGTH:
        h = hashlib.sha256(f"{norm_industry}:{norm_product}".encode()).hexdigest()[:16]
        key = f"voice:script:{CacheConfig.VERSION}:hashed:{h}"
    return key

class VoiceScriptCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def get(self, industry: str, product_category: str) -> Optional[VoiceScript]:
        key = build_cache_key(industry, product_category)
        try:
            data = self.redis.get(key)
            if data:
                logger.info(f"cache_hit key={key}")
                return VoiceScript(**json.loads(data))
        except redis.RedisError as e:
            logger.error(f"redis_read_error key={key} err={e}")
        return None   # explicit None signals "proceed to generation"

    def set(self, script: VoiceScript) -> bool:
        key = build_cache_key(script.industry, script.product_cat)
        try:
            self.redis.setex(key, CacheConfig.TTL_SECONDS, json.dumps(asdict(script)))
            logger.info(f"cache_set key={key} ttl={CacheConfig.TTL_SECONDS}s")
            return True
        except redis.RedisError as e:
            logger.error(f"redis_write_error key={key} err={e}")
            return False  # don't crash — script was generated, just won't be cached

    def invalidate(self, industry: str, product_category: str) -> bool:
        """Call this when a client reports poor script quality for an industry/product."""
        key = build_cache_key(industry, product_category)
        deleted = self.redis.delete(key)
        logger.info(f"cache_invalidated key={key} existed={deleted > 0}")
        return deleted > 0

    def invalidate_all(self) -> int:
        """Nuclear option: flush all scripts. Use on VERSION bump or major prompt update."""
        pattern = f"voice:script:{CacheConfig.VERSION}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
        logger.warning(f"cache_flush_all count={len(keys)}")
        return len(keys)


class VoiceScriptService:
    """Orchestrates: cache → Llama → GPT-4o-mini → emergency template."""

    def __init__(self, cache: VoiceScriptCache, llama_client, openai_client):
        self.cache   = cache
        self.llama   = llama_client
        self.openai  = openai_client

    def get_script(self, industry: str, product_category: str, lead_context: dict) -> VoiceScript:
        start = time.monotonic()

        # 1. Cache lookup
        cached = self.cache.get(industry, product_category)
        if cached:
            _log_latency("cache_hit", start, industry, product_category)
            return cached

        # 2. Acquire distributed lock (thundering herd protection)
        lock_key = f"lock:{build_cache_key(industry, product_category)}"
        with self.cache.redis.lock(lock_key, timeout=CacheConfig.LOCK_TTL, blocking_timeout=35):
            # Re-check after acquiring lock — another worker may have populated it
            cached = self.cache.get(industry, product_category)
            if cached:
                return cached

            # 3. Generate via Llama (primary)
            try:
                script_text, tokens = self._generate_llama(industry, product_category)
                script = self._make_script(script_text, industry, product_category,
                                           ScriptSource.LLAMA, tokens)
                self.cache.set(script)  # best-effort; failure is non-fatal
                _log_latency("llama_generated", start, industry, product_category)
                return script
            except Exception as e:
                logger.warning(f"llama_failed industry={industry} err={e}")

            # 4. Fallback to GPT-4o-mini
            try:
                script_text, tokens = self._generate_openai(industry, product_category)
                script = self._make_script(script_text, industry, product_category,
                                           ScriptSource.OPENAI, tokens)
                self.cache.set(script)
                _log_latency("openai_fallback", start, industry, product_category)
                return script
            except Exception as e:
                logger.error(f"openai_fallback_failed industry={industry} err={e}")

            # 5. Emergency template (never drops a call)
            logger.critical(f"using_emergency_template industry={industry}")
            return self._emergency_template(industry, product_category)

    def _generate_llama(self, industry, product_cat) -> tuple[str, int]:
        prompt = _build_prompt(industry, product_cat)
        resp = self.llama.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.4
        )
        return resp.choices[0].message.content, resp.usage.total_tokens

    def _generate_openai(self, industry, product_cat) -> tuple[str, int]:
        prompt = _build_prompt(industry, product_cat)
        resp = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.4
        )
        return resp.choices[0].message.content, resp.usage.total_tokens

    def _make_script(self, text, industry, product_cat, source, tokens) -> VoiceScript:
        now = datetime.utcnow()
        return VoiceScript(
            script=text, industry=industry, product_cat=product_cat,
            source=source.value,
            generated_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=CacheConfig.TTL_SECONDS)).isoformat(),
            token_count=tokens
        )

    def _emergency_template(self, industry, product_cat) -> VoiceScript:
        now = datetime.utcnow()
        text = (f"Hi, I'm calling about how we help {industry} businesses with {product_cat}. "
                "I'd love 2 minutes to share something that could save you time. "
                "Is now a good moment?")
        return VoiceScript(script=text, industry=industry, product_cat=product_cat,
            source=ScriptSource.EMERGENCY.value, generated_at=now.isoformat(),
            expires_at=now.isoformat(), token_count=0)


def _build_prompt(industry: str, product_category: str) -> str:
    return f"""You are a sales call script writer. Write a warm, natural 30-second cold call opening script.
Industry: {industry}
Product category: {product_category}
Requirements:
- Under 80 words
- Opens with a specific pain point for this industry
- One clear value proposition
- Ends with a soft yes/no question
Return only the script text, no labels or formatting."""

def _log_latency(source: str, start: float, industry: str, product_cat: str):
    ms = round((time.monotonic() - start) * 1000)
    logger.info(f"voice_script source={source} latency_ms={ms} industry={industry} product={product_cat}")
    # Emit to Prometheus: voice_script_latency_ms{source=source, industry=industry}