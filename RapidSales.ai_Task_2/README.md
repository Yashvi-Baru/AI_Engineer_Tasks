# RapidSales.ai — AI Pipeline Redesign

This submission contains two files for the Senior Systems Architecture assessment.

---

## Files

### `RapidSales_AI_Pipeline_Architecture_Redesign.docx`
The full written report covering the revised pipeline architecture. Sections include model selection per channel with justification, async vs synchronous processing decisions, fallback handling, caching strategy, cost analysis, and an observability plan.

### `voice_script_cache.py`
A production-ready implementation of the caching layer described in the report. It uses Redis as the backing store with a 72-hour TTL, keyed on `(industry, product_category)`. The module includes:

- `VoiceScriptCache` — get, set, targeted invalidation, and full flush
- `VoiceScriptService` — orchestrates the full fallback chain: cache → Llama 3.1 on Groq → GPT-4o-mini → hardcoded emergency template
- Distributed locking to prevent thundering-herd on cache misses
- Structured logging on every path for Prometheus instrumentation

---

## Quick start

```bash
pip install redis openai groq

# Set environment variables
export REDIS_URL="redis://localhost:6379"
export OPENAI_API_KEY="..."
export GROQ_API_KEY="..."
```

```python
from voice_script_cache import VoiceScriptCache, VoiceScriptService
import redis, openai
from groq import Groq

r       = redis.from_url(os.environ["REDIS_URL"])
cache   = VoiceScriptCache(r)
service = VoiceScriptService(cache, Groq(), openai.OpenAI())

script = service.get_script("SaaS", "CRM", lead_context={})
print(script.script)
```

---

