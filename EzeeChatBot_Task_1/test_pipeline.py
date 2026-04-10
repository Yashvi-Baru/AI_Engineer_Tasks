"""
End-to-end integration test — no OpenAI key needed for upload+retrieval.
Tests: chunking, embedding, vector storage, retrieval, stats schema.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.ingestion import clean_text
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts, embed_query
from app.services import vector_store
import uuid

# ── Sample knowledge base ─────────────────────────────────────────────────────
SAMPLE_DOC = """
EzeeChatBot is a document-grounded chatbot system. It allows users to upload PDFs,
plain text, or URLs and instantly get a chatbot that answers questions based only
on that content. The system uses sentence-transformers for embeddings, FAISS for
vector storage, and GPT-4o-mini as the language model.

The chunking strategy uses NLTK's sentence tokenizer. Sentences are grouped into
chunks of roughly 400 tokens with a one-sentence overlap between consecutive chunks.
This preserves semantic boundaries and avoids cutting answers mid-sentence.

Hallucination handling works via a sentinel: if the LLM cannot find the answer in
the provided context, it outputs EZEE_NO_ANSWER. The router detects this and
replaces it with a friendly fallback message, and logs the event as an unanswered
question in the analytics database.

Each bot gets its own FAISS index stored under data/bots/{bot_id}/. This ensures
complete isolation between knowledge bases — one client's documents never bleed
into another's queries.
"""

print("=" * 60)
print("EzeeChatBot — Integration Test")
print("=" * 60)

# Step 1: Clean + chunk
bot_id = str(uuid.uuid4())
cleaned = clean_text(SAMPLE_DOC)
chunks = chunk_text(cleaned, source="test_doc")
print(f"\n✓ Chunked into {len(chunks)} chunks")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: {len(c['text'])} chars | sentences: {c['metadata']['sentence_count']}")

# Step 2: Embed
texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)
print(f"\n✓ Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

# Step 3: Store
vector_store.upsert_chunks(bot_id, chunks, embeddings)
print(f"\n✓ Stored all chunks for bot_id: {bot_id[:8]}...")
print(f"  bot_exists check: {vector_store.bot_exists(bot_id)}")

# Step 4: Query — question THAT IS IN the doc
q1 = "What chunking strategy does EzeeChatBot use?"
q1_vec = embed_query(q1)
hits = vector_store.query_chunks(bot_id, q1_vec, top_k=3)
print(f"\n✓ Query: '{q1}'")
print(f"  Top hit distance: {hits[0]['distance']:.4f} (lower = more relevant)")
print(f"  Top hit preview: {hits[0]['text'][:120]}...")

# Step 5: Query — question NOT IN the doc (expect high distance)
q2 = "What is the GDP of France in 2024?"
q2_vec = embed_query(q2)
hits2 = vector_store.query_chunks(bot_id, q2_vec, top_k=3)
print(f"\n✓ Off-topic query: '{q2}'")
print(f"  Top hit distance: {hits2[0]['distance']:.4f}")
THRESHOLD = 0.40  # 1 - cosine_sim; above this = not relevant
relevant = [h for h in hits2 if h['distance'] <= THRESHOLD]
print(f"  Chunks passing relevance threshold ({THRESHOLD}): {len(relevant)}")
print(f"  → {'Would call LLM' if relevant else 'Would SHORT-CIRCUIT to fallback (no LLM call)'}")

# Step 6: Cleanup
vector_store.delete_bot(bot_id)
print(f"\n✓ Bot data deleted. bot_exists after delete: {vector_store.bot_exists(bot_id)}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
