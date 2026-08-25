# Proposed Pipeline Changes — FarmerVision RAG

> Based on analysis of [`run_e2e_eval.py`](file:///d:/Group-7-DS-and-AI-Lab-Project/scripts/run_e2e_eval.py) across this session.

---

## 1. Retrieval — Batched Embedding (High Impact, Zero Cost)

**Problem**: [`retrieve()`](file:///d:/Group-7-DS-and-AI-Lab-Project/scripts/run_e2e_eval.py#L343-L350) calls `embedder.encode()` once per intent in a loop → **N × 50ms** latency.

**Fix**: Single batched `encode()` call for all sub-queries.

```python
# Before (current) — sequential, N × 50ms
for intent in intents:
    q_vec = embedder.encode(sub_query, normalize_embeddings=True).tolist()

# After — batched, ~50ms + (N-1)×5ms regardless of N
sub_queries = [f"{i.replace('_',' ')}: {query}" for i in intents]
vecs = embedder.encode(sub_queries, batch_size=len(sub_queries), normalize_embeddings=True)
```

**Latency saving**: N=3 → saves ~90ms; N=5 → saves ~180ms.

---

## 2. Retrieval — Parallel Qdrant Queries (Medium Impact, Safe)

**Problem**: After encoding, Qdrant searches are still sequential in the current loop.

**Fix**: `ThreadPoolExecutor` for parallel HTTP queries once vectors are ready.

```python
import concurrent.futures

def _search(vec):
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec.tolist(),
        limit=limit_per_intent,
        with_payload=True
    ).points

with concurrent.futures.ThreadPoolExecutor(max_workers=len(vecs)) as pool:
    results = list(pool.map(_search, vecs))
```

**Latency saving**: Qdrant queries collapse from `N × 12ms` to `~12ms + log₂(N)×2ms`.

---

## 3. Retrieval — Intent → Source Routing / Payload Filtering (High Quality Impact)

**Problem**: All 723,439 points are searched blindly. PDF advisory chunks (`ppqs_advisories`: 1,044 chunks, `up_acp`: 4,818 chunks) make up only **~1% of the collection** and are drowned out by 716k KCC chunks — even for queries that should return them (e.g. `disease_pest`).

**Fix**: Map IEG intents to preferred `source` values and apply Qdrant payload filters.

```python
from qdrant_client.models import Filter, FieldCondition, MatchAny

INTENT_SOURCE_MAP = {
    "disease_pest":         ["ppqs_advisories", "kcc"],
    "cultivation_practice": ["up_acp", "kcc"],
    "nutrition_fertilizer": ["ppqs_advisories", "kcc"],
    "yield_estimation":     ["kcc"],
    "general":              None,   # no filter → search all
}

sources = set()
for intent in intents:
    srcs = INTENT_SOURCE_MAP.get(intent)
    if srcs:
        sources.update(srcs)

query_filter = Filter(must=[
    FieldCondition(key="source", match=MatchAny(any=list(sources)))
]) if sources else None

hits = client.query_points(
    collection_name=COLLECTION_NAME,
    query=q_vec,
    query_filter=query_filter,
    limit=TOP_K,
    with_payload=True
).points
```

**Quality impact**: `disease_pest` queries will actually surface ETL tables and pesticide schedules from `ppqs_advisories` instead of competing against 716k KCC Q&A chunks.

---

## 4. Retrieval — Per-Chunk Score Floor (Quick Win)

**Problem**: Only the **top-1 hit score** is used for the tier gate (`TIER_GROUNDED=0.66`). All remaining hits (up to rank 10) enter the generator context unchecked — a rank-8 hit at score 0.41 still gets included.

**Fix**: Filter out low-scoring tail chunks before passing to the generator.

```python
MIN_CHUNK_SCORE = 0.50  # tune based on your score distribution

hits = [h for h in hits if h.score >= MIN_CHUNK_SCORE]
if not hits:
    return [], "abstain", 0.0
```

**Quality impact**: Reduces hallucination risk from weak, tangentially-related chunks contaminating the prompt context.

---

## 5. Retrieval — Minimum Chunk Length Filter (Quick Win)

**Problem**: No check on chunk content length. Empty or near-empty chunks (from PDF extraction noise, headers, table-of-contents fragments) pass through to the generator.

**Fix**:

```python
MIN_CHUNK_CHARS = 50

hits = [h for h in hits if len(h.payload.get("text", "")) >= MIN_CHUNK_CHARS]
```

**Quality impact**: Eliminates OCR noise chunks like `"Sir, > Copy for information:"` or single-line table headers that add tokens but no information.

---

## 6. Retrieval — Source Diversity Cap (Medium Quality)

**Problem**: All 10 top hits could come from the same source document (common in large PDF documents chunked into 60+ pieces like `up_acp` district PDFs).

**Fix**: Cap the number of chunks from any single source.

```python
from collections import Counter

MAX_PER_SOURCE = 3
source_counts = Counter()
filtered_hits = []
for h in sorted(hits, key=lambda x: x.score, reverse=True):
    src = h.payload.get("source", "unknown")
    if source_counts[src] < MAX_PER_SOURCE:
        filtered_hits.append(h)
        source_counts[src] += 1
hits = filtered_hits
```

**Quality impact**: Ensures the generator sees a diversity of sources (KCC + PDF advisory + ACP) rather than 10 fragments from one document.

---

## 7. Retrieval — Cross-Encoder Reranking (High Quality, ~50ms cost)

**Problem**: Pipeline is bi-encoder only (bge-m3 cosine similarity). Chunks that are semantically adjacent to the query but don't actually answer it can score higher than directly relevant chunks.

**Fix**: Add a lightweight cross-encoder reranker after bge-m3 retrieval.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # ~22MB, CPU-friendly

pairs = [(query, h.payload.get("text", "")) for h in hits]
scores = reranker.predict(pairs)   # ~30-50ms for 10 pairs on CPU
hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
# Optionally drop chunks with very low cross-encoder score
hits = [h for h, s in zip(hits, sorted(scores, reverse=True)) if s > 0.0]
```

**Quality impact**: Single highest-quality improvement for chunk ordering. Especially valuable for `disease_pest` queries where the bge-m3 score alone can't distinguish a generic "pest" mention from a specific ETL recommendation.

---

## 8. eval_AB — Parallel ViT + IEG Inference

**Problem**: In [`eval_AB()`](file:///d:/Group-7-DS-and-AI-Lab-Project/scripts/run_e2e_eval.py#L622), ViT inference and IEG inference are run sequentially even though they are independent.

**Fix**: Run both in parallel.

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    vit_future = pool.submit(vit_model.predict, image_col)
    ieg_future = pool.submit(ieg_run, ieg_model, ieg_tok, id2intent, ner_labels, text)
    result = vit_future.result()
    intent, blocked, m_flag, r_flag = ieg_future.result()
```

**Latency saving**: ViT (~200ms) and IEG (~80ms) overlap → saves ~80ms per AB scenario.

---

## 9. `TOP_K` Scaling with Intent Count

**Problem**: `TOP_K = 10` is fixed. With N=3 intents, `limit_per_intent = 10 // 3 = 3`, meaning only 3 hits per intent — too sparse for reliable context.

**Fix**: Scale `TOP_K` with the number of active intents.

```python
TOP_K_BASE = 10
TOP_K = max(TOP_K_BASE, len(intents) * 5)  # 5 hits minimum per intent
```

> [!WARNING]
> Increasing `TOP_K` grows the prompt context and adds 50–150ms to generation time per +500 tokens. Only use when intents > 2.

---

## Summary Table

| # | Change | Where | Latency | Quality | Effort |
|---|---|---|---|---|---|
| 1 | Batched embedding | `retrieve()` | ✅ −90ms (N=3) | = | Low |
| 2 | Parallel Qdrant queries | `retrieve()` | ✅ −20ms | = | Low |
| 3 | Intent → source routing | `retrieve()` | = | ✅✅ Advisory chunks surface | Low |
| 4 | Per-chunk score floor | `retrieve()` | = | ✅ Less hallucination | Low |
| 5 | Min chunk length filter | `retrieve()` | = | ✅ No OCR noise | Low |
| 6 | Source diversity cap | `retrieve()` | = | ✅ Diverse context | Low |
| 7 | Cross-encoder reranking | `retrieve()` | ⚠️ +50ms | ✅✅ Best chunk order | Medium |
| 8 | Parallel ViT+IEG in AB | `eval_AB()` | ✅ −80ms | = | Low |
| 9 | Dynamic TOP_K | `retrieve()` | ⚠️ +gen time | ✅ Better multi-intent | Low |
