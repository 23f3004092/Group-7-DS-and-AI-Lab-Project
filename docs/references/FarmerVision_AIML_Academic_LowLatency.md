# FarmerVision — AI/ML Architecture (Academic Setting: Network Latency ≈ Negligible)

Assumption: farmers have good connectivity (broadband/campus wifi/5G with
strong signal). Network RTT is no longer the constraint — everything below
is now a **compute/serving-architecture problem**. This is the right
framing for a thesis/academic system design where you control the deployment
environment.

## 0. What Actually Eats the Budget Now

With network RTT reduced to ~5-15ms, a 200-300ms budget is entirely GPU
compute time across: vision inference + embedding + vector search + LLM
generation. The LLM is still the dominant cost by an order of magnitude.
So the redesign target is: **can we get a 12B-class model's response
generation down into a ~150-200ms slice of that budget, and if not, what
model size/technique combination does get us there?**

---

## 1. Latency Budget (Academic, Ideal Network)

```
Total budget: 250ms (midpoint of 200-300ms)

┌─────────────────────────────────────────────────────────┐
│ Network (client → server, good connectivity)    ~10-15ms │
├─────────────────────────────────────────────────────────┤
│ Vision Classification (ViT-Small, TensorRT,      ~15-25ms│
│  FP16, on A100/H100, batched)                            │
├─────────────────────────────────────────────────────────┤
│ Text Embedding (MuRIL, FP16, TensorRT)            ~5-10ms│
├─────────────────────────────────────────────────────────┤
│ Vector Search (HNSW, in-memory, top-5)            ~3-8ms │
├─────────────────────────────────────────────────────────┤
│ Parallel Tool Calls (mandi/weather/yield,         ~10-20ms│
│  cached or fast in-region APIs, fired concurrently)       │
├─────────────────────────────────────────────────────────┤
│ Guardrails Pre-Filter (rule-based)                ~2-5ms │
├─────────────────────────────────────────────────────────┤
│ LLM Generation ── THE constraint             ~150-180ms  │
├─────────────────────────────────────────────────────────┤
│ Guardrails Post-Check (regex/classifier)          ~2-5ms │
├─────────────────────────────────────────────────────────┤
│ Network (response back)                          ~10-15ms│
└─────────────────────────────────────────────────────────┘
                                          Total: ~210-280ms
```

Everything except LLM generation is already cheap and parallelizable. The
whole redesign question reduces to: **how do you get LLM generation down
to ~150-180ms?**

---

## 2. Making LLM Generation Fit ~150-180ms

A 12B dense model generating 150-250 tokens autoregressively will not fit
this window even on an H100, even with vLLM batching — decode is
inherently sequential per token. To hit the target, combine several
real, published techniques rather than relying on one:

### a) Shrink the model
- Distill Gemma 12B down to a **2-4B parameter** domain-specific model,
  fine-tuned only on agri-advisory data (crop disease, dosage, mandi
  Q&A). A well-distilled small model on a narrow domain retains most of
  the accuracy that matters here — you don't need general-purpose
  reasoning breadth for "how do I treat wheat rust."
- At 2-4B params, FP16/FP8 on an H100 with vLLM's continuous batching
  realistically hits **80-150 tokens/sec** per stream even under
  moderate concurrent load — enough to generate a ~60-90 token response
  (short, farmer-readable advisory, not an essay) inside ~150-180ms.

### b) Cap and shape the output
- Constrain generation to **60-100 tokens** — a concise 3-4 sentence
  advisory, not a long-form essay. This is also better UX for a mobile
  farmer-facing app regardless of latency.
- Use **structured/constrained decoding** (grammar-constrained or
  JSON-schema-constrained generation via vLLM's guided decoding) so the
  model doesn't wander into longer free-form text — this both shortens
  output and removes the need for a separate guardrails post-parse step.

### c) Speculative decoding
- Pair the 2-4B model with a much smaller draft model (e.g., a
  350M-500M draft) using **speculative decoding** — the draft model
  proposes multiple tokens, the main model verifies them in a single
  forward pass. This can realistically give a **1.5-2.5x speedup** on
  decode-bound generation, which is exactly the bottleneck here.

### d) Serving-level optimizations
- **FP8 quantization** (H100 native support) — roughly 1.5-2x throughput
  vs FP16 with minimal quality loss at this model size.
- **Continuous batching + PagedAttention** (vLLM) — keeps GPU utilization
  high under concurrent requests without the latency cliff of static
  batching.
- **Prefix/KV caching** — the system prompt, guardrail instructions, and
  retrieved-context template are largely shared across requests; caching
  this prefix avoids re-computing attention over it every time, which
  matters a lot when the "enriched prompt" is long relative to the
  60-100 token output.

### e) Skip generation when possible (still valid even with good network)
- Even in an ideal-network academic setup, an **exact-match/template
  cache** for the most common (disease, intent) pairs is still valid and
  still the cheapest way to guarantee <100ms for a meaningful fraction of
  traffic. It's not a network workaround — it's a compute workaround,
  and it's legitimate to include in a thesis as a "warm cache" tier.

---

## 3. Revised Architecture Diagram

```
    [Mobile/Web Client] ─── good connectivity, ~10-15ms RTT
            │
            ▼
    [API Gateway / FastAPI Router]
            │
            ├──► [Vision: ViT-Small, TensorRT FP16]──────┐
            │                                    ~15-25ms │
            ├──► [MuRIL Embed, TensorRT FP16]─────────────┤ fired
            │                                     ~5-10ms │ in
            ├──► [Vector DB Search: HNSW in-memory,────────┤ parallel
            │     Qdrant/ChromaDB, top-5 chunks]            │
            │                                      ~3-8ms   │
            ├──► [Tool call: Mandi/Weather API]──────────────┤
            │     (cached, in-region)              ~5-10ms  │
            │                                                │
            └──► [Tool call: Yield Prediction ML]─────────────┘
                  (small regression/gradient-boosted
                   model, not the main LLM — separate
                   lightweight service)              ~10-15ms
                        │
                        ▼
            [Guardrails Pre-Filter: rule-based]  ~2-5ms
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  Exact-match template cache lookup      │
        │  (disease+intent) — HIT → skip below     │
        └───────────────┬─────────────────────────┘
                     MISS │
                        ▼
        ┌───────────────────────────────────────────────┐
        │  Distilled Agri-LLM (2-4B), FP8, vLLM            │
        │  + speculative decoding (350-500M draft model)   │
        │  + guided/constrained decoding (60-100 tok cap)  │
        │  + cached prefix (system+guardrail instructions) │
        │                                    ~150-180ms    │
        └───────────────┬─────────────────────────────────┘
                        ▼
            [Guardrails Post-Check: regex/classifier] ~2-5ms
                        │
                        ▼
                [Response → Client]  ~10-15ms
                        │
              Total: ~210-280ms (cache miss)
              Total: ~60-100ms  (cache hit, template path)
```

---

## 4. Model Choice Summary Table

| Component | Model | Precision | Serving | Latency |
|---|---|---|---|---|
| Vision | ViT-Small (or distilled MobileViT if pushing further) | FP16 | TensorRT | 15-25ms |
| Embedding | MuRIL-base | FP16 | TensorRT | 5-10ms |
| Retrieval | HNSW (Qdrant), in-memory | — | — | 3-8ms |
| Generation | **Gemma distilled to 2-4B**, agri-domain fine-tuned | FP8 | vLLM, continuous batching, PagedAttention, guided decoding | 150-180ms |
| Speculative draft | 350-500M draft model, same tokenizer family | FP8 | vLLM speculative decoding | (folded into above) |

---

## 5. Why Not Just Keep Gemma 12B?

Worth stating plainly in a thesis defense: 12B is defensible when latency
isn't the constraint (batch/offline advisory generation, complex
multi-turn reasoning). But under a hard 200-300ms real-time constraint,
distillation to 2-4B with domain-specific fine-tuning is the standard,
published approach (this mirrors how production systems like search
autocomplete or coding-assist "fast" tiers work — small model for
latency-critical path, large model reserved for cases that can tolerate
seconds). This is a legitimate and well-supported design tradeoff to
present, not a shortcut — cite it as such.

---

## 6. Compound Query Handling — Upfront Decomposition (before ReAct loop)

**Problem:** naive ReAct reasons turn-by-turn (think → act → observe, repeat).
For a compound query like *"what are the incentives for growing X, what
yield would I get, and would I be successful,"* this means 3-4 sequential
LLM reasoning turns (~150-180ms each) before synthesis — 600ms+ of stacked
agent overhead, blowing the latency budget on exactly the queries that
matter most for a real farmer decision.

**Fix:** a lightweight classifier runs *before* the agent loop, not as part
of it. It doesn't reason — it just extracts intent flags and entities in a
single fast forward pass (small model, not the 12B one), so every needed
tool fires in **one parallel batch** instead of being discovered turn-by-turn.

```
Query → [Compound Intent/Entity Extractor]
         DistilBERT-class model, ~10-15ms, multi-label classification
         + NER for crop/region/quantity entities
              │
              ▼
   { needs_policy: true, needs_yield: true, needs_price: true,
     needs_profitability: true, crop: "X", region: <from farmer profile>,
     is_compound: true }
              │
              ▼
   Fan-out ALL flagged tools in parallel (Vector DB + Yield + Mandi + ─┐
   Profitability Estimator) — same pattern as the simple-query path,   │
   just with more branches active at once                             │
              │                                                        │
              ▼◄───────────────────────────────────────────────────────┘
   ONE synthesis pass (~150-180ms), not four
```

This keeps compound queries inside the same ~250-300ms envelope as simple
ones for the tool-execution side — the only cost that scales with query
complexity is synthesis prompt length (longer context = marginally more
prefill time, not more sequential turns).

---

## 7. Yield Answer Cache — Vector-Indexed Fast Path (separate from the GBT tool)

This is distinct from "replacing yield prediction with RAG," which we ruled
out — the GBT/regression model stays the source of truth for computing a
yield number. What this adds is a **cache layer in front of it**, so
recurring (crop × region × season × soil-type) combinations don't re-run
the model or the multi-tool fan-out at all.

### Why vector, not exact-key hash
Exact-match caching (like the fast-path template cache in §1) only hits
when the *same* (crop, region) pair recurs verbatim. But real farmer
queries vary in phrasing and slightly in inputs (soil type approximated
differently, nearby villages, slightly different acreage) while describing
essentially the same underlying agronomic conditions. A **vector-indexed
cache over the structured input features** (not the raw text) catches
these near-duplicates that an exact hash would miss.

```
[Compound Intent Extractor output: crop, region, soil_type, season, acreage]
              │
              ▼
[Feature Encoder — small trained embedder over structured
 agronomic features (NOT MuRIL, NOT shared with policy/KCC index)]
              │
              ▼
[Yield Cache Vector Namespace — separate Qdrant collection]
   query: nearest neighbor within similarity threshold (e.g., cosine > 0.92)
              │
      ┌───────┴────────┐
      │                 │
   HIT (≥ threshold)   MISS / below threshold
      │                 │
   Return cached        Call GBT Yield Prediction tool (full compute)
   yield answer +       │
   "based on similar    Write result back into Yield Cache Vector
   conditions in your   Namespace for future hits (with TTL — see below)
   area" framing              │
      │                       │
      └───────────┬───────────┘
                   ▼
        Feed into synthesis (single pass)
```

### Guardrail on cache reuse
Because yield estimates feed into financial-adjacent advice ("would I be
successful"), a near-duplicate cache hit should be **labeled as
approximate** in the response ("similar conditions" framing, not stated as
this farmer's exact number) — and the similarity threshold should be
conservative (high cosine cutoff) rather than loose, since a wrong cached
yield is worse than a 150ms latency cost on a cache miss.

### TTL and staleness
Unlike the policy/KCC index (fairly static), yield conditions are
season-dependent — rainfall, pest pressure, and soil moisture shift
within a season. Cache entries should expire on a **seasonal TTL** (e.g.,
30-45 days) rather than being treated as permanent, and should be
invalidated early if a major weather event hits the region (tie this to
your weather tool as an invalidation trigger, not just a timer).

### Net effect on latency
| Case | Path | Latency |
|---|---|---|
| Common (crop, region) combo, cache hit | Yield Cache Vector Namespace | ~5-10ms (replaces the GBT call entirely) |
| Novel combo, cache miss | Full GBT Yield Prediction tool | ~10-15ms (as in §1, runs in parallel with other tools anyway) |
| Compound query overall | Intent extraction + parallel fan-out + one synthesis | ~210-290ms regardless of which yield path is taken |

---

## 8. Evaluation Note for Academic Writeup

If this is going into a report, worth benchmarking and reporting:
- p50/p95/p99 latency broken down per pipeline stage (as in the table above)
- Accuracy delta between the 12B teacher and 2-4B distilled student on a
  held-out agri-advisory eval set (this is your key tradeoff to quantify)
- Throughput (requests/sec) at target latency under concurrent load —
  this is where continuous batching numbers matter and make a good
  graph/table
