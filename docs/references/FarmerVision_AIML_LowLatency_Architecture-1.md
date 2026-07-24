# FarmerVision — AI/ML Architecture for Low-Latency Serving (Target: 200–300ms)

## 0. Reality Check First

A single-pass pipeline of `Vision Classify → Embed → Vector Search → 12B LLM Synthesis`
cannot hit 200–300ms end-to-end. Token generation on a 12B model, even fully
batched on an A100 with vLLM, has a floor of roughly 30–60 tokens/sec per
effective stream under load — a 150-token Hindi response alone costs
800ms–2.5s. No infra trick removes this; it's compute-bound, not network-bound.

**The fix isn't to make the LLM faster — it's to remove the LLM from the
critical path for the queries that don't need fresh generation, and let it
run asynchronously for the ones that do.**

This is a **tiered fast-path / slow-path architecture**:

- **Fast path (target 200–300ms):** covers the ~70–80% of real traffic that
  is a common disease label + a common question pattern (Zipfian distribution
  — a small number of crop/disease/query combos dominate real usage).
- **Slow path (1–4s, streamed):** covers novel queries, rare diseases,
  multi-turn follow-ups — full ReAct agent + 12B synthesis, delivered async
  while the fast-path gives an immediate provisional answer.

---

## 1. Fast Path — Latency Budget (target 200–300ms)

```
Farmer (mobile, 3G/4G rural)
   │
   │  0–120ms: network RTT (compressed image, ~50-150KB after client resize)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ Edge/Regional Inference Node (co-located near telco POPs)     │
│                                                                 │
│  [Distilled Vision Model: MobileViT / EfficientNet-Lite]       │
│   quantized INT8, ONNX Runtime / TensorRT           ~15-30ms   │
│      │                                                          │
│      ▼                                                          │
│  disease_label + confidence                                    │
│      │                                                          │
│      ▼                                                          │
│  [Redis: exact-match lookup]                                   │
│   key = hash(disease_label + query_intent_class + lang)        │
│                                                    ~2-5ms       │
│      │                                                          │
│      ├── HIT (≈70-80% of traffic) ──────────────────┐          │
│      │   pre-generated, guardrail-approved            │        │
│      │   answer template, slot-filled with             │        │
│      │   live mandi price (cached, ~5ms)               │        │
│      │                                    ~10-20ms     │        │
│      │                                                  ▼        │
│      │                              Return to farmer (total: ~200-280ms) │
│      │                                                            │
│      └── MISS ─────────────────────────────────────────────►  handoff to slow path
│                                                                    │
└─────────────────────────────────────────────────────────────────┘
```

**Fast-path total: network (≤120ms) + vision (≤30ms) + cache lookup (≤5ms)
+ template fill (≤20ms) + return trip (≤100ms) ≈ 200–280ms.**

### What makes this possible
1. **No LLM generation in the fast path at all.** Answers are pre-written
   by domain experts/agronomists for the top N (disease × intent × language)
   combinations, reviewed once through guardrails offline, and stored as
   templates with slots for dynamic data (price, dosage per region, date).
2. **Distilled vision model, not the full ViT.** MobileViT-XXS or
   EfficientNet-Lite0, INT8-quantized, run on edge GPU/NPU — sub-30ms
   inference is realistic at this size.
3. **No embedding, no vector search, no retrieval** in the fast path —
   the disease label itself is the retrieval key.
4. **Edge placement matters as much as model size.** If the nearest
   inference node is 2000km away, network RTT alone eats the budget.
   Deploy regional inference nodes (e.g., one per state or telco region)
   so RTT is 30–80ms instead of 150–300ms.

### Cache/template coverage strategy
- Analyze historical query logs (or seed with agronomist input pre-launch)
  to identify the top ~200–500 (disease, intent) pairs — this typically
  covers the bulk of real-world traffic in a domain this concentrated
  (a handful of crop diseases dominate in any given region/season).
- Each template pre-passes guardrails once, offline — not per-request.
- Templates refresh on a schedule (weekly) as new data/policy comes in,
  not per-query.

---

## 2. Slow Path — Full ReAct Pipeline (cache miss / novel query)

```
Cache MISS on fast path
   │
   ▼
[Return immediate provisional response to farmer]
   "Likely {disease_label} ({confidence}%) — preparing detailed advice..."
   (this alone satisfies the "responsive app" feel within the 300ms window)
   │
   ▼ (async, non-blocking)
[Full ML Mesh — as designed in LLD]
   MuRIL Embed → Vector Search → Parallel Tool Calls → Guardrails Pre-Filter
   → Gemma 12B Synthesis (vLLM, streamed token-by-token) → Guardrails Post-Check
   │
   ▼
[Push via WebSocket/SSE to mobile client — streams in as generated]
   Farmer sees the provisional answer immediately, then the full
   synthesized answer fills in over the next 1-4 seconds, token by token.
   │
   ▼
[New (disease, intent) template candidate logged for offline review →
 promoted into fast-path cache if it recurs]
```

Streaming (SSE or WebSocket) is what makes the slow path *feel* fast even
though it isn't 300ms — the farmer sees words appearing immediately rather
than a spinner for 2 seconds.

---

## 3. Model Sizing Decisions

| Task | Model in v1 design | Fast-path model | Why |
|---|---|---|---|
| Vision classification | ViT (via Triton) | MobileViT-XXS / EfficientNet-Lite0, INT8 | 10-20x smaller, edge-deployable, <30ms |
| Query intent classification | (new, doesn't exist in v1) | DistilBERT-tiny or MuRIL-distilled, fine-tuned on ~10-20 intent classes | Needed to pick the right template; <10ms |
| Text embedding | MuRIL (full) | Kept only in slow path | Not needed at all in fast path — no retrieval |
| Synthesis | Gemma 12B | Not used in fast path | Templates replace generation entirely |
| Slow-path synthesis | Gemma 12B via vLLM, batched, streamed | same | Async, latency budget relaxed to 1-4s |

---

## 4. Infra Requirements to Hit the Target

| Component | Requirement |
|---|---|
| Vision model serving | TensorRT/ONNX Runtime on edge GPU (T4/L4-class) or NPU, regional deployment |
| Intent classifier | Same edge node, co-located with vision model to avoid extra hop |
| Redis | Regional Redis cluster, co-located with edge inference (not centralized) — cross-region Redis calls alone can burn 50-100ms |
| CDN/Edge PoPs | Route farmer requests to nearest regional node (not a single central data center) |
| Mobile client | Client-side image resize/compress BEFORE upload — this is often the single biggest lever on rural networks |
| Streaming | SSE or WebSocket connection held open for slow-path token streaming |

---

## 5. Realistic Latency Table

| Path | p50 | p95 | Notes |
|---|---|---|---|
| Fast path (cache hit) | 220ms | 320ms | Target met for ~70-80% of traffic |
| Fast path (cache miss → provisional response only) | 250ms | 350ms | "Thinking..." response, not final |
| Slow path (full synthesis, first token) | 600-900ms | 1.5s | Time to first streamed token |
| Slow path (full synthesis, complete) | 2-3s | 4-5s | Full advisory text, streamed in progressively |

---

## 6. Honest Tradeoffs

- **Coverage vs. speed:** the fast path only works because it trades
  full generative flexibility for pre-approved templates. This is fine
  for "how do I treat X disease" but won't work for open-ended
  conversational follow-ups ("what if I already sprayed neem oil
  yesterday?") — those must go to the slow path regardless of latency
  target.
- **Template staleness:** pre-written answers need a review/refresh
  cadence (weekly is reasonable) or they'll drift from current
  pricing/policy data — this is why price/dosage are still slotted in
  live rather than baked into the template text itself.
- **Cold-start regions:** a new state/language with no historical query
  data won't have fast-path coverage on day one — expect most traffic
  to route to slow path initially until templates are seeded.
