# FarmerVision — Low-Level Design: Critical Path
### (Image Upload → ReAct Agent → Response)

Scope: everything that happens between a farmer tapping "submit" on a crop photo
and receiving synthesized advice. Covers API contracts, data models, sequence
flow, cache keys, timeout/retry config, and guardrail logic.

---

## 1. Sequence Diagram (single request, happy path)

```
Farmer      Java Backend     Redis        Vision Svc      MuRIL        Vector DB     Gemma Agent    Mandi/Wx API   Guardrails
  │              │             │              │             │              │             │              │             │
  │─1. POST /v2/diagnose───────►│             │              │              │             │              │             │
  │  {image, text, farmerId,   │             │              │              │             │              │             │
  │   lang, traceId}           │             │              │              │             │              │             │
  │              │             │              │             │              │             │              │             │
  │              │─2. GET embed:hash(text)───►│              │              │             │              │             │
  │              │◄──cache MISS───────────────│              │              │             │              │             │
  │              │             │              │             │              │             │              │             │
  │              │─3. gRPC ClassifyImage(image)──────────────►│              │             │              │             │
  │              │◄──{label, confidence, bbox}──────────────│              │             │              │             │
  │              │             │              │             │              │             │              │             │
  │              │─4. gRPC Embed(text)───────────────────────────────────►│              │             │              │             │
  │              │◄──{vector[768]}────────────────────────────────────────│              │             │              │             │
  │              │─5. SET embed:hash(text) → vector (TTL 24h)─►│           │              │             │              │             │
  │              │             │              │             │              │             │              │             │
  │              │─6a. VectorSearch(vector, topK=5)──────────────────────────────────────►│             │              │             │
  │              │─6b. GetMandiPrice(crop, district) [PARALLEL]───────────────────────────────────────────►│           │             │
  │              │─6c. EstimateYield(label, region) [PARALLEL]───────────────────────────►│ (async task) │              │             │
  │              │             │              │             │              │◄──chunks[5]──│             │              │             │
  │              │             │              │             │              │             │◄──price/fallback─│         │             │
  │              │             │              │             │              │             │              │             │
  │              │─7. GuardrailsPreFilter(chunks, price, label)────────────────────────────────────────────────────────►│
  │              │◄──{ok:true, filteredChunks}────────────────────────────────────────────────────────────────────────│
  │              │             │              │             │              │             │              │             │
  │              │─8. gRPC Synthesize(enrichedPrompt)────────────────────────────────────►│             │              │
  │              │◄──{advice_text, lang:"hi", citations[]}───────────────────────────────│             │              │
  │              │             │              │             │              │             │              │             │
  │              │─9. GuardrailsPostCheck(advice_text)──────────────────────────────────────────────────────────────────►│
  │              │◄──{ok:true} | {ok:false, reason}────────────────────────────────────────────────────────────────────│
  │              │             │              │             │              │             │              │             │
  │◄─10. 200 OK {advice, label, sources, traceId}──│         │              │             │              │             │
  │              │             │              │             │              │             │              │             │
```

**Key change vs. v1:** steps 6a/6b/6c fire concurrently (`Promise.all` / `CompletableFuture.allOf`), not sequentially. Guardrails runs twice — cheap pre-filter (step 7, on raw retrieved data) and cheap post-check (step 9, regex/classifier on final text) — the LLM is never re-invoked for validation.

---

## 2. API Contract — `POST /v2/diagnose`

**Request**
```json
{
  "farmerId": "string (uuid)",
  "traceId": "string (uuid, generated client-side)",
  "image": "base64 or multipart, max 2MB (client compresses before send)",
  "text": "string, farmer's free-text query, max 500 chars",
  "lang": "string, ISO 639-1 (e.g. hi, en, bho)",
  "location": { "lat": "float", "lng": "float", "district": "string" }
}
```

**Response — 200**
```json
{
  "traceId": "string",
  "diseaseLabel": "string | null",
  "confidence": "float (0-1)",
  "advice": "string (in farmer's lang)",
  "sources": [
    { "type": "policy|kcc|mandi", "ref": "string", "score": "float" }
  ],
  "priceContext": { "crop": "string", "mandiPrice": "float", "asOf": "ISO8601", "stale": "boolean" },
  "guardrailsFlag": "none | dosage_capped | escalate_to_expert"
}
```

**Error responses**
| Code | Condition | Client behavior |
|---|---|---|
| 400 | Image >2MB or unsupported format | Show re-compress prompt |
| 422 | Guardrails post-check rejected advice | Show generic safe fallback message |
| 503 | Vision/Embed service down | Serve from Native Cache (last similar query) if available |
| 504 | Gemma synthesis timeout (>8s) | Return partial answer (retrieved chunks only) with "advice generation delayed" flag |

---

## 3. Internal Service Contracts (gRPC, proto sketch)

```protobuf
service VisionService {
  rpc ClassifyImage (ImageRequest) returns (ClassifyResponse);
}
message ImageRequest { bytes image_data = 1; string trace_id = 2; }
message ClassifyResponse {
  string disease_label = 1;
  float confidence = 2;
  repeated float bbox = 3; // optional localization
}

service EmbeddingService {
  rpc Embed (TextRequest) returns (EmbedResponse);
}
message EmbedResponse { repeated float vector = 1; } // dim=768, MuRIL

service ReActAgentService {
  rpc Synthesize (SynthesizeRequest) returns (SynthesizeResponse);
}
message SynthesizeRequest {
  string enriched_prompt = 1;
  repeated Chunk retrieved_chunks = 2;
  string price_context_json = 3;
  string trace_id = 4;
}
message SynthesizeResponse {
  string advice_text = 1;
  string lang = 2;
  repeated string citation_ids = 3;
}
```

---

## 4. Cache Keys & TTLs (Redis)

| Key pattern | Value | TTL | Notes |
|---|---|---|---|
| `embed:{sha256(normalizedText)}` | float[768] vector | 24h | Normalize: lowercase, strip punctuation, trim before hashing |
| `vsearch:{sha256(vector_bucket)}` | top-5 chunk IDs + scores | 15 min | Bucketed via LSH pre-hash to catch near-duplicate vectors, not exact match |
| `mandi:{crop}:{district}` | price object | 4h | Refresh proactively via scheduled job, not just on-demand |
| `weather:{district}` | forecast object | 1h | |
| `session:{farmerId}` | last 3 queries + responses | 7d | Powers "Native Cache" fallback on mobile |

---

## 5. Timeout / Retry / Circuit-Breaker Config

| Call | Timeout | Retry | Circuit breaker |
|---|---|---|---|
| Vision Classify (gRPC) | 1500ms | 1 retry, no backoff | Trip at 50% failure / 20 req window |
| MuRIL Embed (gRPC) | 800ms | 1 retry | Trip at 50% / 20 req |
| Vector Search | 500ms | 0 (return cached/stale on miss) | Trip at 30% / 20 req |
| Mandi/Weather external API | 2500ms | 2 retries, exponential backoff (500ms base) | Trip at 40% / 10 req, half-open probe every 30s |
| Gemma Synthesize | 6000ms hard cap | 0 (too expensive to retry) | On timeout → return chunks-only partial response |
| Guardrails pre/post | 300ms each | 0 | Fail-closed: on error, treat as `escalate_to_expert` |

---

## 6. Guardrails Logic (both stages are rule/classifier-based, not LLM calls)

**Pre-filter (on retrieved chunks + tool data, before synthesis):**
1. Reject any KCC/policy chunk where extracted dosage value is outside a hardcoded safe range per pesticide category (lookup table, not inference).
2. Flag if `mandiPrice.stale == true` and `asOf` older than TTL — annotate, don't block.
3. Pass filtered chunk set + flags into the enriched prompt.

**Post-check (on generated advice text, after synthesis):**
1. Regex scan for numeric dosage mentions not present in the source chunks (hallucination check) → if found, set `guardrailsFlag = dosage_capped` and strip the unsupported number, replacing with "consult local agri officer for exact dosage."
2. Keyword blocklist for banned/restricted pesticide names → `escalate_to_expert`.
3. Language consistency check (response lang matches requested `lang`) → if mismatch, route to translation fallback rather than re-generating.

---

## 7. Failure Modes & Degraded Behavior

| Failure | Degraded response |
|---|---|
| Vector DB down | Skip retrieval, synthesize from tool data only, flag `sources: []` |
| Mandi API down (circuit open) | Use last cached price, `stale: true` |
| Gemma synthesis timeout | Return raw retrieved chunks + disease label, no synthesized prose |
| Guardrails service down | Fail-closed — never return unvalidated advice; return `escalate_to_expert` |
| Network drop mid-upload (mobile) | Client queues request in Native Cache, retries on reconnect with same `traceId` (idempotent) |
