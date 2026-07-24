# FarmerVision — High-Level Design (v2)

Revised architecture addressing caching, parallel tool-calls, guardrail placement,
external-API resilience, and edge/offline handling identified in the review.

## 3. Subsystem Breakdown & Component Responsibilities (Revised)

```
                                          ┌─────────────────────────────────────┐
                                          │   On-Device Triage (TFLite/ONNX)     │
                                          │   Offline first-pass disease guess   │
                                          └───────────────┬───────────────────────┘
                                                          │ (works offline, low confidence)
                                                          ▼
    [Flutter/Kotlin Mobile Client] ──────────────────────┐
    (Camera + client-side image compression)              │
    (Native Cache: last N responses, offline queue)        │
        │                                                  │
        │ gRPC over Protobuf / HTTP2 (compressed payload)  │
        ▼                                                  │
    [Java Backend Core — Spring Boot 3.x]                   │
    (AuthN/Z, Orchestration, Rate Limiting, Trace Context)   │
        │                                                    │
        ├──➔ [Redis Cache Layer] ◄───────────────────────────┘
        │     • Query embedding cache (hash of normalized text)
        │     • Vector search result cache (short TTL)
        │     • Mandi/Weather API response cache (TTL: hours)
        │     • Session / conversation state
        │
        ├──➔ [External APIs: Agmarknet Mandi / Weather]
        │     via Resilience4j: Circuit Breaker + Retry/Backoff
        │     Timeout: 2500ms (was 250ms) | Fallback: last cached value
        │
        ├──➔ [Vector DB Cluster (Qdrant / ChromaDB)]
        │     HNSW Indexing / Cosine Similarity
        │     • Read replicas (query path)
        │     • Isolated writer node (offline indexing path — no live query impact)
        │
        └──➔ gRPC fan-out (parallel, not sequential) ──┐
                                                         ▼
                                          [Python ML Mesh — vLLM/Triton batched serving]
                                          │
                                          ├──➔ [Bhashini / IndicTrans2 Service]
                                          │
                                          ├──➔ [MuRIL Embedding Instance]
                                          │       └─ writes-through to Redis embedding cache
                                          │
                                          ├──➔ [Triton Server: ViT-Small / ECE Vision Model]
                                          │
                                          ├──➔ [Guardrails Pre-Filter]
                                          │       Fast rule/classifier check on retrieved
                                          │       dosage/policy context BEFORE synthesis
                                          │       (rejects bad tool data early — cheap)
                                          │
                                          ├──➔ [Gemma 12B ReAct Agent — synthesis only]
                                          │       Tool calls fan out in PARALLEL:
                                          │       (vector search ‖ mandi/weather ‖ yield ML)
                                          │       not sequential — cuts stacked latency
                                          │
                                          └──➔ [Guardrails Post-Check]
                                                  Lightweight regex/classifier, not
                                                  another full LLM pass
        │
        ▼ (Asynchronous Out-of-Band Telemetry — schema-registered Avro/Protobuf)
    [Apache Kafka / Cloud Pub/Sub]
        │
        ▼
    [Offline Data Lake & Workers]
    (EvidentlyAI Drift Detection / Prefect Retraining)
    — decoupled from live serving path, cannot back-pressure consumers —

    ══════════════════════════════════════════════════════════════════════════
    Cross-cutting: OpenTelemetry trace context propagated
    Mobile → Java → gRPC → Python ML Mesh → Kafka (single trace ID end-to-end)
    ══════════════════════════════════════════════════════════════════════════
```

## 4. What Changed vs. v1, and Why

| # | Change | Bottleneck it addresses |
|---|--------|--------------------------|
| 1 | Redis cache layer (embeddings, vector search, mandi/weather) | Repeated identical queries hitting LLM/embedder/external API on every request |
| 2 | Parallel tool-call fan-out in ReAct agent | Sequential tool calls stacking latency (4–8s → target ~2–3s) |
| 3 | vLLM/Triton batched serving for Gemma 12B | Poor throughput under concurrent farmer load |
| 4 | Guardrails split: pre-filter + lightweight post-check | Full LLM synthesis wasted on answers later rejected; dosage-safety risk |
| 5 | Timeout 250ms → 2500ms + circuit breaker + fallback | Constant failures against slow gov't APIs; silent degraded UX |
| 6 | Client-side image compression + on-device triage model | Rural 2G/3G bandwidth is the real bottleneck, not backend compute |
| 7 | Vector DB read replicas + isolated writer node | Offline indexing jobs degrading live query latency |
| 8 | Schema registry on Kafka topics | Silent breakage of drift detection / retraining on schema drift |
| 9 | OpenTelemetry trace propagation across language boundary | No way to debug "why did this take 9 seconds" across Java↔Python hop |

## 5. Suggested Rollout Priority

1. Redis caching layer — cheapest, highest impact
2. Parallel tool-call fan-out + vLLM/Triton batching
3. Circuit breakers + realistic timeouts on external APIs
4. Client-side image compression / on-device triage
5. Vector DB read replicas
6. Guardrails pre-filter
7. Distributed tracing
