# FarmerVision — Consolidated LLM / AI / ML Architecture

This is the single reference tying together every model and ML component
discussed: what each one does, what it's built from, how it's served, and
where it sits in the request flow. Setting: good connectivity (academic),
target latency 200-300ms.

---

## 1. Full Model Inventory

| # | Component | Purpose | Model | Size | Precision | Serving | Latency |
|---|---|---|---|---|---|---|---|
| 1 | Compound Intent/Entity Extractor | Detect all sub-intents + entities (crop, region, etc.) in ONE pass, before agent loop | DistilBERT-class, multi-label + NER head | ~66M | FP16 | ONNX Runtime | 10-15ms |
| 2 | Vision Classifier | Crop disease identification from image | ViT-Small (or MobileViT-XXS if pushed further) | ~22M-86M | FP16/INT8 | TensorRT | 15-25ms |
| 3 | Text Embedder | Embed farmer query + documents for semantic retrieval | MuRIL-base | ~236M | FP16 | TensorRT | 5-10ms |
| 4 | Structured Feature Embedder | Embed agronomic features (crop, region, soil, season) for yield-cache lookup — **separate from #3, not MuRIL** | Small trained two-tower encoder | ~5-10M | FP16 | ONNX Runtime | 2-5ms |
| 5 | Policy/KCC Vector DB | Retrieve relevant policy/incentive/advisory chunks | Qdrant, HNSW index | — | — | In-memory | 3-8ms |
| 6 | Yield Cache Vector DB | Fast lookup for recurring (crop×region×season) yield answers — **separate collection from #5** | Qdrant, HNSW index | — | — | In-memory | 5-10ms (hit) |
| 7 | Yield Prediction Tool | Compute yield for novel (crop×region×soil) combos — source of truth, not a retrieval shortcut | XGBoost/LightGBM (GBT) | — | — | Lightweight service | 10-15ms |
| 8 | Profitability/Risk Estimator | Combine yield × price × cost × incentive into a net-return range | Rule-based/statistical composite (not a trained model initially) | — | — | Lightweight service | 5-10ms |
| 9 | Mandi/Weather Tool | Live price + forecast lookup | External API + Redis cache | — | — | Cached | 5-10ms |
| 10 | Guardrails Pre-Filter | Validate retrieved data (dosage ranges, stale-price flags) before synthesis | Rule-based + small classifier | — | INT8 | ONNX Runtime | 2-5ms |
| 11 | **Main Synthesis LLM** | Generate the final farmer-facing advisory text | Gemma distilled to 2-4B, agri-domain fine-tuned | 2-4B | FP8 | vLLM, continuous batching, PagedAttention, guided decoding | 150-180ms |
| 12 | Speculative Draft Model | Propose candidate tokens verified by #11 in one pass — speeds up #11 | Small draft model, same tokenizer family | 350-500M | FP8 | vLLM speculative decoding | (folded into #11's latency) |
| 13 | Guardrails Post-Check | Hallucination/dosage/banned-term check on generated text | Rule-based regex + small classifier | — | INT8 | ONNX Runtime | 2-5ms |
| 14 | Fast-Path Template Cache | Skip generation entirely for the most common (disease/intent) pairs | Exact-match Redis lookup, pre-approved templates | — | — | Redis | 5-10ms (hit) |
| 15 | Speech-to-Text (ASR) | Transcribe farmer's spoken query, streaming | IndicWhisper / Bhashini ASR / IndicConformer (dialect-tuned) | ~250M-1.5B | FP16 | Faster-Whisper (CTranslate2) or Triton, **streaming mode** | duration-bound — see §5 |
| 16 | Dialect Normalizer | Normalize regional dialect transcript → standard Hindi/English before downstream NLP | IndicTrans2 (or dialect-specific fine-tune) | ~1.2B (distilled variant preferred) | FP16 | CTranslate2 / ONNX | 20-40ms (short text) |
| 17 | Text-to-Speech (TTS) | Convert final advisory text to spoken audio for low-literacy farmers | VITS/FastSpeech2-class, Indic multi-speaker | ~30-90M | FP16 | Streaming synthesis, chunked by sentence | see §5 |

---

## 2. Consolidated Architecture Diagram

```
        [Farmer Query]
   ┌───────────┼───────────┐
   │           │           │
 typed       image      spoken (voice)
 text       (photo)          │
   │           │             ▼
   │           │    ┌─────────────────────────┐
   │           │    │ #15 ASR (streaming) →      │
   │           │    │ #16 Dialect Normalizer       │
   │           │    │ (skipped for standard lang)   │
   │           │    │ → produces text, same as        │
   │           │    │   if farmer had typed it           │
   │           │    │ (full detail: §3)                    │
   │           │    └───────────┬─────────────────┘
   │           │                │
   └───────────┴────────────────┘
               │
               ▼
     [Farmer Query: text + optional image]
     ── from here on, voice/typed/image-only
        inputs are indistinguishable — this
        is the single entry point into the
        core pipeline below ──
                                            │
                        ┌───────────────────┴───────────────────┐
                        ▼                                       ▼
        ┌─────────────────────────────┐      ┌───────────────────────────────────┐
        │  #2 Vision Classifier          │      │  #1 Compound Intent/Entity          │
        │  (runs immediately if image     │      │     Extractor                       │
        │   present — NOT gated behind     │      │  → flags: needs_policy, needs_yield,│
        │   intent extraction, since its    │      │    needs_price, needs_profitability,│
        │   output is needed for the         │      │    is_compound; entities: crop,      │
        │   fast-path cache key below)        │      │    region, etc.                      │
        │  → disease_label, confidence         │      └──────────────────┬────────────────┘
        │              ~15-25ms                  │                        │
        └────────────────┬───────────────────┘                         │
                         └──────────────────┬───────────────────────────┘
                                            ▼
                    ┌───────────────────────────────────────────┐
                    │  Fast-Path Key Assembly:                     │
                    │  (disease_label, intent_flags, lang)          │
                    └───────────────────┬───────────────────────┘
                                        │
                     ┌──────────────────┴───────────────────┐
                     │                                        │
              key found in                              key MISS —
              #14 Template Cache                         novel/compound/
                     │                                    no-image query
                     ▼                                        ▼
        ┌─────────────────────────┐         ┌─────────────────────────────────┐
        │ #14 Fast-Path Template    │         │  Parallel Tool Fan-Out (all       │
        │      Cache — HIT           │         │  flagged branches fire together,  │
        │  → skip everything below   │         │  vision result reused from above  │
        │  ~5-10ms total (incl.       │         │  if it already ran — no re-run)   │
        │  vision + intent above)     │         │                                    │
        └─────────────────────────┘         │  #3 Text Embedder → #5 Policy DB  │
                                              │  #4 Feature Embedder → #6 Yield   │
                                              │      Cache DB (HIT→skip #7)        │
                                              │      MISS → #7 Yield Prediction    │
                                              │      (can use disease_label as a   │
                                              │       feature input if relevant)    │
                                              │  #9 Mandi/Weather Tool             │
                                              │  #8 Profitability Estimator        │
                                              │      (waits on #7 + #9 outputs)    │
                                              └────────────────┬───────────────────┘
                                                               ▼
                                              ┌─────────────────────────────────┐
                                              │  #10 Guardrails Pre-Filter        │
                                              │  (dosage bounds, stale-data flag) │
                                              └────────────────┬───────────────────┘
                                                               ▼
                                              ┌─────────────────────────────────┐
                                              │  #11 Main Synthesis LLM (2-4B)    │
                                              │  + #12 Speculative Draft Model     │
                                              │  + guided decoding (60-100 tokens) │
                                              │  + cached system/guardrail prefix  │
                                              │              ~150-180ms            │
                                              └────────────────┬───────────────────┘
                                                               ▼
                                              ┌─────────────────────────────────┐
                                              │  #13 Guardrails Post-Check        │
                                              │  (hallucination/banned-term scan) │
                                              └────────────────┬───────────────────┘
                                                               ▼
                                                     [Response to Farmer]

              Total (cache hit, incl.   ~35-50ms   (vision/intent ~15-25ms
              vision+intent+lookup):                + cache lookup ~5-10ms)
              Total (cache miss,        ~230-310ms  (adds the ~15-25ms vision/
              simple or compound):                   intent step ahead of the
                                                       ~210-290ms from before)

              ── if input was voice: add ~150-400ms ASR finalization
                 (+ ~20-40ms dialect normalization if triggered) from §3
                 BEFORE the pipeline above begins. Output side: if voice
                 reply requested, TTS (§3) streams alongside generation,
                 not stacked after it.
```

**Why this matters beyond just disease queries:** vision classification
and intent extraction are both cheap, independent, Tier-1 operations —
there's no reason to sequence one behind the other. Running them in
parallel upfront means the fast-path cache (which was originally designed
around `disease_label`, back in the very first fast-path architecture) is
actually reachable, instead of silently becoming dead code once compound
queries entered the picture.

---

## 3. Voice I/O Pipeline (Input & Output)

This wraps *around* the core pipeline in §2 — it doesn't replace the
text/vision entry point, it's an alternate front door for farmers who
speak rather than type, and a parallel exit for farmers who can't (or
prefer not to) read the response.

**Important framing:** the 200-300ms target in §2 is for the core
reasoning pipeline, measured from *transcript-ready* to *text-ready*.
Voice adds its own latency envelope on both ends that is **bound by audio
duration**, not just compute speed — you cannot transcribe a 6-second
utterance in 50ms regardless of GPU. The honest metric for voice isn't
"total ms," it's **added delay after the farmer stops speaking**, and
that's what streaming is for.

```
 Farmer speaks (e.g., 4-6 sec utterance in a regional dialect)
        │
        ▼ (streaming, overlaps with speech — not sequential after)
┌─────────────────────────────────────────────────────────────┐
│  #15 ASR (streaming) — IndicWhisper/Bhashini, dialect-tuned     │
│  Partial transcripts emitted continuously as farmer talks.       │
│  By the time farmer stops speaking, transcript is ~90% ready —   │
│  only a short "finalization" tail remains.                        │
│              Added delay after speech ends: ~150-400ms            │
│              (model RTF-dependent, not the full utterance length) │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  #16 Dialect Normalizer (IndicTrans2)                            │
│  Regional dialect transcript → standard Hindi/English            │
│  Only runs if farmer's dialect isn't the model's native training │
│  distribution — SKIPPED for standard Hindi/English input          │
│              ~20-40ms (short utterance-length text)                │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
                  [Enters core pipeline from §2 as
                   normal text input — same 200-300ms
                   budget applies from here]
                               │
                               ▼
                    [Response text generated]
                               │
                               ▼ (streaming — starts as soon as
                               │  first sentence of #11's output
                               │  is guardrail-checked, doesn't
                               │  wait for the full response)
┌─────────────────────────────────────────────────────────────┐
│  #17 TTS (streaming, chunked by sentence)                        │
│  First audio chunk starts playing while later sentences are       │
│  still being synthesized/generated                                 │
│              Time to first audio: ~150-250ms after first            │
│              guardrail-passed sentence is available                 │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    Farmer hears response begin
                    (well before full response is ready)
```

### Why streaming is not optional here, it's the actual design

Without streaming, voice would be: *wait for farmer to finish speaking →
wait for full transcription → wait for full 200-300ms pipeline → wait for
full TTS synthesis of the entire response* — stacked sequentially, easily
2-4+ seconds of dead silence before the farmer hears anything. That's a
bad product experience regardless of how optimized any single component
is. Streaming ASR and streaming TTS are what make voice *feel* responsive:
the farmer starts hearing an answer almost as soon as they stop talking,
even though the full response is still being generated underneath.

### Where dialect handling fits

Your original design flagged dialect audio as a stretch goal specifically
because dialect ASR is genuinely harder — standard Hindi ASR models
degrade noticeably on regional dialects with non-standard vocabulary and
pronunciation. Two honest options, not mutually exclusive:
- **Fine-tune the ASR model (#15) directly on dialect data** if you have
  it — better than a post-hoc normalization step, since errors introduced
  at the ASR stage can't be fully recovered by translation afterward.
- **Keep IndicTrans2 (#16) as a normalization layer** for dialects where
  you don't yet have enough data to fine-tune ASR directly — treat it as
  a stopgap, not the permanent solution, and plan to fold dialect-specific
  ASR fine-tuning in once you have transcribed dialect data (which,
  notably, your own usage logs will generate over time).

### TTS voice/language selection

Multi-speaker Indic TTS should select voice/language based on the
farmer's detected input language (from #1's language flag) or explicit
profile setting — not default to Hindi for all users, since your original
diagram already scopes toward multi-state/multi-language support in the
stretch goals.

### ASR confidence check on key entities (before entering core pipeline)

ASR errors on a crop name, district, or scheme reference don't just
corrupt the transcript — they silently feed a **wrong entity** into #1's
extraction, and downstream components (policy retrieval, yield lookup,
profitability estimate) will confidently act on it. This is worse for
compound queries (policy + profitability) than for a simple disease photo
query, since a wrong crop/region name changes financial-adjacent advice,
not just a disease label.

```
              Final transcript + per-token ASR confidence scores
                               │
                               ▼
              ┌───────────────────────────────────┐
              │  Check confidence on entity spans     │
              │  extracted by #1 (crop, region,         │
              │  scheme name) — NOT the whole utterance  │
              └───────────────┬───────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                                 │
      confidence ≥ threshold            confidence < threshold
      (e.g., >0.85)                     on a key entity
              │                                 │
              ▼                                 ▼
      Proceed directly into            Quick spoken confirmation:
      core pipeline, no                "आपने गेहूं कहा, सही?"
      added delay                      (short TTS prompt + short
                                        ASR listen for yes/no/correction)
                                                 │
                                                 ▼
                                        Added latency: ~1-2sec round trip,
                                        but only triggered on low-confidence
                                        key entities — not every query
```

This is a deliberate latency-vs-correctness tradeoff: it adds a visible
delay, but only on the subset of queries where ASR is genuinely unsure
about something that would otherwise silently corrupt a financial or
agronomic recommendation. Cheaper than the alternative of confidently
answering about the wrong crop.

### Worked example — voice input with an already-uploaded photo

Query (spoken, standard Hindi): *"मेरी गेहूं में ये बीमारी लगी है, क्या
करूं और सरकारी योजना क्या मिलेगी"* ("my wheat has this disease, what do
I do, and what govt scheme is available") — farmer also snapped a photo
of the crop before speaking.

```
Farmer taps mic (photo already attached)
        │
        ▼ audio streams in real-time as farmer talks
┌─────────────────────────────────────────────────────┐
│ #15 ASR (streaming) — partial transcripts build up        │
│ continuously. #2 Vision Classifier runs IN PARALLEL         │
│ on the attached photo — doesn't wait for voice to finish.     │
└──────────────────────┬──────────────────────────────┘
                       ▼ farmer stops talking → ~150-400ms finalization
        Final transcript + entity confidence scores
                       │
                       ▼
        All entity confidences ≥ threshold → no confirmation
        needed → #16 Dialect Normalizer SKIPPED (standard Hindi)
                       │
                       ▼
        Transcript enters core pipeline exactly like typed text.
        #1 extracts: needs_policy=true, needs_yield=false,
        is_compound=true, crop="wheat" (from transcript)
        Vision result already available: disease_label="wheat rust"
                       │
                       ▼
        Fast-path key MISS (compound: disease + policy) →
        Parallel Tool Fan-Out: Policy Vector DB (wheat scheme
        info) + Guardrails
                       │
                       ▼
        #11 Synthesis LLM generates response, streamed
                       │
                       ▼
        #17 TTS speaks response back — first audio chunk plays
        before full text generation completes
```

Note the convergence point: **once ASR produces text, voice and typed
input are indistinguishable to everything downstream** (§2's pipeline).
Voice is only a different way of producing input and delivering output —
it is not a separate reasoning path.

---

## 4. Why This Model Mix, Not a Single Big Model

The system deliberately avoids routing everything through the 12B model.
Three tiers, each sized to its job:

- **Tier 1 — sub-20ms classifiers/embedders** (#1, #2, #3, #4, #10, #13):
  small, task-specific, INT8/FP16, run on every request regardless of
  path. These are cheap enough that "always run them" is the right
  default rather than trying to skip them.
- **Tier 2 — lookup/compute tools** (#5, #6, #7, #8, #9): no LLM
  involved at all. Retrieval, cached API calls, and a GBT model. This is
  where most of the actual "intelligence" about yield/price/policy lives
  — not in the LLM's parametric knowledge.
- **Tier 3 — generation** (#11, #12): the only place a large-ish
  generative model runs, and only for turning already-retrieved,
  already-validated facts into fluent farmer-readable language. It is
  explicitly *not* the source of factual claims — guardrails treat any
  ungrounded number it produces as a hallucination to strip (see #13).

This separation is also the honest answer to "why not just make the LLM
bigger/smarter" — bigger models help with reasoning over ambiguous or
truly novel queries, but for a domain this structured (finite crops,
finite districts, numeric yield/price data), routing facts through
dedicated tools is both faster and more reliable than asking a generative
model to recall or compute them.

---

## 5. What's Still a Placeholder / Needs Real Design Work

Being direct about what's sketched vs. what needs its own LLD:

- **#8 Profitability Estimator** is described as rule-based here
  (yield × price − cost + incentive), but input-cost modeling
  (seed/fertilizer/labor pricing per crop/region) isn't designed yet —
  that's a real data-sourcing problem, not just an architecture one.
- **#12 Speculative decoding** requires the draft and main model to
  share a tokenizer and ideally come from the same model family
  (distillation lineage) — this constrains which base model you pick
  for #11 early on.
- **Distillation methodology for #11** (12B teacher → 2-4B student) is
  referenced but not yet specified — worth its own LLD covering
  training data, distillation objective, and the accuracy-eval plan
  mentioned earlier.
- **Dialect-specific training data for #15/#16** doesn't exist yet — the
  streaming ASR architecture in §3 is sound, but its real-world accuracy
  on non-standard dialects depends entirely on data you haven't
  collected. This was flagged as a stretch goal in your original
  diagram for exactly this reason, and that's still the right call —
  ship standard-language voice first, expand dialect coverage as usage
  data accumulates.
- **Confidence threshold for the ASR entity-confirmation step (§3)** is
  stated as an example (0.85) but needs actual calibration against a
  labeled dataset of transcription errors on entity spans specifically
  — too low a threshold triggers annoying unnecessary confirmations, too
  high lets bad entities silently through.
