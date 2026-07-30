# Milestone 4 — RAG Vector Database and Generation

**Notebooks:** `08c_rag_vector_db_bge_m3.ipynb` (index build) · `10_rag_generation.ipynb` (retrieval + synthesis)
**Run config hash:** `d9f0e9118d6e` · **Prepared:** 31 July 2026 · **Status:** built and measured

---

## 0. What this document is

Milestone 3 was a design document with one built exception — retrieval. This records the two
notebooks that turned that exception into a working end-to-end path: the vector database that
serves the corpus, and the generation notebook that puts an LLM behind it and measures what
comes out.

Everything below is **measured on the production index**, not estimated. Where a number
contradicts an M3 claim it is reported as measured, not reconciled. The embedder bake-off that
selected bge-m3 is not repeated here — see `RAG_Vector_DB_Audit_and_Runbook.md`.

**One thing to state before any numbers are read: the generator is not trained.** It is the M3
§5.4 off-the-shelf fallback (`google/gemma-3-4b-it`, 4-bit) run zero-shot.

**As of 31 July 2026 this is the final generator — distillation is dropped.** M3 §5.4 named the
off-the-shelf model a legitimate fallback "if distillation is constrained", and it is: the
distillation plan's own §8 throughput table puts a 12B teacher at ~3–4 tok/s on a T4, and its §15
risk #1 states outright that the teacher does not fit the schedule on free hardware. The write-up
follows the template in §16 of that plan. See [§C.5](#c5-distillation-dropped--what-that-does-and-does-not-change).

One reason that is **not** valid and should not appear in the report: that distillation would
have worsened the 11 s latency. The planned method is QLoRA on this same 4B checkpoint — a
rank-16 adapter, mergeable into the base weights at zero inference cost. Parameter count and
tokens/sec would have been unchanged. The 11 s is the serving stack (§C.2), not model capacity.

---

## Table of contents

1. [Part A — Vector database build](#part-a--vector-database-build)
2. [Part B — Generation](#part-b--generation)
3. [Key findings](#key-findings)
4. [Artifacts](#artifacts)
5. [Reproduction runbook](#reproduction-runbook)
6. [Open items for Milestone 5](#open-items-for-milestone-5)

---

# Part A — Vector database build

`08c_rag_vector_db_bge_m3.ipynb`. Run once, ~4 hours from scratch (~3 h of it GPU encoding).

## A.1 Corpus as indexed

| Measure | Value |
|---|---|
| Total chunks | **723,439** |
| KCC | 716,303 |
| PDF | 7,136 |
| Unique `chunk_id` | 723,439 — verified, no collisions |
| Embedding model | `BAAI/bge-m3`, frozen |
| Vector dimension | 1024 |
| Distance | cosine |
| Sequence cap at build | **512 tokens** |

**Language, re-detected per chunk.** This corrects a Milestone 2 assumption that came from
tagging queries only:

| source | language | chunks |
|---|---|---|
| kcc | mixed | 698,283 |
| kcc | en | 9,522 |
| kcc | hi | 8,498 |
| pdf | en | 7,052 |
| pdf | mixed | 84 |

KCC is overwhelmingly **code-mixed**, not the "~99.98% English" the query-only tag implied.
That matters downstream: it is why the distillation pipeline's script-agreement filter removes
most of the corpus (see `Synthesis_LLM_Distillation_Plan.md`), and why the Hinglish retrieval
gap in M3 §14 is a first-order problem rather than an edge case.

**Top crops after canonicalization:** rice 112,269 · wheat 95,836 · sugarcane 58,646 ·
potato 40,818 · mustard 39,147 · mango 36,809. These counts are the direct evidence that the
bracketed-crop-name fix worked — before it, `crop="rice"` matched **0** rows.

## A.2 Build mechanics, and why they are shaped this way

**512-token cap, pinned deliberately.** bge-m3 accepts 8192. The cap is set explicitly so every
model notebook truncates input identically — the embedder bake-off was a controlled comparison
in which only the model was allowed to vary. It also keeps a T4 from exhausting memory.

**Shard-streamed embedding, 20,000 chunks per shard (37 shards).** The full embedding array is
never held in RAM at once; an earlier one-shot version OOM-crashed because it held the whole
array *while* Qdrant built the HNSW graph. Shards are checkpointed to an embedding cache, so an
interrupted Colab runtime resumes at the last completed shard instead of repeating hours of GPU
work. Shard size is a crash-cost dial: at 50k, a crash lost up to 50k chunks of encoding.

**HNSW indexing deferred until after bulk load.** The graph is built once at the end rather than
incrementally after each batch.

**Server mode is mandatory.** In Qdrant's embedded local mode both `HnswConfigDiff` and the
payload indexes are silently ignored and every query becomes an O(n) scan. Nothing errors — it
is only slow, which is exactly why it is easy to ship by accident.

## A.3 Threshold calibration

Thresholds are **derived from the measured score distribution on the real index**, then written
into `manifest.json` so the serving path reads them rather than having them retyped:

```
in-domain   p50 = 0.683   min = 0.590
off-domain  p90 = 0.526   max = 0.530
-> TIER_FALLBACK = 0.56    TIER_GROUNDED = 0.66
```

The Milestone 1 values (0.85 / 0.65) were calibrated against a different embedder. Applied
unchanged to bge-m3 they would refuse **every query, including exact matches** — similarity
scales are not comparable across models.

## A.4 Retrieval evaluation (as run in `08c` §5)

| Check | Result |
|---|---|
| A1 pdf-only + scheme category | **PASS** — 5 hits, top 0.681, grounded, 210 ms |
| A2 kcc-only + crop=rice | **PASS** — 5 hits, top 0.801, grounded, 586 ms |
| A3 dosage tables only (pdf) | **PASS** — 5 hits, top 0.788, grounded, 464 ms |
| A4 district canon `allahabad→prayagraj` | **PASS** — 5 hits, top 0.638, fallback, 996 ms |
| A5 season filter (kcc Rabi) | **PASS** — 5 hits, top 0.811, grounded, 492 ms |
| A6 `year_from=2020` | **PASS** — 5 hits, top 0.625, fallback, 982 ms |
| B bilingual (wheat irrigation) | **WARN** — EN~HI jaccard 0.00 |
| B bilingual (rice blast) | **WARN** — EN~HI jaccard 0.00 |
| C1 policy intent → pdf-heavy top-3 | **PASS** — 3/3 |
| C2 field intent → kcc-heavy top-3 | **PASS** — 3/3 |
| D1 domain separation | **PASS** — in-domain min 0.590, off-domain max 0.530, margin **+0.059** |

**The two B warnings are a broken check, not a broken system.** It compares whether English and
Hindi phrasings return the *same record IDs*. Across 112,269 rice chunks both can return entirely
correct but different records and score 0.00. Its output is disregarded, and it is recorded here
rather than deleted, because a check known to be wrong is worse than no check if someone later
reads it as signal. Rewriting it to compare retrieved *topics* is an M5 task.

---

# Part B — Generation

`10_rag_generation.ipynb`. Retrieval is loaded from the snapshot; nothing is rebuilt.

## B.1 Configuration

| Setting | Value |
|---|---|
| Generator | `google/gemma-3-4b-it` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Training | **none — zero-shot** |
| temperature | 0.3 |
| top_p | 0.9 |
| max_new_tokens | 100 (M3 §10 caps answers at 60–100) |
| ctx_top_k | 5 |
| Eval set | 24 queries across policy / field / hindi / hinglish / gap / offdomain |

## B.2 The tier gate

Retrieval tier decides whether the generator runs at all:

| Tier | Raw score | Behaviour |
|---|---|---|
| `grounded` | ≥ 0.66 | answer and cite |
| `fallback_with_disclaimer` | 0.56 – 0.66 | answer + "verify with your local KVK" |
| `abstain_out_of_scope` | < 0.56 | refuse; **the generator is never invoked** |

Confidence is computed on the **raw** score, never the intent-weighted one. Weighting reorders
candidates; letting it feed the threshold would let a multiplier manufacture confidence the
underlying match does not support.

## B.3 Retrieval sweep (no LLM)

| config | crop_hit | route_acc | abstain | margin | p50 ms |
|---|---|---|---|---|---|
| top_k=3 | 0.846 | 1.000 | 0.958 | 0.070 | 3337.7 |
| top_k=5 | 0.923 | 1.000 | 0.958 | 0.070 | 45.8 |
| top_k=10 | **1.000** | 1.000 | 0.958 | 0.070 | 52.4 |
| intent=off | 0.923 | **0.889** | 0.958 | 0.070 | 39.0 |
| fusion=flat 1.0/1.0 | 0.923 | **0.889** | 0.958 | 0.070 | 39.2 |
| fusion=3.0/0.3 | 0.923 | 1.000 | 0.958 | 0.070 | 37.3 |
| fusion=manifest (baseline) | 0.923 | 1.000 | 0.958 | 0.070 | 37.8 |

**Read `margin` first** — it is the headroom the abstain tier needs, and it is unchanged at 0.070
across every config. No setting here trades safety for ranking.

Two things the table settles: **intent weighting earns its place** (route accuracy 1.000 → 0.889
without it, and flat fusion is identical to switching intent off, which is the expected result
and a useful consistency check); and **top_k=10 buys the last 7.7% of crop_hit for ~7 ms**, which
is affordable. The `top_k=3` p50 of 3337 ms is a cold-cache artifact of being the first config
run, not a property of top_k — the ordering makes that visible rather than hiding it.

## B.4 Generation sweep (12-query subset)

| config | n | ground | lang | cited | out_tok | tok/s | gen_ms |
|---|---|---|---|---|---|---|---|
| baseline | 12 | 0.907 | 0.33 | 0.92 | 82.2 | 7.3 | 11396 |
| temperature=0.0 | 12 | 0.912 | 0.33 | 0.92 | 81.9 | 7.4 | 11161 |
| temperature=0.7 | 12 | 0.917 | 0.33 | 0.92 | 80.1 | 7.3 | 10844 |
| max_new_tokens=60 | 12 | 0.844 | 0.33 | 0.75 | 57.5 | 6.8 | 8554 |
| max_new_tokens=150 | 12 | **1.000** | 0.33 | **1.00** | 90.2 | 7.5 | 11984 |
| ctx_top_k=3 | 12 | 0.908 | **0.42** | 1.00 | 66.0 | 7.6 | 8643 |
| ctx_top_k=8 | 12 | 0.942 | **0.42** | 0.92 | 77.8 | 6.4 | 12103 |

> **`ground` is a lower bound, not a measurement.** The scorer that produces this column has a
> confirmed false-positive defect (§C.1): it splits numbers on thousands separators, so a context
> reading "Rs 6,000" does not match an answer reading "6000". Every figure in this column is
> therefore **at least** the value shown. Do not quote the column as a grounding rate until the
> scorer is fixed and the sweep re-run.

**The clearest result is that truncation degrades citation.** `max_new_tokens=60` is the worst
config on citation rate (0.75); `max_new_tokens=150` is the only config reaching 1.00. The
mechanism is mundane — the citation line comes last, so cutting the answer short removes it.
Citation rate is computed by `has_citation()` and is **not** affected by the §C.1 scorer defect,
so this finding stands on its own. **The 100-token cap in M3 §10 is in direct tension with the
§10 citation requirement**, and this sweep is the evidence.

The apparent grounding ordering (0.844 at 60 tokens, 1.000 at 150) points the same way but cannot
be relied on independently, since it is the corrupted column. Temperature barely moves anything
on any axis (0.0–0.7).

**`lang` at 0.33 is alarming and is not what it looks like.** The metric requires the answer's
script to match the question's. Because it is computed over a subset that is ¾ non-English while
the corpus answers are overwhelmingly code-mixed, it penalises answers that are substantively
correct. See §C.3 — it needs rewriting before it is quoted.

## B.5 Health check

| # | Check | Result |
|---|---|---|
| R1 | filter kcc + crop=rice (canon `Paddy (Dhan)`) | **PASS** — 5 hits |
| R2 | district canon `allahabad→prayagraj` | **PASS** — 5 hits |
| R3 | domain separation margin > 0.02 | **PASS** — in-min 0.593, off-max 0.522, margin +0.070 |
| G1 | abstain never invokes the generator | **PASS** — 0/4 generated |
| G2 | every number traces to context | **FAIL as reported — 12/84.** Manually reviewed and found to be a **scorer defect, not a model failure** (§C.1) |
| G3 | Hindi question → Hindi answer | **PASS** — 0/21 mismatched |
| G4 | fallback tier carries the KVK line | **PASS** — 35 answers |
| G5 | answered queries cite their sources | **PASS** — 77/84 |

**7 pass / 1 fail as reported. After manual review of the G2 instances, effectively 8/8** — the
single failure is in the scorer, not the system (§C.1). The run output is left unedited above;
the correction is recorded rather than retro-fitted, because the raw output is what
`health_check.json` contains.

Note that G3 passes (0/21 script mismatches) while the sweep's `lang` metric reads 0.33 — the two
disagree because they measure different things, and G3 is the one built to test the actual rule.
That is now the second of three scoring defects found in this run (§C.3).

## B.6 Latency decomposition (n = 84)

| stage | p50 ms | p95 ms | share of p50 |
|---|---|---|---|
| embed | 21.2 | 34.1 | 0.2% |
| search | 19.3 | 31.9 | 0.2% |
| **generation** | **11011.6** | **15374.5** | **99.5%** |
| total | 11065.1 | 15413.6 | |

---

# Key findings

## C.1 G2 is a scorer defect, not a hallucination

The run reported **12 of 84 answers containing a number with no source in any retrieved chunk**:

- `pm kisan samman nidhi eligibility and benefits` → unsupported `6000`, `4`, `5`
- `गेहूं में पीला रतुआ की रोकथाम कैसे करें` → unsupported `1`, `2`, `3`, `4`

**The retrieved chunks were inspected by hand and the figures are present in them.** The model
did not invent ₹6,000; it read it. G2 is measuring the scorer, not the system.

**Mechanism — thousands separators.** `numeric_grounding()` extracts numbers with
`NUM_RE = \d+(?:\.\d+)?`, which has no notion of a thousands separator. A context reading
`Rs 6,000` tokenizes to `['6', '000']`; an answer reading `6000` tokenizes to `['6000']`. The
membership test then fails on a figure that is verbatim present. Reproduced directly:

| context | answer | ctx tokens | ans tokens | grounding | flagged |
|---|---|---|---|---|---|
| `Rs 6,000 per year` | `Rs 6000 per year` | `['6','000']` | `['6000']` | **0.000** | `['6000']` |
| `Rs 6000 per year` | `Rs 6000 per year` | `['6000']` | `['6000']` | 1.000 | — |
| `Rs. 2,000 each in 3 instalments totalling 6,000` | `3 instalments of 2000, total 6000` | `['2','000','3','6','000']` | `['3','2000','6000']` | **0.333** | `['2000','6000']` |

Government scheme documents write currency with separators as a matter of course, so this defect
fires hardest on exactly the content it most needs to score correctly.

**A second, independent contributor: truncation.** Answers cut at the 100-token cap lose their
trailing `Sources: [n]` line. That is what `has_citation()` measures (G5, 77/84) — it does not
feed `numeric_grounding()`, which strips citation markers before extracting numbers. So
truncation explains the citation gap, not the G2 count; the two failures are separate and were
initially conflated.

**The small integers are a third, distinct false positive.** `1, 2, 3, 4` in the Hindi answer are
list enumeration, not quantities. The scorer has no way to tell an enumerated step from a dose.

**What must not be written.** The 12/84 figure is not a hallucination rate and must not be quoted
as one. The correct statement is: *no confirmed hallucination was found in this run; the G2
failure was traced to a scorer that cannot parse thousands separators or distinguish list markers
from quantities.*

**What this does not clear.** A grounding check that cannot be trusted is a safety check that is
not running. The dosage-invention risk M3 §14 identifies remains **unmeasured**, not disproven —
this run produced no evidence either way. Fixing the scorer and re-running is a safety task, not
a tidy-up, and it is the highest-priority item in §M5.

**Process point, and the reason this section exists in this form.** A silent failure was found
where one was designed for — but it was in the *instrument*, not the system, and the first draft
of this document reported it as a confirmed model hallucination on the strength of the scorer's
output alone. That is the same error the M3 §9.10 health-check post-mortem records: trusting a
check without verifying the check. The finding was corrected only because someone opened the
chunks and looked.

## C.2 Latency misses the M3 §1.2 budget by ~40×, and generation is the whole reason

Measured p50 is **11,065 ms** against a 200–300 ms end-to-end target. **Retrieval is not the
problem** — embed + search together are 40.5 ms, 0.4% of the total. Generation is 99.5%.

This also corrects M3 §12.3, which named measured *retrieval* latency (p50 466 ms on 2-core
Colab) as the report's most significant unresolved problem. With a GPU present, retrieval is two
orders of magnitude inside budget; the budget risk was always generation.

The measurement is not on target hardware: Appendix A specifies FP8 on vLLM, this is 4-bit
bitsandbytes on a Colab T4 at ~7.3 tok/s. The decomposition says *which component to fix* rather
than licensing a conclusion that the architecture is wrong. What it does establish is that
optimizing retrieval further would be wasted effort.

## C.3 Three metrics are known-broken and must not be quoted

| Metric | Problem | Fix |
|---|---|---|
| `numeric_grounding` → G2, sweep `ground` | Splits numbers on thousands separators (`6,000` → `6`,`000`); counts list markers as quantities (§C.1) | Normalise separators both sides before comparison; ignore small integers adjacent to list punctuation |
| Bilingual retrieval check (`08c` B) | Compares record *IDs*; across 112,269 rice chunks two correct answers score 0.00 | Compare retrieved topics |
| `lang` in the generation sweep | Reads 0.33 while the purpose-built G3 check reads 21/21 pass | Align with G3's definition |

Three of the run's scoring instruments are defective, and the one that matters most is the safety
check. Every one of them failed *toward alarm* rather than silence, which is the survivable
direction — but none of them was validated before use. That is the M3 §9.10 lesson repeating.

## C.4 The 100-token cap conflicts with the citation rule

M3 §10 caps answers at 60–100 tokens. §B.4 shows `max_new_tokens=60` degrades citation rate to
0.75 while 150 reaches 1.00 — the `Sources: [n]` line is last, so truncation removes it. Citation
rate comes from `has_citation()` and is unaffected by the §C.1 scorer defect, so this stands.

A cap chosen for latency is costing traceability on a path where generation already dominates
latency by 99.5% — so the cap is not buying what it was introduced to buy. Raising it to 150 costs
~600 ms on an 11 s path and recovers full citation coverage. Resolve in M5.

## C.5 Distillation dropped — what that does and does not change

**Decision, 31 July 2026: the zero-shot 4-bit `gemma-3-4b-it` measured here is the final
generator.** No distillation, no QLoRA fine-tune.

**Grounds.** M3 §5.4 pre-authorised the off-the-shelf model as "a legitimate fallback if
distillation is constrained." It is constrained: the distillation plan's §8 table puts a 12B
teacher at ~3–4 tok/s on a T4, its §15 risk #1 states the teacher does not fit the schedule on
free hardware, and §13's Tier 0 exists precisely for this case. The dataset pipeline was built
(`13a_distill_pipeline_kaggle.ipynb`) and is retained; it is not run.

**One argument that must not be used.** That distillation would have worsened the 11 s latency is
false. The planned method is QLoRA on *this same 4B checkpoint* — a rank-16 adapter, mergeable
into the base weights at zero inference cost. Parameter count and tokens/sec would be unchanged.
The 11 s is bitsandbytes 4-bit on a T4 at ~7.3 tok/s (§C.2); the model was never the reason.

**What is lost.** The §14 dosage-invention risk was the thing fine-tuning was meant to reduce, by
teaching the model to say "the figure is not available" instead of reaching for parametric
memory. That behaviour is now carried entirely by the retrieval-side tier gate and the
post-generation checks — and per §C.1, one of those checks does not currently work. The
architecture's safety story therefore rests on fewer working layers than M3 assumed.

**What is not lost.** Citation coverage (77/84), language fidelity (21/21 on G3), and abstain
correctness (G1, 0/4 generated below threshold) are all measured behaviours of the untrained
model, and all hold. The quality argument for skipping is defensible on those; it is not
defensible on grounding, which is unmeasured (§C.1).

**How to write it up:** follow §16 of `Synthesis_LLM_Distillation_Plan.md`, which exists for this
case. State it as a scope decision with hardware grounds, not as a finding that distillation was
unnecessary — no comparison was run, so no such finding exists.

---

# Artifacts

**Index (`08c` §8):** written to `rag_production_bge_m3/`

| Artifact | Size | Required to serve |
|---|---|---|
| `agri_knowledge-*.snapshot` | 3.80 GB | **Yes** |
| `manifest.json` | ~1 KB | **Yes** — model, prefix policy, dim, thresholds, fusion weights |
| Embedding cache (~37 `.npy` shards) | ~1.5 GB | No — rebuild insurance |

The snapshot and manifest are a **matched pair**. Without the manifest the serving path must
guess prefix policy, vector dimension and thresholds, and every wrong guess fails silently.

**Generation (`10` §16):** `rag_generation_m4/`, config hash **`d9f0e9118d6e`**

| File | Contents |
|---|---|
| `generations.jsonl` | 90 rows — every answer with scores and citations |
| `run_manifest.json` | 1,043 B — full config |
| `retrieval_sweep.json` | 1,070 B |
| `generation_sweep.json` | 1,456 B |
| `health_check.json` | 977 B |
| `latency.json` | 240 B |

Quote `d9f0e9118d6e` next to every number taken from Part B.

---

# Reproduction runbook

**Build the index** — `08c_rag_vector_db_bge_m3.ipynb`, GPU, ~4 h from scratch or ~1 h from the
embedding cache. Run once; it is already done. Set `RESTORE_MODE = False` to build.

**Query it** — `09_rag_retrieval.ipynb`. Builds nothing, ~10 min to load.

**Generation + sweeps** — `10_rag_generation.ipynb`, GPU. `RUN_QUANT_SWEEP = False` by default
(it reloads the model 3× for ~15 min). Everything else on is ~10 min plus the snapshot restore.

Build and query paths are deliberately separate: building needs a GPU and hours, querying needs
neither.

**Two failure modes worth pre-empting:**

- Qdrant **must** run in server mode. Local mode ignores the HNSW graph and never errors.
- Restore reads thresholds, prefix policy and vector dim from `manifest.json`. Do not retype
  them; a threshold calibrated for another model refuses every query including exact matches.

---

# Open items for Milestone 5

1. **Fix `numeric_grounding` and re-run — safety-critical, do first.** Normalise thousands
   separators on both sides; stop counting list markers as quantities. Until this lands, the
   system's only check on invented dosages is **not running** (§C.1). With distillation dropped
   (§C.5) this is now the primary defence against the M3 §14 risk, not a secondary one.
2. **Re-run the generation sweep** once the scorer is fixed. The `ground` column in §B.4 is a
   lower bound and cannot be quoted until then.
3. **Raise the token cap to 150** (§C.4). Costs ~600 ms on an 11 s path, recovers full citation
   coverage. Then re-measure.
4. **Rewrite the other two broken metrics** (§C.3) — the bilingual check and `lang`.
5. **Re-measure latency on target-class hardware** with vLLM + FP8. Generation is 99.5% of p50, so
   this is where the entire latency budget lives. Update M3 §12.3, which currently names retrieval
   as the risk when the data says otherwise.
6. **Enable hybrid dense+sparse retrieval** against the M3 §14 Hinglish gap. bge-m3 emits sparse
   vectors from the same checkpoint; the corpus is 96% code-mixed (§A.1), so this is not an edge
   case.

**No longer an open item:** fine-tuning the generator. Dropped as a scope decision — see §C.5 for
the grounds and for the one argument not to use in support of it.

---

**Related:** `RAG_Vector_DB_Audit_and_Runbook.md` (embedder bake-off, bug post-mortems) ·
`Synthesis_LLM_Distillation_Plan.md` (the M5 student) · `docs/reports/Milestone_3_Report_Updated.md`
