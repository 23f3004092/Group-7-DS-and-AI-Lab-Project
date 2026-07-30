# FarmerVision — Milestone 4 Report
## Model Training, Hyperparameter Experiments, and Model Selection

**Version:** v1 · **Prepared:** 30 July 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Training Datasets](#2-training-datasets)
3. [Model Configuration](#3-model-configuration)
4. [Training Environment](#4-training-environment)
5. [Training Methodology](#5-training-methodology)
6. [Hyperparameter Experiments](#6-hyperparameter-experiments)
7. [Optimization Methods](#7-optimization-methods)
8. [Regularization Techniques](#8-regularization-techniques)
9. [Training Progress](#9-training-progress)
10. [Model Selection](#10-model-selection)
11. [Challenges Encountered](#11-challenges-encountered)
12. [Summary and Next Steps](#12-summary-and-next-steps)

**Appendices:** [A — Vision phase configs](#appendix-a--vision-phase-configurations) · [B — Training logs](#appendix-b--training-logs) · [C — Vision per-class test results](#appendix-c--vision-per-class-test-results) · [D — Artifacts](#appendix-d--artifacts) · [E — Known-broken measurement instruments](#appendix-e--known-broken-measurement-instruments)

---

## 1. Introduction

### 1.1 Recap of Milestone 3

M3 selected one model per module: **ViT-Small** for disease classification, **BAAI/bge-m3** (frozen)
for retrieval embedding, a **DistilBERT-class three-head model** for intent/entity/guardrail, **GBT**
for yield, a **distilled 2–4B LLM** for synthesis, and a rule-based profitability estimator. Voice
(ASR/TTS) was excluded from capstone scope. Retrieval was the one component already built at M3;
everything else was design.

### 1.2 Objectives of Milestone 4

Train the selected models, run hyperparameter and optimization experiments, and justify the final
checkpoint for each. Two M3 open items were also in scope: the Intent/Entity data gap (§7.0.3) and
benchmarking the built index against the latency budget (§12.3).

### 1.3 What was and was not trained

| Component | M4 status |
|---|---|
| Vision (ViT-S/16) | **Trained** — 3 phases, held-out test evaluated once |
| Intent / Entity / Guardrail (DistilBERT) | **Trained** — intent and NER usable; guardrail head fails (§10.2) |
| Retrieval (bge-m3 + Qdrant) | **No training by design** (M3 §7.2, frozen); index built, swept, measured |
| Synthesis LLM | **Not trained.** Distillation dropped as a scope decision (§3.4); zero-shot `gemma-3-4b-it` is final |
| Yield | **Prototype only, not usable.** An MLP was run instead of the selected GBT; the run has target leakage and did not complete (§10.4) |

Two components were trained to completion, one is measured without training by design, and two have
no usable training result — stated plainly rather than presented as partially done.

---

## 2. Training Datasets

### 2.1 Splits actually used

| Model | Dataset | Split | Sizes |
|---|---|---|---|
| Vision | Merged 20-class rice+wheat corpus | **Scene-grouped, rebuilt in M4** (`clean_index_v3.csv`) | train 10,252 / val 1,284 / test 1,287 |
| Intent/Entity/Guardrail | KCC cleaned corpus, 30,000-row stratified sample + 8,000 AG News rows | 80/10/10, stratified on guardrail label | 30,400 / 3,800 / 3,800 |
| Retrieval | RAG PDF corpus + KCC UP 2020–2025 | No split — frozen embedder, no training | 723,439 chunks (716,303 KCC + 7,136 PDF) |
| Generation | — | No training set; **24-query eval set**, 84 generations | policy / field / hindi / hinglish / gap / offdomain |
| Yield (prototype) | `production_unified_imputed.csv` | Random 80/20 | 352,769 / 88,193 |

### 2.2 The vision split had to be rebuilt before training

The published split could not be used. Measured on frozen ViT features, **391 eval images (15.3% of
val+test) shared a photographic scene with a training image** at cosine > 0.95 — the rice sources are
burst-captured (`rice__tungro`: 461 images, only 207 distinct scenes). 36 images were also excluded
as degenerate slivers or cross-split near-duplicates.

The first repair (v2) made scenes atomic and eliminated leakage, but the score *rose* +3.7 points:
greedy "largest groups first" packing put big clusters into val/test, leaving the eval set only
**84% effective** (30.4% of `rice__tungro` was one repeated scene). Reversing the packing order —
singletons first, multi-image clusters to train — produced v3: **0 cross-split groups and 100%
effective size simultaneously.** All M4 vision numbers use v3. Two independent mechanisms would each
have inflated the reported score, and neither was visible without explicit measurement.

### 2.3 Preprocessing and augmentation applied at training time

**Vision (`mid_tf`, frozen in `pipeline.py`):** `RandomResizedCrop(224, scale 0.65–1.0, bicubic)` →
`HFlip(0.5)` → `RandomAffine(15°, translate 0.08, scale 0.92–1.08)` → `ColorJitter(0.35, 0.30, 0.25,
no hue)` → `RandomExposureGamma(target 0.18–0.55, p=0.9)` → `GaussianBlur(p=0.35)` →
`RandomAdjustSharpness(2.5, p=0.35)` → `Normalize(0.5, 0.5, 0.5)` → `RandomErasing(p=0.15)`.
Eval transform is deterministic `CenterCrop(224)` on a 256 image (crop_pct 0.875).

The policy was **selected against measured shortcut destruction**, not by convention. Two
non-pathological cues predicted the label: brightness identified `rice__leaf_smut` (AUC 0.821) and
blur separated rice from wheat (AUC 0.833). Multiplicative brightness jitter cannot remove the first
— it widens each class's distribution without moving its mean — and a scale-and-clamp exposure
transform failed for a second reason: clipping is asymmetric, so a dark class structurally cannot
reach the bright end. A **power-law tone curve** (`x → x^γ`, γ by bisection) is monotone and bijective
on [0,1], so no clipping occurs. Result: brightness AUC **0.821 → 0.552**, exposure-normalised
sharpness **0.803 → 0.595**.

**Intent/Entity:** stratified sampling by `QueryType`; taxonomy collapsed to top-11 + `other`;
bracketed KCC crop names (`Paddy (Dhan)`) parsed into alias surface forms; BIO tags produced by
**distant supervision** (string matching against `Crop`/`District`), covering 82.9% of rows; AG News
rows carry `-100` on intent/NER so they contribute only to the guardrail loss.

---

## 3. Model Configuration

### 3.1 Vision

`vit_small_patch16_224.augreg_in21k_ft_in1k` (timm 1.0.26), **pretrained**, ImageNet-21k → 1k.
21.67M params, classification head 384 → 20. Normalisation is the checkpoint's native
`(0.5, 0.5, 0.5)`, not ImageNet — selected empirically in §6.1.

### 3.2 Intent / Entity / Guardrail

`distilbert-base-multilingual-cased`, **pretrained**, with three linear heads on one backbone:
intent 768→12, NER 768→5 (`O, B-CROP, I-CROP, B-DISTRICT, I-DISTRICT`), guardrail 768→2. Max
sequence length 64. Total 134.7M params. One forward pass produces all three outputs, per M3 §7.9.

### 3.3 Retrieval

bge-m3 **frozen**, 1024-dim, cosine, no query/passage prefix, 512-token build cap. Qdrant HNSW
(`m=16`, `ef_construct=128`), server mode. Thresholds derived from the measured score distribution
(in-domain p50 0.683 / min 0.590; off-domain p90 0.526 / max 0.530 → `TIER_FALLBACK` 0.56,
`TIER_GROUNDED` 0.66) and written into `manifest.json`.

### 3.4 Synthesis LLM — distillation dropped

**Decision, 30 July 2026: the zero-shot 4-bit `google/gemma-3-4b-it` is the final generator.** M3 §5.4
pre-authorised an off-the-shelf model "if distillation is constrained." It is: the distillation plan's
own throughput table puts a 12B teacher at ~3–4 tok/s on a T4, and its risk register states the
teacher does not fit the schedule on free hardware. The dataset pipeline
(`13a_distill_pipeline_kaggle.ipynb`) was built and retained, but not run.

This is a **scope decision on hardware grounds, not a finding that distillation was unnecessary** —
no comparison was run. One argument must not be used in its support: that distillation would have
worsened the 11 s latency. The planned method was QLoRA on this same 4B checkpoint (rank-16 adapter,
mergeable at zero inference cost), so parameter count and tok/s would have been unchanged.

What is lost: fine-tuning was meant to reduce the M3 §14 dosage-invention risk by teaching the model
to say "the figure is not available" rather than reach for parametric memory. That behaviour now
rests entirely on the retrieval tier gate and post-generation checks — one of which is currently
defective (Appendix E).

Generation config: temperature 0.3, top_p 0.9, `max_new_tokens` 100, `ctx_top_k` 5, 4-bit NF4.

### 3.5 Yield (prototype)

An MLP (925 → 256 → 128 → 64 → 1, ReLU, dropout 0.3/0.3/0.2) was run, **not the GBT selected in M3
§5.3**. See §10.4 — this run is not a usable result and the M3 selection stands unchallenged by it.

---

## 4. Training Environment

| Component | Hardware | Software | Wall time |
|---|---|---|---|
| Vision | Kaggle, 2 × Tesla T4 (15.6 GB), 4 vCPU, 33.7 GB RAM | PyTorch, timm 1.0.26 | **~42 min total GPU** for all 3 phases + eval |
| Intent/Entity | Colab T4 (16 GB), 2 vCPU | PyTorch, HF `transformers`, `seqeval` | **~589 s** (3 epochs) |
| Index build | Colab GPU | `sentence-transformers`, Qdrant (server mode) | ~4 h from scratch (~3 h GPU encoding) |
| Generation | Colab T4 | `transformers` + `bitsandbytes` 4-bit NF4 | ~10 min for 84 generations + sweeps |
| Yield prototype | **CPU only** (`Using device: cpu`) | PyTorch, scikit-learn | Interrupted at epoch 2/10 |

**Resource constraints shaped design, not just schedule.** The 4 vCPU limit made the vision pipeline
CPU-bound in every configuration (best loader 263 img/s against a GPU ceiling of 373 img/s), which
motivated the augmentation cost audit in §6.4. The T4 memory limit is why the generator runs 4-bit,
why the index is embedded in 20,000-chunk shards, and why distillation was dropped (§3.4).

---

## 5. Training Methodology

### 5.1 Vision — progressive unfreezing, one change per phase

The schedule changed exactly **one structural thing per phase** so any gain was attributable.

| Phase | Trainable | Key config | Epochs | Time |
|---|---|---|---|---|
| D1 — head only | 7.7K (0.04%) | frozen backbone, lr 1e-3, cosine | 12 | 9 min |
| D2 — blocks 9–11 | 5.33M (25%) | lr 1e-4 backbone / 1e-3 head, 2-epoch warmup, grad-clip 1.0 | 20 | 15 min |
| D3 — full | 21.67M (100%) | LLRD 0.70 (base 3e-4), drop-path 0.10, 1-epoch warmup | 18 | 15 min |

Batch size 64 (160 optimiser steps/epoch), 4 workers, label-smoothed cross-entropy, cosine LR decay.
Selection metric throughout: **val wheat-15 macro-F1** (§10.1).

### 5.2 Intent / Entity / Guardrail — joint multi-task loss

Single loop, one backward pass over the summed loss:

| Head | Loss | Weight | Masking |
|---|---|---|---|
| Intent | CrossEntropy | 1.0 | `ignore_index=-100` on AG News rows |
| NER | Token-level CrossEntropy | 1.0 | `ignore_index=-100` on special/pad tokens |
| Guardrail | CrossEntropy, class-weighted | 1.0 | off-domain weight **3.75×** (recall-prioritized, M3 §7.9) |

AdamW, lr 3e-5, batch 32, weight decay 0.01, 3 epochs, max_len 64.

### 5.3 Retrieval and generation — configuration search, not training

No gradients. The workflow is: build index once → sweep retrieval configs without the LLM → sweep
generation configs on a 12-query subset → run a health check → decompose latency. Both sweeps are
in §6.5–6.6.

---

## 6. Hyperparameter Experiments

### 6.1 Vision — normalisation and class balancing (linear probe, frozen backbone, 60 epochs)

Three normalisations were defensible: the checkpoint's native `(0.5, 0.5, 0.5)`, ImageNet stats, and
our own dataset stats (which differ from ImageNet by 0.152 in mean). Measured rather than argued:

| norm | balance | macro-F1 | acc | min class F1 |
|---|---|---|---|---|
| **native (0.5)** | **none** | **0.8461** | 0.8564 | 0.507 |
| native | sqrt | 0.8453 | 0.8587 | 0.521 |
| imagenet | none | 0.8326 | 0.8470 | 0.496 |
| dataset | none | 0.8274 | 0.8470 | 0.478 |
| imagenet | sqrt | 0.8236 | 0.8423 | 0.460 |
| dataset | sqrt | 0.8216 | 0.8423 | 0.429 |

**Selected: native normalisation, no class balancing.** Matching the *pretraining* statistics beats
matching the target domain by +1.9 F1 — the opposite of the usual intuition. `sqrt` sampling was
neutral-to-negative on macro-F1 in every pairing.

### 6.2 Vision — batch size

| batch | img/s | s/epoch (GPU-bound) | peak memory |
|---|---|---|---|
| 32 | 341 | 30.0 | 1.6 GB |
| **64** | 360 | 28.5 | 2.7 GB |
| 96 | **373** | 27.4 | 3.8 GB |
| 128 | 368 | 27.9 | 4.9 GB |

**Selected: 64.** Throughput plateaus by 96 and memory is never the constraint (4.9 GB of 15.6 GB at
the largest setting). 64 is within 3% of peak throughput and gives 160 optimiser steps/epoch versus
106 at bs=96 — on a 10K-image fine-tune, step count matters more than 3% of wall clock. Workers were
swept separately and set to 4 (263 img/s, the best available on 4 vCPUs).

### 6.3 Vision — unfreezing depth (the experiment that mattered most)

| Metric | frozen probe | D1 head | D2 blocks 9–11 | D3 full |
|---|---|---|---|---|
| **wheat-15 macro-F1 (val)** | 0.7966 | 0.7682 | **0.8660** | **0.8793** |
| 20-way macro-F1 (val) | 0.8382 | 0.7655 | 0.8830 | 0.9042 |
| accuracy | 0.8559 | 0.8131 | 0.9042 | 0.9143 |
| train/val gap | — | 0.047 | 0.092 | **0.073** |

Cluster-bootstrap 95% CI on wheat-15 is **±0.025**; anything smaller is not a result.

- **D1 < probe is expected**, not a regression. The probe was trained on un-augmented features; D1 is
  the first run with the shortcut-destroying augmentation active, and a frozen backbone cannot learn
  invariance from augmentation — it only scatters the cloud the head must fit.
- **D1 → D2: +0.098**, nearly 4× the noise floor. Adapting the top three blocks is where the gain is.
- **D2 → D3: +0.013, inside the CI.** Reported as *comparable to D2*, not better — a 4× parameter
  increase bought no measurable return.

### 6.4 Vision — augmentation cost/benefit veto test

Before adopting any speed-up, a pre-registered check: does trimming augmentation cost the shortcut
kills?

| variant | ms/img | s/epoch | bright_auc | sharp_auc |
|---|---|---|---|---|
| `full_tf` (bicubic + hue + gamma i10/s31) | 11.51 | 54.5 | 0.549 | 0.584 |
| **`mid_tf` (hue removed)** | **8.41** | **42.1** | **0.560** | 0.605 |
| `fast_tf` (bilinear + coarser gamma) | 7.63 | 39.0 | 0.558 | 0.581 |

**A hue jitter of ±0.02 cost 3.10 ms/image — 27% of the whole pipeline — and moved no AUC outside the
±0.02 noise floor.** It was removed. `fast_tf` was rejected despite being faster: its bilinear
resampling touches fine lesion texture, which the audit does not measure. **Selected: `mid_tf`.**

### 6.5 Retrieval sweep (24 queries, no LLM)

| config | crop_hit | route_acc | abstain | margin | p50 ms |
|---|---|---|---|---|---|
| top_k=3 | 0.846 | 1.000 | 0.958 | 0.070 | 3337.7\* |
| top_k=5 | 0.923 | 1.000 | 0.958 | 0.070 | 45.8 |
| top_k=10 | **1.000** | 1.000 | 0.958 | 0.070 | 52.4 |
| intent weighting off | 0.923 | **0.889** | 0.958 | 0.070 | 39.0 |
| fusion flat 1.0/1.0 | 0.923 | **0.889** | 0.958 | 0.070 | 39.2 |
| fusion 3.0/0.3 | 0.923 | 1.000 | 0.958 | 0.070 | 37.3 |
| fusion manifest (baseline) | 0.923 | 1.000 | 0.958 | 0.070 | 37.8 |

\* cold-cache artifact of being the first config run, not a property of top_k.

**Read `margin` first** — it is the headroom the abstain tier needs, and it is unchanged at 0.070
across every config, so no setting trades safety for ranking. Two results settled: intent weighting
earns its place (route accuracy 1.000 → 0.889 without it; flat fusion is identical to switching
intent off, the expected consistency check), and top_k=10 buys the last 7.7% of crop_hit for ~7 ms.

### 6.6 Generation sweep (12-query subset)

| config | ground† | cited | out_tok | tok/s | gen_ms |
|---|---|---|---|---|---|
| baseline (t=0.3, 100 tok, k=5) | 0.907 | 0.92 | 82.2 | 7.3 | 11396 |
| temperature=0.0 | 0.912 | 0.92 | 81.9 | 7.4 | 11161 |
| temperature=0.7 | 0.917 | 0.92 | 80.1 | 7.3 | 10844 |
| max_new_tokens=60 | 0.844 | **0.75** | 57.5 | 6.8 | 8554 |
| max_new_tokens=150 | 1.000 | **1.00** | 90.2 | 7.5 | 11984 |
| ctx_top_k=3 | 0.908 | 1.00 | 66.0 | 7.6 | 8643 |
| ctx_top_k=8 | 0.942 | 0.92 | 77.8 | 6.4 | 12103 |

† **`ground` is a lower bound, not a measurement** — the scorer has a confirmed false-positive defect
(Appendix E). Do not quote this column as a grounding rate.

**The one finding that stands on its own is that truncation destroys citation.** `max_new_tokens=60`
gives the worst citation rate (0.75); 150 is the only config reaching 1.00. The mechanism is mundane
— the `Sources: [n]` line comes last, so cutting the answer removes it. Citation rate comes from
`has_citation()` and is unaffected by the scorer defect. **The 100-token cap in M3 §10 is therefore
in direct tension with M3 §10's own citation requirement.** Raising it to 150 costs ~600 ms on an
11 s path; resolve in M5. Temperature moves nothing measurable across 0.0–0.7.

### 6.7 Intent / Entity — no sweep was run

Hyperparameters are the M3 Appendix B starting points (AdamW, lr 3e-5, batch 32), with batch size set
by T4 memory and epoch count by val loss. **No learning-rate or epoch sweep was conducted** — the
notebook's effort went into constructing labels for the one component that had no dataset at all (M3
§7.0.3). This is the largest gap in this section, and is M5 work (§12.3).

---

## 7. Optimization Methods

| Technique | Where used | Why |
|---|---|---|
| **AdamW** | Intent/Entity (lr 3e-5, wd 0.01); vision per M3 App. B | Standard for transformer fine-tuning; decoupled weight decay |
| **Adam** | Yield prototype (lr 1e-3) | Default; the run is not usable regardless (§10.4) |
| **Cosine LR decay** | All three vision phases | Drives LR to zero at schedule end — visible in the D3 log, where epochs 15–17 produce identical val metrics to 4 dp |
| **Warmup** | D2 (2 epochs), D3 (1 epoch) | Prevents the newly-unfrozen blocks from taking a large first step and disturbing the warm start |
| **Layer-wise LR decay (0.70)** | D3 | Block 11 ≈ 2.1e-4 (continuous with D2, so the warm start is undisturbed); patch embedding ≈ 2.9e-6, effectively frozen — protecting generic edge/texture detectors learned from a million ImageNet images |
| **Gradient clipping (1.0)** | D2 | Guards the first unfreeze |
| **Early stopping** | Intent (val loss plateau at epoch 3); vision D1 (val plateaued epoch 5) | — |
| **4-bit NF4 quantization** | Generation | The only way `gemma-3-4b-it` fits a T4 alongside the index |
| **Mixed precision** | **Not used / not recorded** in any training run | Stated rather than claimed. The vision run was CPU-bound on the data path (§6.2), so AMP would not have been the binding speed-up |

The most consequential optimization result in M4 is negative: further retrieval optimization is now
known to be wasted effort, since generation is 99.5% of p50 (§9.4).

---

## 8. Regularization Techniques

| Technique | Where | Justification |
|---|---|---|
| **Label smoothing** | Vision, all phases | Empirical loss floor ≈0.62 on 20 classes. Also made the D2 memorisation diagnosis possible: train loss 0.6515 against that floor is how we knew D2 had saturated |
| **Drop-path 0.10** | Vision D3 only | Introduced *as the counterweight to* the full unfreeze. It worked: the gap held at **0.073 with 21.7M trainable params**, versus 0.092 at D2 with 5.3M. Regularization was scaled with unfrozen capacity |
| **Data augmentation** | Vision | See §2.3 / §6.4. Selected on measured shortcut destruction, which is the unusual part — the policy is defensible because the AUCs moved, not because the transforms are conventional |
| **Random erasing (p=0.15)** | Vision | Occlusion robustness; 1.9% of pipeline cost |
| **Weight decay 0.01** | Intent/Entity (AdamW default) | — |
| **Class weighting (3.75×)** | Guardrail head | Off-domain is a minority by construction; recall-prioritized per M3 §7.9. Note this did not save the head (§10.2) |
| **Class balancing** | **Rejected for vision** | §6.1 grid: `sqrt` sampling was neutral-to-negative on macro-F1 in all three pairings |
| **Cross-validation** | **Not used anywhere** | Vision uses a single scene-grouped split with cluster-bootstrap CIs instead, which is the appropriate uncertainty estimate when eval images are not independent. K-fold on a 30K intent sample remains open |

---

## 9. Training Progress

### 9.1 Vision — convergence, and what the curves showed

Per-epoch histories are in `runs/vits16_m1/logs/{phase}_history.csv` and
`{phase}_per_class_f1.csv`; curve figures accompany the ViT training report.

- **D1 converged and plateaued.** Val stopped moving at epoch 5. Train/val 0.860 / 0.813 — a 0.047
  gap, **no overfitting**.
- **D2 converged, then began memorising.** Train accuracy reached 0.9943 and train loss 0.6515
  against the ~0.62 label-smoothing floor — near the floor, i.e. the model had run out of anything
  left to fit. The generalisation gap widened 0.047 → **0.090**, and val stopped improving at epoch
  12; epoch 18 beat epoch 12 by 0.0004, which is noise selection. **This diagnosis is what determined
  the D3 configuration** (drop-path added alongside the unfreeze).
- **D3 fully converged, and did not overfit.** Epochs 15, 16 and 17 produced *identical* val metrics
  to four decimal places — cosine drove the LR to zero and predictions froze. More epochs would have
  bought nothing. The gap held at 0.073 despite 4× the trainable parameters.

### 9.2 The shortcut confirmation — the clearest evidence training behaved as designed

`rice__leaf_smut` scored **0.923** on the frozen probe with only 25 training images, while
`wheat__tan_spot` scored 0.407 with 516. That anomaly was flagged in advance as a *symptom*, not a
success. Under the exposure-randomised pipeline it collapsed to **0.320** — the 0.923 was the
brightness shortcut almost in full — then recovered to **0.769** at D2 from actual pathology.

Both halves were needed: without D1 the model ships reporting 0.92 on a class it identifies by
darkness. A second open risk closed in the good direction at the same time — leaf_smut images are
~71% replicate padding, and had padding been carrying the class it would have held near 0.92 after
brightness was neutralised. It did not.

### 9.3 Intent / Entity — plateau at 3 epochs

| Epoch | Train Loss | Intent | NER | Guardrail | Val Loss |
|---|---|---|---|---|---|
| 1 | 0.993 | 0.930 | 0.052 | 0.011 | 0.835 |
| 2 | 0.714 | 0.691 | 0.022 | 0.001 | **0.762** |
| 3 | 0.620 | 0.601 | 0.019 | 0.000 | 0.765 |

Training loss falls monotonically; **val loss bottoms at epoch 2 and ticks up at epoch 3** — the onset
of overfitting, and the reason training stopped there. The train/val gap remains small, with no
gradient instability. The per-head losses are the diagnostic: **guardrail loss reaches 0.000 by epoch
3**, i.e. the task as posed is trivially separable — the first warning that the off-domain proxy is
too easy (§10.2). Almost all remaining loss is the intent head.

### 9.4 Retrieval and generation — latency decomposition (n = 84)

| stage | p50 ms | p95 ms | share of p50 |
|---|---|---|---|
| embed | 21.2 | 34.1 | 0.2% |
| search | 19.3 | 31.9 | 0.2% |
| **generation** | **11,011.6** | 15,374.5 | **99.5%** |
| total | 11,065.1 | 15,413.6 | |

**This corrects M3 §12.3**, which named measured retrieval latency (p50 466 ms on 2-core Colab) as
the report's most significant unresolved problem. With a GPU present, embed + search together are
40.5 ms — two orders of magnitude inside budget. The budget risk was always generation. Measured p50
misses the M3 §1.2 target of 200–300 ms by ~40×, on a 4-bit T4 stack at ~7.3 tok/s rather than the
FP8/vLLM stack M3 Appendix A specifies. The decomposition says *which component to fix*; it does not
license a conclusion that the architecture is wrong.

---

## 10. Model Selection

### 10.1 Vision — `p3_full_best.pt`, epoch 16

**Selection metric: val wheat-15 macro-F1** — not 20-way accuracy or 20-way F1. A frozen backbone
already separates rice from wheat at 0.9835 F1, so any 20-way number is partly free; the honest
sub-problems are rice-5 (0.9642 on a frozen probe) and **wheat-15 (0.8017)**. Wheat-15 is the real
task.

D3 won on val and was selected. The test set — constructed contamination-free in M4 and touched by no
decision — was evaluated **once**:

| Metric | val (selection) | **test (held out)** | cluster 95% CI | delta |
|---|---|---|---|---|
| **wheat-15 macro-F1** | 0.8793 | **0.8631** | [0.8403, 0.8853] | **−0.016** |
| 20-way macro-F1 | 0.9042 | **0.8671** | [0.8396, 0.8905] | −0.037 |
| rice-5 macro-F1 | 0.9828 | 0.8823 | [0.7915, 0.9523] | −0.100 |
| accuracy | 0.9143 | **0.8998** | — | −0.014 |

**The primary metric generalised** — wheat-15 fell 0.016, well inside the ±0.023 interval, so the
val-driven selection did not overfit the val split. The rice-5 drop of 0.100 is one class, not a
trend: four of five rice classes score ≥0.97 on test, and the entire deficit is `rice__leaf_smut` at
0.4615 on **8 test images**.

The defensible headline claim is the **cumulative** one: **+0.083 wheat-15 over the frozen baseline,
3.3× the noise floor.** The D2→D3 step alone is not significant and is not claimed.

### 10.2 Intent / Entity / Guardrail — selected with one head explicitly not accepted

| Head | Result | Verdict |
|---|---|---|
| Intent | Accuracy 77.7%, **macro-F1 59.9%**, weighted-F1 76.6% | MVP-ready. Macro-F1 is the honest headline — accuracy masks near-zero F1 on rare classes (`Vegetative Propagation` 0.00 on 24 support; `Field Preparation` 0.14 on 40) |
| NER | Entity F1 93.8% (P 89.7 / R 98.0), coverage 82.9% | MVP-ready, on **labeled spans only**; unlabeled mentions are not evaluated |
| Guardrail | 1.00 / 1.00 / 1.00 on the test split | **Not accepted** |

**The guardrail head has a perfect test score and does not work.** Both "what is the capital of
France" and "stock market crashed" are classified in-domain. AG News teaches the model *news style*,
not *non-agricultural semantics*, and its headlines are structurally unlike user queries. The perfect
score is an artifact of a toy negative class — the same failure mode as the M3 §9.10 health check,
which passed comfortably while testing an easier question than the real one. This confirms M3
§7.0.3's warning that guardrail labels must be authored, not synthesized. Until they are, **the
system should not claim any safety capability from this head.**

**Checkpoint selection was by schedule end, not by best val loss** — the epoch-2 checkpoint had lower
val loss (0.762 vs 0.765). The difference is small, but not saving best-on-val is a process gap to
close in M5.

### 10.3 Generation — configuration, not checkpoint

No checkpoint was trained. The selected serving configuration is the baseline run recorded under
config hash **`d9f0e9118d6e`**, with one pending change: `max_new_tokens` 100 → 150 (§6.6).

Health check, 84 generations: **7 pass / 1 fail as reported.** G1 abstain never invokes the generator
(0/4 generated below threshold); G3 Hindi question → Hindi answer (0/21 mismatched); G4 fallback tier
carries the KVK line (35 answers); G5 answered queries cite sources (77/84). The single failure, G2
"every number traces to context" (12/84), was **manually reviewed and traced to a scorer defect, not
a model failure** — see Appendix E.

### 10.4 Yield — no model selected

The M4 run does not produce a usable yield model, for three separate reasons:

1. **Wrong model class.** An MLP was trained; M3 §5.3 selected GBT, on the reasoning that yield is
   tabular regression. No GBT was trained, so M3's choice was never tested.
2. **Target leakage.** `area` and `production` were both left in the feature matrix while
   `yield = production / area`. Any score from this setup is meaningless.
3. **The run did not complete.** Training was interrupted at epoch 2 of 10 (`Using device: cpu`;
   epoch 1 loss 7.39e7 → epoch 2 loss 1.30e4), and the evaluation cell produced no output. The split
   is also a random 80/20, not the chronological split M3 §7.4 specified for out-of-time evaluation.

Both yield datasets remain ready (M3 §7.0 rows 5–6), so this is a training task carried into M5, not
a data problem.

---

## 11. Challenges Encountered

| Challenge | Evidence | How it was addressed |
|---|---|---|
| **Evaluation contamination** | 15.3% of val+test shared a scene with train; the first fix *raised* the score +3.7 by concentrating clusters in eval | Scene-atomic split with reversed packing order → v3, 0% contamination and 100% effective (§2.2) |
| **Shortcut learning** | Brightness AUC 0.821 predicted `leaf_smut`; two augmentation designs failed to remove it before the third worked | Power-law tone curve; verified by AUC 0.821 → 0.552 and independently replicated at 0.549 (§2.3) |
| **CPU-bound data pipeline** | Best loader 263 img/s vs GPU ceiling 373 img/s on 4 vCPUs | Decoded uint8 cache; hue removed (27% of pipeline cost, zero measured benefit); `CropDS` switched to a contiguous numpy array so copy-on-write is not defeated |
| **GPU memory** | One-shot index embedding OOM-crashed while Qdrant built the HNSW graph; `gemma-3-4b-it` does not fit a T4 unquantized | 20,000-chunk shard streaming with checkpointed resume; 4-bit NF4 |
| **Class imbalance** | Vision 43:1 (`leaf_smut` 25 train images); intent 46.1% Plant Protection vs 0.7% Vegetative Propagation | Vision: macro-F1 selection, per-class reporting with n attached; balancing tested and rejected (§6.1). Intent: stratified sampling — insufficient, rare-class F1 still ≈0 |
| **Dataset quality** | 21.8% of the vision corpus is replicate padding (71.3% for `leaf_smut`); NER labels are distant supervision; guardrail negatives are a proxy | Measured and documented; padding not destroyed (open); guardrail head not accepted (§10.2) |
| **Hyperparameter sensitivity** | Normalisation worth 1.9 F1; unfreeze depth worth 9.8; temperature 0.0–0.7 worth nothing; `max_new_tokens` worth 0.25 citation rate | Swept where it mattered; §6.7 records where no sweep was run |
| **Measurement instruments failing before the system did** | Three of the run's scorers are defective (Appendix E), plus M3's own health check | Each recorded rather than deleted; a check known to be wrong is worse than no check if someone later reads it as signal |

**The recurring theme drives the M5 plan.** M3's three notable failures — MuRIL's collapsed embedding
space, the crop filter matching zero rows, thresholds carried across incompatible models — were all
*silent*. In M4 the pattern repeated one level up: the failures were in the **instruments**. The G2
grounding failure was reported as a confirmed model hallucination in the first draft of the RAG
write-up on the strength of the scorer's output alone, and was corrected only because someone opened
the retrieved chunks and looked.

---

## 12. Summary and Next Steps

### 12.1 Best-performing configurations

| Component | Final configuration | Headline result |
|---|---|---|
| **Vision** | ViT-S/16, native norm, bs 64, `mid_tf`, 3-phase progressive unfreeze, D3 epoch 16 | **test wheat-15 macro-F1 0.8631**, 20-way 0.8671, accuracy 0.8998 (n=1,287) |
| **Intent** | DistilBERT-multilingual, AdamW 3e-5, bs 32, 3 epochs | accuracy 77.7%, **macro-F1 59.9%** |
| **NER** | same backbone, distant supervision | entity F1 93.8% on labeled spans, coverage 82.9% |
| **Guardrail** | same backbone, class weight 3.75 | 1.00 on the proxy task — **not accepted** |
| **Retrieval** | bge-m3 frozen, Qdrant HNSW server mode, top_k 10, manifest fusion | route accuracy 1.000, abstain 0.958, domain margin **+0.070**, p50 40.5 ms |
| **Generation** | `gemma-3-4b-it` 4-bit zero-shot, t=0.3, `max_new_tokens` → 150 | citation 77/84, abstain 4/4 correct, p50 11.1 s |
| **Yield** | — | none (§10.4) |

### 12.2 Key observations

1. **Evaluation-set construction mattered more than any modelling choice.** Two distinct mechanisms
   would each have inflated the vision score, and neither was visible without explicit measurement.
2. **Progressive unfreezing is where the gain lives, and it saturates.** +0.098 wheat-15 from
   adapting three blocks; +0.013 (inside the CI) from unfreezing the remaining 16M parameters.
3. **Class separability, not data volume, is the binding constraint.** `rice__leaf_smut` reaches 0.92
   on 25 training images; `wheat__tan_spot` reaches 0.61 on 516.
4. **Regularization must scale with unfrozen capacity** — D3 held a *smaller* gap than D2 with 4× the
   parameters, purely because drop-path was added with the unfreeze.
5. **Generation is the entire latency budget** at 99.5% of p50, so retrieval optimization is now
   known to be wasted effort.
6. **A perfect score is a reason to check the test, not to stop checking.** Both the guardrail head
   and M3's original health check scored perfectly on a task easier than the real one.

### 12.3 Readiness for Milestone 5

**Ready for evaluation and error analysis now:**

- **Vision** — a selected checkpoint, a contamination-free held-out test set already scored once,
  per-class results with confidence intervals and n attached, and a characterised error structure
  (the `{tan_spot, leaf_blight, common_root_rot, mite, mildew}` necrotic-lesion cluster, plus a
  cross-crop `leaf_smut → tan_spot` error worth a dedicated probe).
- **Intent / NER** — per-class F1 with support counts, and documented failure modes (Hindi entity
  mentions missed by distant supervision; false positives such as "Kis" in "PM Kisan").
- **Retrieval** — 11-check evaluation, thresholds calibrated from the measured score distribution, and
  a stable 0.070 abstain margin across all seven swept configs.
- **Generation** — 90 recorded generations with scores and citations under one config hash, ready for
  re-scoring once the instruments are fixed.

**Must be completed early in M5, in priority order:**

1. **Fix `numeric_grounding` and re-run — safety-critical, do first.** With distillation dropped
   (§3.4), this is now the *primary* defence against the M3 §14 dosage-invention risk, not a
   secondary one. Until it lands, that check is not running.
2. **Author real guardrail data** (dosage bounds, banned terms, ambiguous off-domain) and retrain the
   head. Without it, no safety claim can be made.
3. **Train the yield model properly** — GBT as selected, leakage columns removed, chronological split
   restored (§10.4).
4. **Raise the generation token cap to 150** and re-measure (§6.6).
5. **Rewrite the other two broken metrics** — the bilingual retrieval check and `lang` (Appendix E).
6. **Re-measure latency on target-class hardware** with vLLM + FP8; update M3 §12.3.
7. **Run a shortcut re-probe on the trained vision model.** The B-series audits proved the *pipeline*
   destroys the cues; no probe yet confirms the *final model* ignores them.
8. **Calibrate the vision classifier** (temperature scaling + abstention threshold), which the RAG
   router needs to decide when to ask a clarifying question instead of asserting a diagnosis.
9. Sweep intent hyperparameters and train on more than 30K rows (§6.7).
10. Enable hybrid dense+sparse retrieval against the Hinglish gap — the corpus is 96% code-mixed, so
    this is not an edge case.

---

## Appendix A — Vision Phase Configurations

| Setting | D1 | D2 | D3 |
|---|---|---|---|
| Trainable params | 7.7K (0.04%) | 5.33M (25%) | 21.67M (100%) |
| Unfrozen | head only | blocks 9–11 + head | all |
| LR | 1e-3 | 1e-4 backbone / 1e-3 head | LLRD 0.70, base 3e-4 |
| Warmup | — | 2 epochs | 1 epoch |
| Schedule | cosine | cosine | cosine |
| Grad clip | — | 1.0 | — |
| Drop-path | — | — | 0.10 |
| Epochs | 12 | 20 | 18 |
| Wall time | 9 min | 15 min | 15 min |

Constant across phases: batch 64 (160 steps/epoch), 4 workers, label-smoothed cross-entropy,
`mid_tf` train transform, deterministic `CenterCrop(224)` eval transform, split `clean_index_v3.csv`,
selection on val wheat-15 macro-F1.

## Appendix B — Training Logs

**Vision, val macro-F1 by phase:**

| Metric | frozen probe | D1 | D2 | D3 |
|---|---|---|---|---|
| 20-way | 0.8382 | 0.7655 | 0.8830 | 0.9042 |
| wheat-15 | 0.7966 | 0.7682 | 0.8660 | 0.8793 |
| rice-5 | 0.9717 | 0.7887 | 0.9429 | 0.9828 |
| accuracy | 0.8559 | 0.8131 | 0.9042 | 0.9143 |
| train/val gap | — | 0.047 | 0.092 | 0.073 |

Diagnostic classes across phases: `wheat__tan_spot` 0.407 → 0.306 → 0.569; `rice__leaf_smut`
0.923 → 0.320 → 0.769 (§9.2).

**Intent/Entity, per-epoch (total ~589 s):**

| Epoch | Time (s) | Train | Intent | NER | Guardrail | Val |
|---|---|---|---|---|---|---|
| 1 | 194 | 0.993 | 0.930 | 0.052 | 0.011 | 0.835 |
| 2 | 197 | 0.714 | 0.691 | 0.022 | 0.001 | 0.762 |
| 3 | 198 | 0.620 | 0.601 | 0.019 | 0.000 | 0.765 |

**Yield prototype (incomplete, not usable — §10.4):** epoch 1 loss 73,900,670.0 → epoch 2 loss
12,959.3; interrupted at 2/10; no evaluation output.

## Appendix C — Vision Per-Class Test Results

| Class | F1 | n | | Class | F1 | n |
|---|---|---|---|---|---|---|
| `rice__leaf_smut` | **0.4615** | 8 | | `wheat__aphid` | 0.9157 | 86 |
| `wheat__leaf_blight` | **0.5849** | 57 | | `wheat__mildew` | 0.9432 | 112 |
| `wheat__tan_spot` | **0.6129** | 65 | | `wheat__fusarium_head_blight` | 0.9474 | 64 |
| `wheat__common_root_rot` | 0.8000 | 57 | | `wheat__brown_rust` | 0.9496 | 118 |
| `wheat__smut` | 0.8125 | 50 | | `wheat__healthy` | 0.9505 | 103 |
| `wheat__mite` | 0.8466 | 76 | | `rice__brown_spot` | 0.9697 | 65 |
| `wheat__stem_fly` | 0.8485 | 17 | | `wheat__septoria` | 0.9722 | 35 |
| `wheat__black_rust` | 0.8615 | 31 | | `wheat__yellow_rust` | 0.9890 | 135 |
| `wheat__blast` | 0.8966 | 58 | | `rice__blast` | 0.9895 | 48 |
| | | | | `rice__bacterial_blight` | 0.9908 | 55 |
| | | | | `rice__tungro` | **1.0000** | 47 |

16 of 20 classes exceed 0.80; 11 exceed 0.94. The three below 0.65 are the same three the
frozen-probe diagnostics predicted before any fine-tuning. `rice__leaf_smut` should always be quoted
with n=8 attached — its F1 swung 0.833 → 0.471 → 0.769 → 0.923 across consecutive D3 epochs.

**Top test confusions** — concentrated, not scattered: `leaf_blight → mildew` (7, 12.3%),
`leaf_blight → tan_spot` (7, 12.3%), `tan_spot → common_root_rot` (7, 10.8%), `tan_spot → leaf_blight`
(6, 9.2%), `tan_spot → mite` (6, 9.2%), `leaf_smut → tan_spot` (3, **37.5%** — a cross-crop error in a
model that separates rice from wheat at 0.98 elsewhere; worth an explicit probe in M5).

## Appendix D — Artifacts

| Artifact | Path / size | Notes |
|---|---|---|
| Selected vision checkpoint | `runs/vits16_m1/ckpt/p3_full_best.pt` (epoch 16) | Phase checkpoints `p1_head_best.pt`, `p2_blocks9_11_best.pt` retained |
| Vision data path | `pipeline.py` | Frozen module — training imports exactly the audited transform object |
| Vision split | `clean_index_v3.csv` (12,823 rows) | + `split_plan_v3.csv`, `exclusions.csv` |
| Vision logs | `runs/vits16_m1/logs/{phase}_history.csv`, `{phase}_per_class_f1.csv` | Per-epoch and per-class |
| Vision test results | `test_results.json` | Single held-out evaluation |
| Augmentation config + audit tables | `aug_config.json` | All shortcut AUCs |
| Intent/entity model | `final/intent_entity_guardrail_model.pt` (~540 MB) | + `intent_entity_label_maps.json`, `intent_entity_training_summary.json` |
| Retrieval index | `agri_knowledge-*.snapshot` (3.80 GB) + `manifest.json` (~1 KB) | **Matched pair — the snapshot alone is unsafe.** Without the manifest the serving path must guess prefix policy, vector dim and thresholds, and every wrong guess fails silently |
| Embedding cache | ~37 `.npy` shards, ~1.5 GB | Rebuild insurance; not needed to serve |
| Generation run | `rag_generation_m4/`, config hash **`d9f0e9118d6e`** | `generations.jsonl` (90 rows), `run_manifest.json`, `retrieval_sweep.json`, `generation_sweep.json`, `health_check.json`, `latency.json` |

Notebooks: `08c_rag_vector_db_bge_m3.ipynb` (build, run once) · `09_rag_retrieval.ipynb` (query) ·
`10_rag_generation.ipynb` (generation + sweeps) · `11_kcc_intent_entity_guardrail.ipynb` ·
`yield_training.ipynb` (prototype, §10.4) · `13a_distill_pipeline_kaggle.ipynb` (built, not run).

Quote `d9f0e9118d6e` next to every generation number.

## Appendix E — Known-Broken Measurement Instruments

Three scorers used in this milestone are defective. All are recorded rather than deleted, because a
check known to be wrong is worse than no check if someone later reads it as signal.

| Instrument | Defect | Consequence | Fix |
|---|---|---|---|
| `numeric_grounding` → health check G2, sweep `ground` | `NUM_RE = \d+(?:\.\d+)?` has no notion of a thousands separator. Context `Rs 6,000` tokenizes to `['6','000']`; answer `6000` to `['6000']`; the membership test fails on a figure that is verbatim present. It also counts list markers (`1, 2, 3`) as quantities | The reported 12/84 is **not a hallucination rate and must not be quoted as one.** Retrieved chunks were inspected by hand and the figures are present. Correct statement: *no confirmed hallucination was found in this run.* This does **not** clear the M3 §14 dosage risk — it leaves it **unmeasured** | Normalise separators on both sides; ignore small integers adjacent to list punctuation; re-run the sweep |
| Bilingual retrieval check (`08c` §5, B1–B2) | Compares whether English and Hindi phrasings return the same record *IDs*. Across 112,269 rice chunks, two entirely correct answers score jaccard 0.00 | Two WARNs that are a broken check, not a broken system. Output disregarded | Compare retrieved *topics*, not record identity |
| `lang` in the generation sweep | Reads 0.33 while the purpose-built G3 check reads 21/21 pass. It requires the answer's script to match the question's, computed over a ¾ non-English subset against an overwhelmingly code-mixed corpus | Penalises substantively correct answers. G3 is the one built to test the actual rule | Align with G3's definition |

All three failed *toward alarm* rather than silence, which is the survivable direction — but none was
validated before use.

---

## Team Review & Sign-Off

Reviewers should read §6.3 (unfreeze depth), §9.4 (latency decomposition), §10.2 (guardrail not
accepted), §10.4 (yield not delivered), §3.4 (distillation dropped) and Appendix E specifically —
these are the sections that change a Milestone 3 position or record a component as not delivered.

| # | Team Member | Approved | Date |
|:-:|-------------|:--------:|:----:|
| 1 | Mahesh | Yes | 30 Jul 2026 |
| 2 | Harliv | Yes | 30 Jul 2026 |
| 3 | Lokesh | Yes | 30 Jul 2026 |
| 4 | Aneeqa | Yes | 30 Jul 2026 |
| 5 | Tanmay | Yes | 30 Jul 2026 |

**Document version:** Milestone 4 — v1 · **Prepared:** 30 July 2026
