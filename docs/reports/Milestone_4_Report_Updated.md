# FarmerVision — Milestone 4 Report
## Model Training and Experiments

**Prepared:** 10 August 2026
**Team 7 — FarmerVision / AgriAssist**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Training Data](#2-training-data)
3. [Model Configuration](#3-model-configuration)
4. [Training Environment](#4-training-environment)
5. [Training Methodology](#5-training-methodology)
6. [Model Training Experiments](#6-model-training-experiments)
7. [System Configuration Experiments (Not Model Training)](#7-system-configuration-experiments-not-model-training)
8. [Regularization Techniques](#8-regularization-techniques)
9. [Engineering / System Optimizations (Not Model Optimization)](#9-engineering--system-optimizations-not-model-optimization)
10. [Validation Results and Model Selection](#10-validation-results-and-model-selection)
11. [Milestone 3 (Planned) vs Milestone 4 (Final) Configuration](#11-milestone-3-planned-vs-milestone-4-final-configuration)
12. [Hyperparameter Experiment Summary — All Components](#12-hyperparameter-experiment-summary--all-components)
13. [Checkpoint and Version Reference](#13-checkpoint-and-version-reference)
14. [Known Limitations Carried Into Milestone 5](#14-known-limitations-carried-into-milestone-5)

**Appendices:** [A — Vision augmentation pipeline](#appendix-a--vision-augmentation-pipeline-full-detail) · [B — Vision phase configs](#appendix-b--vision-phase-configurations-full-detail) · [C — Training logs](#appendix-c--training-logs-full-detail) · [D — Test-set results (preview)](#appendix-d--test-set-results-preview-only--not-used-for-m4-selection) · [E — Broken measurement instruments](#appendix-e--known-broken-measurement-instruments) · [F — Artifacts](#appendix-f--artifacts) · [G — Distillation hyperparameter leaderboard](#appendix-g--distillation-hyperparameter-study-full-leaderboard)

---

## 1. Introduction

### 1.1 Recap of Milestone 3

Milestone 3 picked one model per module: ViT-Small for disease classification, BAAI/bge-m3 (frozen) for retrieval, a DistilBERT-based three-head model for intent/entity/guardrail detection, a Gradient-Boosted Tree (GBT) for yield prediction, a distilled 2–4B language model for writing answers, and a rule-based estimator for profitability. Voice input/output was dropped from the project scope at M3. Of all these, only retrieval already existed as working code at M3 — everything else was still just a design.

### 1.2 What this report covers

Milestone 4 is about training models and running training experiments. Detailed test-set scoring, error analysis, and "is this ready for the next stage" checks are Milestone 5 topics. This report keeps to validation-based results and moves anything test-set or evaluation-related to the appendices, so it stays a training report.

### 1.3 What was, and was not, trained

| Component | M4 status |
|---|---|
| Vision (ViT-S/16) | Trained — 3 phases. Validation used to select the final phase; test set checked once (Appendix D). |
| Intent / Entity / Guardrail (DistilBERT) | Trained. Intent and NER are usable; the guardrail head is combined with rule-based checks for full safety coverage (§10.2). |
| Retrieval (bge-m3 + Qdrant) | No training, by design — the embedder is frozen (M3 decision). Index was built and its settings were tuned (§7.1). |
| Synthesis LLM (Generation) | Trained — a QLoRA adapter distilled from teacher-written answers. Checkpoint chosen by validation loss (§3.4, §10.4). |
| Yield | Trained — three GBM families compared against a PyTorch baseline. LightGBM (Tuned) selected (§3.5, §10.5). |

*Four components were trained and validated. One (Retrieval) was configured, not trained, by design.*

---

## 2. Training Data

### 2.1 Data splits used

| Model | Dataset | Split | Sizes |
|---|---|---|---|
| Vision | Merged 20-class rice+wheat corpus | Scene-grouped, rebuilt in M4 (`clean_index_v3.csv`) | train 10,252 / val 1,284 / test 1,287 |
| Intent/Entity/Guardrail | KCC corpus (30,000-query stratified sample) + 5,172 generated non-agri examples | 80/10/10, stratified by guardrail label | train 28,772 / val 3,597 / test 3,597 |
| Retrieval | RAG PDF corpus + KCC UP 2020–2025 | No split — frozen embedder, no training | 723,439 chunks |
| Generation (distillation) | Teacher-written answers over retrieved context (gemini-2.5-flash-lite) | 90/10, stratified by row kind; validation capped at 200 rows | train 2,868 / val 200 |
| Yield | `production_unified_imputed.csv` (cleaned: 426,803 of 440,962 rows) | Two-stage random split, random_state=42 (not chronological — see §14) | train 308,364 / val 54,418 / test 64,021 |

### 2.2 Preprocessing and augmentation (summary)

Vision training used a standard resize/crop/flip/colour/blur augmentation pipeline. The full transform list is in Appendix A. The pipeline was chosen by testing, not convention: two "shortcut" signals were found in the raw data — image brightness alone could identify one disease class (leaf_smut, AUC 0.821), and blur alone could separate rice from wheat (AUC 0.833). A simple linear brightness jitter cannot remove a shortcut like this, because it only spreads out each class's brightness range without re-centring it. A power-law brightness transform was needed instead, and it worked: brightness AUC dropped from 0.821 to 0.552, and the blur-based signal dropped from 0.833 to 0.595 — both close to the 0.5 "no signal" level.

For Intent/Entity: sampling was stratified by query type from a 710,616-query KCC pool. Entity (NER) tags were generated by rule-based crop-alias matching, covering 86.4% of rows. 5,172 generated non-agri examples were added as negative (off-topic) training data for the guardrail head.

---

## 3. Model Configuration

### 3.1 Vision

Backbone: `vit_small_patch16_224.augreg_in21k_ft_in1k` (timm 1.0.26), pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. 21.67M parameters. Classification head: 384 → 20 classes. Normalisation uses the checkpoint's own pretraining statistics (0.5, 0.5, 0.5) rather than standard ImageNet statistics — this choice was tested, not assumed (§6.1).

### 3.2 Intent / Entity / Guardrail

Backbone: `distilbert-base-multilingual-cased`, pretrained, with three linear heads sharing one backbone: Intent (768→7 classes), NER (768→3 tags: O, B-CROP, I-CROP), Guardrail (768→2, safe vs unsafe). Max sequence length 64 tokens. Total 134.7M parameters. One forward pass produces all three outputs.

### 3.3 Retrieval

Embedding model: bge-m3, frozen, 1024 dimensions, cosine similarity, 512-token chunk cap. Index: Qdrant with an HNSW graph (m=16, ef_construct=128) in server mode. Two thresholds were set from the measured score distribution — below 0.56 triggers a fallback answer, above 0.66 is treated as a confident/grounded match. Both are stored in `manifest.json` next to the index.

### 3.4 Synthesis LLM — distilled model

The Milestone 3 plan was to fine-tune a small (2–4B) model by knowledge distillation from a larger teacher model, so it would learn to say "the figure is not available" instead of guessing at numbers (this was meant to reduce the dosage-invention risk flagged in M3 §14). That plan was carried out.

Base model: `unsloth/gemma-3-4b-it-bnb-4bit`, a pre-quantised 4-bit (NF4) copy of `google/gemma-3-4b-it`. A LoRA adapter was trained on top of it with QLoRA (rank 32, alpha 64, dropout 0.05, applied to all seven attention/MLP projection modules). Only the adapter is saved (262 MB) — the base model itself stays frozen and 4-bit.

Training targets came from a teacher model that wrote answers over the same retrieved context the student would see (knowledge distillation) — not from real KCC expert answers. Expert answers were written without seeing any retrieved context, so training on them would have taught the model to ignore its sources and to state numbers not present in the source text.

**Config note:** the data-prep notebook's configuration variable names the teacher as a 12B Gemma model, and that string was carried into the training manifest. The generation code that actually produced the answers, however, called `gemini-2.5-flash-lite` through an API, and every training row records this correctly. The manifest field is describing a setting that was never used — the real teacher was Gemini 2.5 Flash Lite, not a local 12B model — and this report describes it as such rather than repeating the manifest's label.

873 refusal examples (a real question paired with someone else's context, and an off-topic question with no context at all) were added to the training set — 26% of all rows — so the model practises refusing instead of relying only on the retrieval score gate. The final training set is 3,187 rows, kept from 3,368 generated after dropping answers with unsupported numbers, missing citations, or other rule violations (§6.3 covers how the training settings for this run were chosen; §10.4 covers the training result and checkpoint selection).

### 3.5 Yield — selected model: LightGBM (Tuned)

Milestone 3 selected a GBT (gradient-boosted tree) model for yield prediction, since yield is a tabular regression problem. Three GBM families were trained and compared: XGBoost, LightGBM, and CatBoost, each at reasonable default settings and then tuned with RandomizedSearchCV (3-fold cross-validation on the training split, scored on negative MSE). A PyTorch MLP was trained alongside them as a baseline, not as a deployment candidate.

Features: 6 categorical (crop, state, district, season, data_source, crop_type) and 5 numerical (year, area, annual_rainfall, fertilizer, pesticide). Target: yield. `production` is not used as an input feature, so there is no leakage between an input and the target's own definition (yield = production / area).

Final selected model: **LightGBM (Tuned)** — colsample_bytree 0.75, learning_rate 0.254, max_depth 12, min_child_samples 30, n_estimators 202, num_leaves 94, reg_alpha 0.0020, reg_lambda 0.0005, subsample 0.657. It belongs to the same GBT family Milestone 3 selected; the choice of LightGBM over XGBoost and CatBoost is justified on held-out test performance in §10.5.

---

## 4. Training Environment

| Component | Hardware | Software | Wall time |
|---|---|---|---|
| Vision | Kaggle, 2× Tesla T4, 4 vCPU | PyTorch, timm 1.0.26 | ~42 min total (3 phases + eval) |
| Intent/Entity | Colab T4 | PyTorch, HF transformers, scikit-learn | ~949 s (~15.8 min, 5 epochs) |
| Index build | Colab GPU | sentence-transformers, Qdrant | ~4 h (mostly GPU encoding) |
| Generation (HPO study) | Kaggle, 1× T4 | transformers, peft, bitsandbytes | ~6.1 h (27 configs) |
| Generation (final training) | Kaggle, 1× T4 (forced single-GPU) | transformers, peft, bitsandbytes, accelerate | ~9.6 h (4 epochs, best at epoch 2) |
| Generation (serving sweep) | Colab T4 | transformers + bitsandbytes 4-bit | ~10 min for 84 generations |
| Yield | Kaggle, CPU only | pandas, scikit-learn, xgboost, lightgbm, catboost, torch | ~4 min (CatBoost default); tuning 60/60/45 fits, ~14–70s per fit |

The 4-vCPU limit made the vision data-loading pipeline CPU-bound rather than GPU-bound (loader speed, not the GPU, was the bottleneck — see §9). The T4 memory limit is also why generation serves in 4-bit. Kaggle's second GPU had to be hidden from the distillation training run (§9) — a 4-bit model cannot be split across two cards the normal way. Yield training used CPU only by choice — gradient boosting at this data scale does not need a GPU.

---

## 5. Training Methodology

### 5.1 Vision — progressive unfreezing

Training changed exactly one structural thing per phase, so any gain could be traced to that one change.

| Phase | Trainable params | Key config | Epochs | Time |
|---|---|---|---|---|
| D1 — head only | 7.7K (0.04%) | frozen backbone, lr 1e-3, cosine schedule | 12 | 9 min |
| D2 — blocks 9–11 | 5.33M (25%) | lr 1e-4 backbone / 1e-3 head, 2-epoch warmup, grad-clip 1.0 | 20 | 15 min |
| D3 — full model | 21.67M (100%) | layer-wise LR decay 0.70 (base 3e-4), drop-path 0.10, 1-epoch warmup | 18 | 15 min |

Constant across all phases: batch size 64, 4 data-loader workers, label-smoothed cross-entropy, cosine learning-rate decay. Selection metric: validation wheat-15 macro-F1 (§10.1).

### 5.2 Intent / Entity / Guardrail — joint multi-task loss

A single training loop with one backward pass over the summed loss from all three heads:

| Head | Loss | Weight | Notes |
|---|---|---|---|
| Intent | Cross-entropy, ignore_index=-100 | 1.0 | — |
| NER | Token-level cross-entropy, ignore_index=-100 | 1.0 | Special/pad tokens masked out |
| Guardrail | Weighted cross-entropy (class-balanced) | 1.0 | Combined with rule-based checks at inference (§10.2) |

Optimiser: AdamW, learning rate 3e-5, batch size 32, weight decay 0.01, gradient clipping 1.0, linear schedule with 10% warmup, 5 epochs, max length 64.

### 5.3 Retrieval — no training

Retrieval has no gradients. The index was built once, and its serving configuration was chosen by sweeping settings, not by training. These configuration experiments are reported separately in §7, to keep them clearly apart from real model-training experiments.

### 5.4 Generation — QLoRA distillation

Standard causal-LM cross-entropy, with two rules that change what the model actually learns: only the answer tokens count towards the loss (the prompt is masked with `-100`, so the model is not trained to echo its input), and the answer is never truncated — if a row is too long for the context window, tokens are cut from the front of the prompt instead.

Batch size was not hand-picked: the notebook computes it from free GPU memory at run time. At a 1,024-token window, one row's loss needs about 2.7 GB, which left room for 1 row at a time with 16 steps of gradient accumulation — an effective batch of 16, 179 optimiser steps per epoch. Training was allowed up to 4 epochs, with early stopping (patience 2) on validation loss.

### 5.5 Yield — gradient boosting with hyperparameter search

The raw dataset (440,962 rows) was cleaned before training: exact duplicates and physically impossible rows (area ≤ 0, negative production) were dropped, rows where the stored yield disagreed with production/area were flagged, and yield outliers above the 99th percentile within each crop type were removed, leaving 426,803 rows.

Three gradient-boosted tree families (XGBoost, LightGBM, CatBoost) were trained at default settings, then each was tuned with RandomizedSearchCV: 20 candidates × 3-fold CV for XGBoost and LightGBM, 15 candidates × 3-fold CV for CatBoost, all scored on negative MSE. A PyTorch MLP was trained in parallel as a baseline, using one-hot encoded categorical features (923 columns) and 10 epochs of Adam.

---

## 6. Model Training Experiments

> This section covers experiments on components that were actually trained (gradients updated): Vision, Intent/Entity/Guardrail, and Generation (distillation). Retrieval's configuration sweep is not a training experiment — it is covered separately in §7.

### 6.1 Vision experiments

**Normalisation and class balancing** (linear probe, frozen backbone, 60 epochs)

| Normalisation | Class balancing | macro-F1 | accuracy | min class F1 |
|---|---|---|---|---|
| native (0.5,0.5,0.5) — selected | none — selected | 0.8461 | 0.8564 | 0.507 |
| native | sqrt | 0.8453 | 0.8587 | 0.521 |
| imagenet | none | 0.8326 | 0.8470 | 0.496 |
| dataset stats | none | 0.8274 | 0.8470 | 0.478 |
| imagenet | sqrt | 0.8236 | 0.8423 | 0.460 |
| dataset stats | sqrt | 0.8216 | 0.8423 | 0.429 |

Selected: the checkpoint's own (native) normalisation, no class balancing. Matching the pretraining statistics beat matching our own dataset statistics by 1.9 F1 points. sqrt-sampling never helped macro-F1 in any pairing.

**Batch size**

| Batch size | img/s | s/epoch | peak memory |
|---|---|---|---|
| 32 | 341 | 30.0 | 1.6 GB |
| 64 — selected | 360 | 28.5 | 2.7 GB |
| 96 | 373 | 27.4 | 3.8 GB |
| 128 | 368 | 27.9 | 4.9 GB |

Selected: 64. Throughput levels off by 96, and memory is never a limit (under 5 GB of 15.6 GB available). 64 stays within 3% of peak speed while giving more optimiser steps per epoch (160 vs 106 at batch 96), which matters more than a small speed gain on a 10K-image fine-tune.

**Unfreezing depth — the experiment that mattered most**

| Metric | frozen probe | D1 (head) | D2 (blocks 9–11) | D3 (full) |
|---|---|---|---|---|
| wheat-15 macro-F1 (val) | 0.7966 | 0.7682 | 0.8660 | 0.8793 |
| 20-way macro-F1 (val) | 0.8382 | 0.7655 | 0.8830 | 0.9042 |
| accuracy | 0.8559 | 0.8131 | 0.9042 | 0.9143 |
| train/val gap | — | 0.047 | 0.092 | 0.073 |

A cluster-bootstrap 95% confidence interval on wheat-15 macro-F1 is ±0.025 — any difference smaller than that is not a real result.

- D1 scoring below the frozen probe is expected, not a regression: D1 is the first run with shortcut-destroying augmentation active, and a frozen backbone cannot yet learn to ignore it.
- D1 → D2 gained +0.098 — nearly 4× the noise floor. Most of the benefit comes from adapting the top three blocks.
- D2 → D3 gained only +0.013, inside the confidence interval. Unfreezing the remaining 16M parameters bought no measurable improvement on its own — reported as comparable to D2, not better.

**Augmentation pipeline: cost vs. benefit**

| Variant | ms/img | s/epoch | brightness AUC | blur AUC |
|---|---|---|---|---|
| full_tf (with hue jitter) | 11.51 | 54.5 | 0.549 | 0.584 |
| mid_tf (hue removed) — selected | 8.41 | 42.1 | 0.560 | 0.605 |
| fast_tf (bilinear, coarser) | 7.63 | 39.0 | 0.558 | 0.581 |

A pre-registered check before adopting any speed-up: does trimming the pipeline cost any of the shortcut-removal effect? Removing a small hue jitter cut 27% of processing time and moved neither AUC outside the ±0.02 noise floor, so it was removed. fast_tf was rejected despite being faster, because its bilinear resizing could blur fine lesion texture that the AUC check does not measure. Selected: mid_tf.

### 6.2 Intent / Entity / Guardrail experiments

A sweep compared optimizer, learning rate, batch size, epoch count, and label smoothing, each scored by a composite validation score (a weighted combination of intent macro-F1 and NER entity F1):

| Configuration | Optimizer | LR | Batch | Epochs | Label smoothing | Intent macro-F1 | NER F1 | Composite |
|---|---|---|---|---|---|---|---|---|
| **adamw_lr3e-5_5ep — selected** | AdamW | 3e-5 | 32 | 5 | 0.0 | 0.785 | 0.962 | 1.747 |
| adamw_lr3e-5_bs32 | AdamW | 3e-5 | 32 | 3 | 0.0 | 0.753 | 0.960 | 1.713 |
| adamw_labelsmooth | AdamW | 3e-5 | 32 | 3 | 0.1 | 0.753 | 0.960 | 1.713 |
| adamw_lr3e-5_bs64 | AdamW | 3e-5 | 64 | 3 | 0.0 | 0.747 | 0.953 | 1.700 |
| adamw_lr1e-5_bs32 | AdamW | 1e-5 | 32 | 3 | 0.0 | 0.730 | 0.937 | 1.667 |
| sgd_lr1e-3_bs32 | SGD | 1e-3 | 32 | 3 | 0.0 | 0.477 | 0.802 | 1.279 |

Findings: AdamW clearly beats SGD (composite 1.7+ vs 1.279). Learning rate 3e-5 beats 1e-5. Batch size 32 is slightly better than 64. Label smoothing (0.1) gave no measurable benefit over 0.0. Extending training from 3 to 5 epochs gave the largest single gain in the sweep (composite 1.713 → 1.747) and is the selected configuration, matching §5.2.

### 6.3 Generation (Distilled LLM) experiments

Before the final training run, 27 QLoRA fine-tunes of the same 4B student were run on a small, fixed slice (160 training rows, 48 validation rows, identical across every config) to find which settings actually matter. Every config changed exactly one setting from a shared baseline, and all 27 were ranked by best validation loss. A noise floor was measured first by running the baseline on three seeds: best validation loss ranged 0.4197–0.4271, a band 0.0074 wide — any difference smaller than that is not a real result.

| Setting family | Configs tried | Best result | Worst result | Beyond noise? |
|---|---|---|---|---|
| Optimization (LR, scheduler, optimizer, batch) | 12 | 0.3370 | 0.7088 | Yes — 7 of 12 |
| Capacity (LoRA rank, target modules) | 6 | 0.3667 | 0.5360 | Yes — 5 of 6 |
| Regularization (dropout, weight decay) | 5 | 0.4243 | 0.4277 | No — 1 of 5, barely |
| Epochs (3-epoch run) | 1 | 0.3148 | — | Yes — best config overall |

Headline findings, each checked against the 0.0074 noise band:

- Learning rate is the sharpest lever. 2e-4 is right; 1e-4 is measurably worse and 5e-5 is much worse (the single largest degradation in the study).
- More epochs helps most. The 3-epoch run (`epochs_curve`) is the best config in the whole study.
- LoRA rank matters, and 16 is not enough. Rank 8 is measurably worse, rank 64 is measurably better, rank 32 sits inside the noise band.
- The alpha = 2×rank convention is a real choice, not a habit — setting alpha equal to rank (1× ratio) is measurably worse.
- Both attention and MLP modules are needed. Adapting only one or the other is worse than adapting all seven projection modules; attention-only is the worse of the two.
- Regularization (dropout, weight decay) does nothing measurable — the entire spread across five configs is smaller than the seed-to-seed noise.

The full 27-config leaderboard is in Appendix G.

### 6.4 Yield experiments

Tuning helped two of the three GBM families, but not the third:

| Model | RMSE (default) | RMSE (tuned) | Change | R² (tuned) |
|---|---|---|---|---|
| LightGBM | 2.5588 | 2.4112 | −5.8% (better) | 0.9572 |
| CatBoost | 2.9032 | 2.7051 | −6.8% (better) | 0.9461 |
| XGBoost | 2.4671 | 2.5447 | +3.1% (worse) | 0.9523 |

Tuning gave a clear, consistent improvement for LightGBM and CatBoost. It did not help XGBoost — its untuned defaults (n_estimators=1000, lr=0.05, depth=7) were already close to the best RMSE found, and the 20-candidate search budget, scored on negative MSE, did not line up with RMSE/R² outcomes for this model.

A second experiment tested a log1p transform on the yield target, since the distribution is heavily right-skewed (skewness 198.6). It improved MAE for LightGBM and CatBoost (better typical-case accuracy) but worsened RMSE and R² for all three models — compressing large values during training under-penalises big misses on high-yield crops like sugarcane, and those errors re-expand after inverting the transform. Decision: keep the original-scale target, since RMSE/R² are the primary selection criteria for this milestone.

---

## 7. System Configuration Experiments (Not Model Training)

> Retrieval uses a frozen embedding model — there is no training here, so its sweep (§7.1) is an indexing/configuration choice, not a training experiment. §7.2 is a separate serving-time sweep of decoding settings (temperature, output length, context size); these are inference parameters, independent of which generator weights are loaded, so they are kept apart from the training experiments in §6.

### 7.1 Retrieval configuration sweep (24 queries, no LLM involved)

| Configuration | crop_hit | route_acc | abstain | margin | p50 ms |
|---|---|---|---|---|---|
| top_k = 3 | 0.846 | 1.000 | 0.958 | 0.070 | 3337.7* |
| top_k = 5 | 0.923 | 1.000 | 0.958 | 0.070 | 45.8 |
| top_k = 10 — selected | 1.000 | 1.000 | 0.958 | 0.070 | 52.4 |
| intent weighting off | 0.923 | 0.889 | 0.958 | 0.070 | 39.0 |
| fusion flat 1.0/1.0 | 0.923 | 0.889 | 0.958 | 0.070 | 39.2 |
| fusion 3.0/0.3 | 0.923 | 1.000 | 0.958 | 0.070 | 37.3 |
| fusion (manifest, baseline) — selected | 0.923 | 1.000 | 0.958 | 0.070 | 37.8 |

*\* cold-cache artifact of being the first config run, not a real property of top_k.*

The margin (headroom for the abstain decision) stayed at 0.070 across every configuration, so no setting traded safety for ranking quality. Intent weighting matters — turning it off dropped route accuracy from 1.000 to 0.889. top_k=10 buys the remaining crop_hit gain for about 7 ms extra.

### 7.2 Generation configuration sweep (12-query subset)

| Configuration | grounded* | cited | out tokens | tok/s | gen ms |
|---|---|---|---|---|---|
| baseline (t=0.3, 100 tok, k=5) | 0.907 | 0.92 | 82.2 | 7.3 | 11396 |
| temperature = 0.0 | 0.912 | 0.92 | 81.9 | 7.4 | 11161 |
| temperature = 0.7 | 0.917 | 0.92 | 80.1 | 7.3 | 10844 |
| max_new_tokens = 60 | 0.844 | 0.75 | 57.5 | 6.8 | 8554 |
| max_new_tokens = 150 — recommended | 1.000 | 1.00 | 90.2 | 7.5 | 11984 |
| ctx_top_k = 3 | 0.908 | 1.00 | 66.0 | 7.6 | 8643 |
| ctx_top_k = 8 | 0.942 | 0.92 | 77.8 | 6.4 | 12103 |

*\* The "grounded" column comes from a scorer with a known bug (see Appendix E) and should be read as a lower bound only, not a real grounding rate.*

The clearest finding: cutting the answer short hurts citations. max_new_tokens=60 gives the worst citation rate (0.75); 150 is the only setting that reaches 1.00, because the answer's source citation line comes last and gets cut off when the token budget runs out. Raising the cap to 150 costs about 600ms and is recommended for the next milestone. Temperature made no measurable difference across 0.0–0.7.

---

## 8. Regularization Techniques

| Technique | Where used | Note |
|---|---|---|
| Label smoothing | Vision, all phases | Loss floor ~0.62 on 20 classes; used to diagnose D2 memorisation |
| Drop-path 0.10 | Vision D3 only | Added alongside the full unfreeze; train/val gap held at 0.073 vs 0.092 at D2, despite 4× more trainable parameters |
| Data augmentation | Vision | Chosen by testing shortcut destruction (§2.3, §6.1), not by convention |
| Random erasing (p=0.15) | Vision | Occlusion robustness; small pipeline cost |
| Weight decay 0.01 | Intent/Entity (AdamW default) | — |
| Class-balanced weighting | Guardrail head | Improves recall; combined with rule-based checks for full coverage (§10.2) |
| Class balancing | Rejected for Vision | sqrt sampling was neutral-to-negative on macro-F1 (§6.1) |
| Cross-validation | Not used | Vision uses a single scene-grouped split with bootstrap confidence intervals instead, which is the right choice when eval images are not independent |

---

## 9. Engineering / System Optimizations (Not Model Optimization)

> These changes make things run faster. They do not change what any model learns or how accurate it is.

| Area | Finding | Fix |
|---|---|---|
| Vision data loader | Best loader speed was 263 img/s against a GPU ceiling of 373 img/s on 4 vCPUs — the CPU was the bottleneck, not the GPU | Decoded images cached as uint8; dataset switched to a contiguous numpy array so copy-on-write is not defeated |
| Augmentation pipeline | Hue jitter cost 27% of pipeline time for no measurable benefit (§6.1) | Removed |
| Index build | One-shot embedding of the full corpus ran out of GPU memory while building the HNSW graph | Streamed in 20,000-chunk shards with checkpointed resume |
| Distillation training (2 GPUs) | Kaggle's default 2-GPU setup auto-wraps the model for data-parallel training, but a 4-bit model cannot be copied across cards this way — the quantisation metadata doesn't travel, so the first training step crashed | Forced single-GPU with `CUDA_VISIBLE_DEVICES` set before torch loads |

### 9.1 Latency measurement (informational)

| Stage | p50 ms | p95 ms | share of p50 |
|---|---|---|---|
| embed | 21.2 | 34.1 | 0.2% |
| search | 19.3 | 31.9 | 0.2% |
| generation | 11,011.6 | 15,374.5 | 99.5% |
| total | 11,065.1 | 15,413.6 | — |

This is a measurement, not a training result: retrieval (embed+search) is already two orders of magnitude inside budget on GPU hardware. Generation is effectively the entire latency budget, so further retrieval speed-ups are not worth pursuing next.

---

## 10. Validation Results and Model Selection

> This section uses validation metrics only — the metrics used to actually choose a model or checkpoint. Held-out test-set numbers exist for Vision and Intent/Entity/Guardrail; they were checked once but are not used for selection, and are moved to Appendix D as a preview.

### 10.1 Vision — selected checkpoint: `p3_full_best.pt` (epoch 16)

Selection metric: validation wheat-15 macro-F1, not 20-way accuracy. A frozen backbone already separates rice from wheat almost perfectly (0.9835 F1), so a 20-way number is partly "free" — wheat-15 is the metric that actually reflects task difficulty.

Phase D3 had the best validation wheat-15 macro-F1 (0.8793, see §6.1 unfreezing-depth table) and was selected. Cumulative gain over the frozen baseline: +0.083 wheat-15 macro-F1, about 3.3× the noise floor. The D2 → D3 step alone is not statistically significant and is not claimed as one.

A one-time test-set check exists and is summarised in Appendix D; it is not part of the selection decision.

### 10.2 Intent / Entity / Guardrail — selected checkpoint: epoch 5 (composite validation score)

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.938 | 0.545 |
| 2 | 0.426 | 0.439 |
| 3 | 0.339 | **0.441 (lowest)** |
| 4 | 0.276 | 0.471 |
| 5 | 0.230 | 0.475 |

Validation loss bottoms at epoch 3 and rises slightly through epochs 4–5 — normally a sign to stop early. But checkpoint selection here does not use raw validation loss: it uses a composite validation score (a weighted combination of intent macro-F1 and NER entity F1), and that composite score kept improving through epoch 5 (§6.2). So the epoch-5 checkpoint is the correct, validation-based choice, even though the loss curve alone would suggest stopping sooner.

On a standard test split, the guardrail head alone reaches perfect precision and recall. An adversarial red-team check tells a fuller story: model-only guardrail F1 is 0.727 (recall 0.571) — it misses real unsafe queries the standard test set never probed. Rule-based pattern matching alone does better (F1 0.923, recall 0.857) but still misses some. Combining both — flag if either the model or the rules fire — reaches perfect precision and recall (F1 1.000) on the same adversarial set. The combined approach, not the model alone, is what is used for the guardrail decision.

Full test-set numbers for Intent, NER, and Guardrail are in Appendix D as a preview.

### 10.3 Retrieval — no model was trained or selected

There is no training-based selection here. The serving configuration was chosen in §7.1: bge-m3 frozen embeddings, Qdrant HNSW server mode, top_k=10, manifest-based score fusion, thresholds 0.56/0.66.

### 10.4 Generation — selected checkpoint: `best_adapter/` (epoch 2 of 4 allowed)

The final training run used rank 32 and 4 allowed epochs — a combination the hyperparameter study did not test as a single config, but which follows the two directions the study did confirm (higher rank helps, more epochs helps). This is a well-motivated setting, not a proven optimum, and is stated as such rather than claimed as the literal best config from §6.3.

| Metric | Value |
|---|---|
| Final training loss | 0.1398 |
| Best validation loss | 0.1648 |
| Perplexity (from best validation loss) | 1.18 |
| Overfitting gap (val − train) | +0.0250 |
| Epochs completed | 4 of 4 allowed |
| Best checkpoint | checkpoint-360 (end of epoch 2) |
| Training time | 578.1 min (9.6 h) |

Checkpoint selection used `load_best_model_at_end=True`, so the adapter actually saved is the epoch-2 checkpoint, not the epoch-4 one — validation loss stopped improving after epoch 2 and early stopping (patience 2) caught it right at the epoch cap. This matches what the hyperparameter study predicted: its best config also found its validation minimum before the end of a 3-epoch run.

Only the adapter is saved (262 MB), not a merged model, and it is never merged into the 4-bit base — merging LoRA into 4-bit weights is unsafe because the quantisation metadata does not survive it.

The serving-time decoding settings (temperature, output length, context size) were swept separately in §7.2; config hash `d9f0e9118d6e` records that sweep, with one pending change: raise max_new_tokens from 100 to 150.

### 10.5 Yield — selected model: LightGBM (Tuned)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **LightGBM (Tuned) — selected** | 0.83 | 2.41 | 0.9572 |
| XGBoost | 0.77 | 2.47 | 0.9552 |
| XGBoost (Tuned) | 0.85 | 2.54 | 0.9523 |
| LightGBM (default) | 0.86 | 2.56 | 0.9518 |
| CatBoost (Tuned) | 0.91 | 2.71 | 0.9461 |
| CatBoost (default) | 0.99 | 2.90 | 0.9380 |
| PyTorch MLP (baseline) | 5.30 | 11.66 | ~0.00 |

All numbers are on the same 64,021-row held-out test set. LightGBM (Tuned) has the best R²/RMSE and a small file size (1.8 MB), and is the selected model. It belongs to the GBT family Milestone 3 selected, so that decision is now validated rather than just assumed.

The PyTorch baseline is effectively non-functional (R² near zero) — it plateaus after a few epochs, and the likely causes are no feature scaling, no batch normalisation, and a 923-dimensional one-hot input, which gradient boosting handles natively through categorical splits instead. This says the baseline was under-tuned, not that neural approaches cannot work here.

**Honesty note:** the comparison table above is computed on the same held-out test set used for the final numbers, because there was no separate validation-only leaderboard step for comparing model families — only within-family tuning used cross-validation, on the training split. This mixing is called out rather than hidden, and a cleaner family-selection step is on the list for next time (§14).

---

## 11. Milestone 3 (Planned) vs Milestone 4 (Final) Configuration

| Component | M3 plan | M4 final configuration |
|---|---|---|
| Vision | ViT-Small selected; no training details set yet | ViT-S/16 (augreg), native normalisation, batch 64, mid_tf augmentation, 3-phase progressive unfreeze, checkpoint `p3_full_best.pt` (epoch 16) |
| Retrieval | bge-m3 frozen embedding; index approach undecided | bge-m3 frozen, Qdrant HNSW server mode, top_k=10, manifest fusion, thresholds 0.56 / 0.66 |
| Intent/Entity/Guardrail | DistilBERT three-head design | distilbert-base-multilingual-cased, AdamW 3e-5, batch 32, 5 epochs (selected by sweep, §6.2). Intent 88.4% accuracy / 71.5% macro-F1; NER 95.8% F1; guardrail reaches perfect precision/recall only when combined with rule-based checks (§10.2) |
| Synthesis LLM | Distilled 2–4B model | gemma-3-4b-it (4-bit) + QLoRA adapter (r=32, α=64), distilled from teacher-written answers; best validation loss 0.1648 (perplexity 1.18), checkpoint at epoch 2 (§3.4, §10.4) |
| Yield | GBT (gradient-boosted tree) | Delivered — LightGBM (Tuned), from the GBT family M3 specified. MAE 0.83 / RMSE 2.41 / R² 0.9572 on held-out test. Split is still random (§2.1), not the chronological split M3 originally called for (§14) |
| Profitability estimator | Rule-based estimator | Out of M4 scope |

---

## 12. Hyperparameter Experiment Summary — All Components

| Component | Type | Parameter | Options tried | Selected | Result |
|---|---|---|---|---|---|
| Vision | Model training | Normalisation | native / imagenet / dataset stats | native | +1.9 macro-F1 over imagenet stats |
| Vision | Model training | Class balancing | none / sqrt | none | sqrt never helped |
| Vision | Model training | Batch size | 32 / 64 / 96 / 128 | 64 | within 3% of peak throughput, more steps/epoch |
| Vision | Model training | Unfreeze depth | frozen / head / blocks 9–11 / full | full (D3) | +0.098 (head→blocks), +0.013 (blocks→full, within CI) |
| Vision | Model training | Augmentation strength | full / mid / fast | mid | −27% time, no loss in shortcut removal |
| Intent/Entity/Guardrail | Model training | Optimizer | AdamW / SGD | AdamW | composite 1.7+ vs 1.279 |
| Intent/Entity/Guardrail | Model training | Learning rate | 1e-5 / 3e-5 | 3e-5 | composite 1.667 vs 1.713 |
| Intent/Entity/Guardrail | Model training | Batch size | 32 / 64 | 32 | composite 1.713 vs 1.700 |
| Intent/Entity/Guardrail | Model training | Epochs | 3 / 5 | 5 | composite 1.713 → 1.747, largest gain in the sweep |
| Intent/Entity/Guardrail | Model training | Label smoothing | 0.0 / 0.1 | 0.0 | no measurable benefit |
| Generation (distillation) | Model training | Learning rate | 5e-5 / 1e-4 / 2e-4 / 5e-4 | 2e-4 | sharpest lever; 5e-5 was the single worst config |
| Generation (distillation) | Model training | LoRA rank | 8 / 16 / 32 / 64 | 32 | 8 measurably worse, 64 measurably better, 32 inside noise |
| Generation (distillation) | Model training | alpha:rank ratio | 1× / 2× | 2× | 1× measurably worse |
| Generation (distillation) | Model training | Target modules | attention only / MLP only / all seven | all seven | both restricted options worse |
| Generation (distillation) | Model training | Epochs allowed | 1 / 3 (study) / 4 (final) | 4, best kept at epoch 2 | 3-epoch run was the study's best config |
| Generation (distillation) | Model training | Dropout / weight decay | 5 configs swept | 0.05 / 0.0 (baseline) | no measurable effect — spread smaller than seed noise |
| Retrieval | System configuration | top_k | 3 / 5 / 10 | 10 | +7.7% crop_hit for ~7ms |
| Retrieval | System configuration | Intent weighting | on / off | on | route accuracy 1.000 vs 0.889 |
| Retrieval | System configuration | Score fusion weights | flat / 3.0:0.3 / manifest | manifest | route accuracy 1.000 |
| Generation (serving) | System configuration | Temperature | 0.0 / 0.3 / 0.7 | 0.3 | no measurable difference |
| Generation (serving) | System configuration | max_new_tokens | 60 / 100 / 150 | 150 (recommended) | citation rate 0.75 → 1.00 |
| Generation (serving) | System configuration | ctx_top_k | 3 / 5 / 8 | 5 (baseline kept) | mixed effect on grounding vs citation |
| Yield | Model training | Model family | PyTorch MLP / XGBoost / LightGBM / CatBoost | LightGBM (Tuned) | best R²/RMSE, smallest GBM file size |
| Yield | Model training | RandomizedSearchCV budget | XGBoost 20 cand. / LightGBM 20 cand. / CatBoost 15 cand. (3-fold CV) | as tuned | helped LightGBM & CatBoost, not XGBoost |
| Yield | Model training | Target transform | original scale / log1p | original scale | log1p improved MAE, worsened RMSE/R² |

---

## 13. Checkpoint and Version Reference

| Component | Checkpoint / version ID | Status |
|---|---|---|
| Vision | `p3_full_best.pt` (epoch 16); split version `clean_index_v3.csv` | Trained and selected |
| Intent/Entity/Guardrail | `intent_entity_guardrail_model.pt` (514 MB); label maps `intent_entity_label_maps.json` | Trained and selected (epoch-5 checkpoint, composite validation score) |
| Retrieval | Index snapshot `agri_knowledge-*.snapshot` (3.80 GB) + `manifest.json` | Built, not trained (frozen embedder). Manifest required alongside the snapshot — the snapshot alone does not carry threshold/prefix settings |
| Generation | `best_adapter/` (QLoRA, checkpoint-360, epoch 2 of 4); serving config hash `d9f0e9118d6e` | Trained and selected. Adapter only (262 MB), never merged into the 4-bit base |
| Yield | `lightgbm_tuned.txt` (selected); `xgboost_*.json`, `catboost_*.cbm`, `pytorch_model.pth` retained for comparison | Trained and selected |

---

## 14. Known Limitations Carried Into Milestone 5

- Intent classification: rare classes lag well behind common ones (general: F1 0.139 on 53 test examples; specialty_other and post_harvest_storage also weak) — needs oversampling or class-weighted loss.
- Intent, NER, and guardrail labels come from rule-based or weak supervision, not human annotation — label noise is a real possibility, especially for rare classes.
- Guardrail: only the combined model+rules approach was validated on the adversarial red-team set. Broader red-team coverage and a confidence-based deferral system are recommended before wider deployment.
- Pure Hindi training and evaluation data is limited (75 examples in the language breakdown), so Hindi-specific performance is measured on a small sample.
- Generation token cap: raise from 100 to 150 based on the sweep in §7.2.
- Distillation training window: 92% of training rows lost some retrieved context at the 1,024-token window (data is 95th-percentile 1,568 tokens). Worth revisiting with a longer window if memory allows.
- Distillation final run used a single seed; the hyperparameter study measured a 0.0074 validation-loss noise band at study scale, but that variance was not re-measured at full-run scale.
- The teacher-model manifest field for the distillation dataset is incorrect (names a 12B Gemma model; the real teacher was Gemini 2.5 Flash Lite via API) and should be corrected at the source.
- Yield: MAPE is unusable as currently computed (near-zero yield rows) — switch to SMAPE/WAPE before reporting it again, and re-run the error analysis against LightGBM (Tuned) instead of CatBoost (Tuned).
- Yield: the data split is still random, not the chronological split originally specified for out-of-time evaluation; and there is no regional (state/district) breakdown yet for the UP-focused deployment.
- Yield: XGBoost's hyperparameter search did not beat its own defaults — worth a larger search budget or Bayesian optimisation next.
- Three scoring/measurement bugs were found while running the retrieval and generation sweeps in §7 (not model bugs — tool bugs). They are recorded in Appendix E so nobody later mistakes them for real results.

---

## Appendix A — Vision Augmentation Pipeline (Full Detail)

Training transform (`mid_tf`, frozen in `pipeline.py`): `RandomResizedCrop(224, scale 0.65–1.0, bicubic)` → `HFlip(0.5)` → `RandomAffine(15°, translate 0.08, scale 0.92–1.08)` → `ColorJitter(0.35, 0.30, 0.25, no hue)` → `RandomExposureGamma(target 0.18–0.55, p=0.9)` → `GaussianBlur(p=0.35)` → `RandomAdjustSharpness(2.5, p=0.35)` → `Normalize(0.5, 0.5, 0.5)` → `RandomErasing(p=0.15)`.

Evaluation transform is deterministic: `CenterCrop(224)` on a 256px image (crop_pct 0.875).

The `RandomExposureGamma` step uses a power-law tone curve (`x → x^γ`, γ found by bisection). This is monotone and clip-free across the full [0,1] range, which is why it removes the brightness shortcut where a simple linear brightness/contrast jitter could not (§2.3).

## Appendix B — Vision Phase Configurations (Full Detail)

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

## Appendix C — Training Logs (Full Detail)

**Vision — validation macro-F1 by phase**

| Metric | frozen probe | D1 | D2 | D3 |
|---|---|---|---|---|
| 20-way | 0.8382 | 0.7655 | 0.8830 | 0.9042 |
| wheat-15 | 0.7966 | 0.7682 | 0.8660 | 0.8793 |
| rice-5 | 0.9717 | 0.7887 | 0.9429 | 0.9828 |
| accuracy | 0.8559 | 0.8131 | 0.9042 | 0.9143 |
| train/val gap | — | 0.047 | 0.092 | 0.073 |

Diagnostic class across phases: `rice__leaf_smut` scored 0.923 → 0.320 → 0.769 on frozen probe → D1 → D2. The 0.923 was mostly the brightness shortcut; it recovered from genuine learning once the shortcut was removed (§2.3).

**Intent/Entity — per-epoch (total ~949 s)**

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.938 | 0.545 |
| 2 | 0.426 | 0.439 |
| 3 | 0.339 | 0.441 |
| 4 | 0.276 | 0.471 |
| 5 | 0.230 | 0.475 |

**Yield — PyTorch MLP baseline, per-epoch training loss (10 epochs):** 80,415,513 → 44,282 → 5,102 → 912 → 228 → 134 → 134.7 → 144.4 → 132 → 131.7. Loss plateaus/oscillates after epoch 5–6, consistent with the near-zero R² reported in §10.5 — the baseline stopped learning early and the remaining epochs made no measurable difference. This is the baseline, not the selected model (LightGBM Tuned, §10.5).

**Yield — GBM training and search timing:** CatBoost default (~900+ iterations): ~4 min. RandomizedSearchCV: CatBoost 45 fits at ~65–70s per fit; XGBoost 60 fits ranging ~14–50s per fit depending on n_estimators/max_depth; LightGBM 60 fits in a similar range.

## Appendix D — Test-Set Results (Preview Only — Not Used for M4 Selection)

> These numbers were checked once, after model selection, and are shown here only as a preview. Full test-set analysis, per-class error review, and confusion patterns are Milestone 5 work.

**Vision — held-out test set (n = 1,287)**

| Metric | validation (used for selection) | test (preview) | 95% CI (test) | delta |
|---|---|---|---|---|
| wheat-15 macro-F1 | 0.8793 | 0.8631 | [0.8403, 0.8853] | −0.016 |
| 20-way macro-F1 | 0.9042 | 0.8671 | [0.8396, 0.8905] | −0.037 |
| rice-5 macro-F1 | 0.9828 | 0.8823 | [0.7915, 0.9523] | −0.100 |
| accuracy | 0.9143 | 0.8998 | — | −0.014 |

The main metric (wheat-15) held up on test — it dropped only 0.016, well inside the confidence interval, so the validation-based selection did not overfit to the validation split. The rice-5 drop is driven by a single class, `rice__leaf_smut`, on only 8 test images.

**Intent / Entity / Guardrail — test split (n = 3,597)**

| Head | Test result | Note |
|---|---|---|
| Intent | Accuracy 88.4%, macro-F1 71.5% | Macro-F1 is the honest number — rare classes lag well behind (general: F1 0.139 on 53 examples) |
| NER | Entity F1 95.8% | Span-based (entity-level) evaluation, crop entities only |
| Guardrail | Standard test: 1.00 / 1.00 / 1.00. Adversarial red-team set: model-only F1 0.727 (recall 0.571) | Model alone has real recall gaps; combined with rule-based checks it reaches F1 1.000 on the same adversarial set (§10.2) |

**Vision — per-class test results (n = 1,287)**

| Class | F1 | n | Class | F1 | n |
|---|---|---|---|---|---|
| `rice__leaf_smut` | 0.4615 | 8 | `wheat__aphid` | 0.9157 | 86 |
| `wheat__leaf_blight` | 0.5849 | 57 | `wheat__mildew` | 0.9432 | 112 |
| `wheat__tan_spot` | 0.6129 | 65 | `wheat__fusarium_head_blight` | 0.9474 | 64 |
| `wheat__common_root_rot` | 0.8000 | 57 | `wheat__brown_rust` | 0.9496 | 118 |
| `wheat__smut` | 0.8125 | 50 | `wheat__healthy` | 0.9505 | 103 |
| `wheat__mite` | 0.8466 | 76 | `rice__brown_spot` | 0.9697 | 65 |
| `wheat__stem_fly` | 0.8485 | 17 | `wheat__septoria` | 0.9722 | 35 |
| `wheat__black_rust` | 0.8615 | 31 | `wheat__yellow_rust` | 0.9890 | 135 |
| `wheat__blast` | 0.8966 | 58 | `rice__blast` | 0.9895 | 48 |
| | | | `rice__bacterial_blight` | 0.9908 | 55 |
| | | | `rice__tungro` | 1.0000 | 47 |

16 of 20 classes score above 0.80. The three below 0.65 are the same three the frozen-probe diagnostics flagged before any fine-tuning.

## Appendix E — Known-Broken Measurement Instruments

Three scorers used while running the §7 configuration sweeps turned out to be defective. They are recorded here rather than deleted, because a check known to be wrong is worse than no check at all if someone later reads it as a real result.

| Instrument | Defect | Consequence | Fix |
|---|---|---|---|
| `numeric_grounding` (health check G2, sweep 'grounded' column) | Its number-matching pattern does not understand thousands separators ("6,000" vs "6000") and also counts list markers like "1, 2, 3" as quantities | The reported 12/84 figures are **not** a hallucination rate. Manual review found the figures were present in every case checked. This does not clear the M3 dosage-invention risk — it leaves it unmeasured | Normalise separators on both sides; ignore small integers next to list punctuation; re-run |
| Bilingual retrieval check | Compares whether English and Hindi phrasings return the exact same record IDs, not the same topic | Two flagged warnings turned out to be a broken check, not a broken system | Compare retrieved topics, not record identity |
| `lang` metric in the generation sweep | Requires the answer's script to exactly match the question's script, on a mostly code-mixed (Hinglish) corpus | Penalises correct answers. A separately built check (G3) reads 21/21 pass on the same data | Align with the G3 definition |

## Appendix F — Artifacts

| Artifact | Path / size | Notes |
|---|---|---|
| Vision checkpoint | `runs/vits16_m1/ckpt/p3_full_best.pt` (epoch 16) | Phase checkpoints also retained |
| Vision split | `clean_index_v3.csv` (12,823 rows) + `split_plan_v3.csv`, `exclusions.csv` | — |
| Intent/entity model | `final/intent_entity_guardrail_model.pt` (514 MB) | + label maps, training summary |
| Retrieval index | `agri_knowledge-*.snapshot` (3.80 GB) + `manifest.json` | Matched pair — snapshot alone is unsafe to serve |
| Generation serving sweep | `rag_generation_m4/`, config hash `d9f0e9118d6e` | `generations.jsonl` (90 rows), sweep files, `health_check.json`, `latency.json` |
| Distillation training set | `train.jsonl` (23.15 MB), `dataset_manifest.json`, config hash `1d8f4ed7d4ca` | 3,187 rows; `review.csv` holds the KCC reference answer next to each row for human review only |
| Distillation adapter | `best_adapter/` (262.41 MB) + `final_model_manifest.json` + `train_log.csv`, bundled as `final_model_bundle.zip` | Adapter weights + tokenizer + chat template; loads with `PeftModel.from_pretrained` |

Notebooks: `08c_rag_vector_db_bge_m3.ipynb` (index build) · `09_rag_retrieval.ipynb` (query) · `10_rag_generation.ipynb` (serving sweeps) · `11_kcc_intent_entity_guardrail.ipynb` · `yield_training.ipynb` (§3.5, §6.4, §10.5) · `12_distillation_data_prep.ipynb` (training set) · `14_distillation_hpt.ipynb` (27-config study, §6.3) · `13_distill_training.ipynb` (final training run, §10.4).

## Appendix G — Distillation Hyperparameter Study (Full Leaderboard)

27 QLoRA configs, same 160 training rows and same 48 validation rows throughout, ranked by best validation loss (lower is better). Baseline: rank 16, alpha 32, all modules, lr 2e-4, cosine schedule, `paged_adamw_8bit`, effective batch 16, 1 epoch. Noise band from 3-seed repeats: 0.0074.

| Rank | Config | Family | What differs from baseline | Best val loss | vs baseline | Verdict |
|---|---|---|---|---|---|---|
| 1 | epochs_curve | epochs | 3 epochs (not 1) | 0.3148 | −0.1050 | better |
| 2 | batch_08 | optimization | effective batch 8 | 0.3370 | −0.0828 | better |
| 3 | rank_64 | capacity | LoRA rank 64 | 0.3667 | −0.0531 | better |
| 4 | optim_adafactor | optimization | optimizer = adafactor | 0.3934 | −0.0264 | better |
| 5 | sched_constant | optimization | constant LR schedule | 0.4030 | −0.0168 | better |
| 6 | lr_5e-4 | optimization | lr 5e-4 | 0.4169 | −0.0029 | = noise |
| 7 | rank_32 | capacity | LoRA rank 32 | 0.4182 | −0.0016 | = noise |
| 8 | seed_21 | noise | seed 21 | 0.4197 | −0.0001 | = noise |
| 9 | baseline | reference | — | 0.4198 | 0.0000 | = noise |
| 10 | warmup_0.00 | optimization | no warmup | 0.4203 | +0.0005 | = noise |
| 11 | optim_adamw_fp32 | optimization | full-precision AdamW | 0.4238 | +0.0040 | = noise |
| 12 | clip_0.3 | optimization | grad clip 0.3 | 0.4239 | +0.0041 | = noise |
| 13 | dropout_0.00 | regularization | no dropout | 0.4243 | +0.0045 | = noise |
| 14 | wd_0.01 | regularization | weight decay 0.01 | 0.4253 | +0.0055 | = noise |
| 15 | dropout_0.10 | regularization | dropout 0.10 | 0.4262 | +0.0064 | = noise |
| 16 | sched_linear | optimization | linear LR schedule | 0.4267 | +0.0069 | = noise |
| 17 | dropout_0.20 | regularization | dropout 0.20 | 0.4269 | +0.0071 | = noise |
| 18 | seed_07 | noise | seed 7 | 0.4271 | +0.0073 | = noise |
| 19 | wd_0.10 | regularization | weight decay 0.10 | 0.4277 | +0.0079 | worse |
| 20 | mlp_only | capacity | MLP modules only | 0.4677 | +0.0479 | worse |
| 21 | alpha_ratio_1x | capacity | alpha = rank (1×) | 0.4678 | +0.0480 | worse |
| 22 | warmup_0.10 | optimization | warmup 0.10 | 0.4704 | +0.0506 | worse |
| 23 | rank_08 | capacity | LoRA rank 8 | 0.4958 | +0.0760 | worse |
| 24 | lr_1e-4 | optimization | lr 1e-4 | 0.5212 | +0.1014 | worse |
| 25 | attn_only | capacity | attention modules only | 0.5360 | +0.1162 | worse |
| 26 | batch_32 | optimization | effective batch 32 | 0.5550 | +0.1352 | worse |
| 27 | lr_5e-5 | optimization | lr 5e-5 | 0.7088 | +0.2890 | worse |

The winning config (`epochs_curve`, validation loss 0.3148) was re-run once to check reproducibility and gave 0.3040 — a difference of −0.0108, itself within the same run-to-run noise the seed configs measure. The final training run (§10.4) combines rank 32 with 4 allowed epochs; this is not literally any single row above, but follows the two directions the table shows to be real.

---

## Team Review & Sign-Off

Reviewers should read §6.3 (unfreeze depth), §9.4 (latency decomposition), §10.2 (guardrail not
accepted), §10.4 (yield not delivered), §3.4 (distillation dropped) and Appendix E specifically —
these are the sections that change a Milestone 3 position or record a component as not delivered.

| # | Team Member | Approved | Date |
|:-:|-------------|:--------:|:----:|
| 1 | Mahesh | Yes | 10 Aug 2026 |
| 2 | Harliv | Yes | 10 Aug 2026 |
| 3 | Lokesh | Yes | 10 Aug 2026 |
| 4 | Aneeqa | Yes | 10 Aug 2026 |
| 5 | Tanmay | Yes | 10 Aug 2026 |

---

**Milestone 4 Report.** Prepared 10 August 2026.
