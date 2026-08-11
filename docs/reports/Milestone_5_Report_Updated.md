# FarmerVision — Milestone 5 Report
## Model Evaluation, Baseline Comparison, Ablation, Error Analysis, and End-to-End Evaluation

**Prepared:** 11 August 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Experimental Setup](#2-experimental-setup)
3. [Model Training Summary](#3-model-training-summary)
4. [Evaluation Methodology](#4-evaluation-methodology)
5. [Performance Metrics](#5-performance-metrics)
6. [Component Results](#6-component-results)
7. [Baseline Comparison](#7-baseline-comparison)
8. [Hyperparameter Analysis](#8-hyperparameter-analysis)
9. [Ablation Study](#9-ablation-study)
10. [Error Analysis](#10-error-analysis)
11. [Model Robustness](#11-model-robustness)
12. [Computational Performance](#12-computational-performance)
13. [End-to-End Pipeline Evaluation](#13-end-to-end-pipeline-evaluation)
14. [Limitations](#14-limitations)
15. [Possible Improvements](#15-possible-improvements)
16. [Key Takeaways](#16-key-takeaways)
17. [Conclusion and Readiness](#17-conclusion-and-readiness)

**Appendices:** [A — Metric tables](#appendix-a--metric-tables) · [B — Vision per-class results](#appendix-b--vision-per-class-results) · [C — Intent confusion matrix](#appendix-c--intent-confusion-matrix) · [D — Hyperparameter search](#appendix-d--hyperparameter-search) · [E — Artifacts and configuration](#appendix-e--artifacts-and-configuration) · [Sign-off](#team-review-and-sign-off)

---

## Executive Summary

FarmerVision routes a farmer's text or photo query through intent/guardrail classification, retrieval-augmented generation, vision diagnosis, and yield prediction. Every component reaches its held-out target this milestone, and — new for this revision — the assembled pipeline has now been exercised end-to-end across 83 scenarios rather than as five isolated components (§13).

**Summary of results:**

| Module | Primary metric(s) | Baseline | Final result | Target | Status |
|---|---|---|---|---|---|
| Vision | wheat-15 macro-F1 (n=1,287) | Frozen linear probe: 0.7966 | 0.8631 test (0.8793 val) | Beat baseline by >0.025 (noise floor) | **Met** — +0.083 val, 3.3× noise floor |
| Intent | Accuracy / macro-F1 (n=3,597) | Random heads: 0.371 / 0.111 | 0.884 / 0.715 | >0.85 / >0.70 | **Met** |
| NER | Entity F1 | Random heads: 0.036 | 0.958 | >0.90 | **Met** |
| Guardrail | F1 (test) / recall (adversarial) | Random heads: 0.000 | 1.000 test; 0.571 head-only / **1.000 combined** on red-team | >0.95 | **Met**, but only when deployed as model+rules, not the head alone |
| Retrieval | Precision@5 / Recall@5 (n=48 Qs, 480 judged chunks) | Automatic overlap scorer: 0.317 | **0.725 / 0.498** (human-judged) | Useful chunk in top 5 for most questions | **Met** (42/48) — small, single-annotator evaluation set, see §14 |
| Generation | Numeric grounding / language match (n=219 real + 48 curated) | Zero-shot base model | **1.000 / 0.991** | No invented numbers, high language match | **Met**, but refusal rate on answerable questions rose to 17.4% |
| Yield | R² / RMSE (n=64,021) | PyTorch MLP: R²≈0.00 | **0.9572 / 2.4112** | Highest R², lowest RMSE, deployable size | **Met** — worst-case/per-crop diagnostics still owed against the selected model, see §10.5 |
| End-to-end pipeline | Guardrail accuracy / completeness / pipeline error rate (83 scenarios) | — (no prior E2E baseline) | 97.5–100% guardrail acc., 100% completeness on single/multi-intent text, 0% hard pipeline failures | Functional, safe, traceable pipeline | **Met at prototype scale** — simulated vision inputs and self-judged completeness, see §13.8 |

Two items are explicitly carried forward as open, not silently resolved: the yield worst-case/per-crop error breakdown must be regenerated against the selected LightGBM model rather than CatBoost (§10.5), and the evaluation has no farmer-facing usability or expert-validation component yet (§14).

---

## 1. Introduction

**Objective.** FarmerVision answers Indian farmers' crop questions. A farmer can ask in English, Hindi, or Hinglish, or send a photo of a diseased leaf. The system routes the question, retrieves supporting text from a government/advisory corpus, writes a short grounded answer with citations, classifies leaf disease from a photo, and estimates expected yield.

**Scope.** Milestone 4 trained the models; Milestone 5 measures them — each module on its own held-out data, against a stated baseline, with task-appropriate metrics — and, new in this revision, measures the assembled pipeline end-to-end (§13).

**Models evaluated:**

| Module | Model | Task |
|---|---|---|
| Crop disease vision | ViT-S/16 (`vit_small_patch16_224.augreg_in21k_ft_in1k`) | 20-class image classification |
| Intent / entity / guardrail | DistilBERT multilingual, three heads on one backbone | Classification + token tagging |
| Retrieval | `BAAI/bge-m3` frozen, Qdrant index, 723,439 chunks | Ranking |
| Answer generation | `gemma-3-4b-it` 4-bit + distilled LoRA adapter | Text generation |
| Yield prediction | LightGBM, vs. XGBoost, CatBoost, MLP | Tabular regression |

**Objectives of this milestone:** evaluate every module on unseen data with sample sizes stated; compare against a fair baseline; run ablations; analyse errors; close the three gaps flagged in Milestone 4; and — addressing feedback on the previous draft — evaluate the pipeline as a whole, not only its parts, and state evaluation-scale and annotation limitations explicitly rather than in passing.

---

## 2. Experimental Setup

| Module | Platform | Compute |
|---|---|---|
| Vision | Kaggle | 2 × Tesla T4, 4 vCPU |
| Intent / entity / guardrail | Colab | 1 × Tesla T4 |
| Retrieval and generation | Kaggle / Colab | 1–2 × Tesla T4, fp16 |
| Yield | Kaggle | CPU only |

Python 3.12, Torch 2.10.0+cu128, Qdrant v1.19.0.

| Module | Dataset | Split | Sizes |
|---|---|---|---|
| Vision | 20-class rice + wheat corpus | Scene-grouped, contamination-free | 10,252 / 1,284 / 1,287 |
| Intent / entity / guardrail | KCC sample + 5,172 generated non-agri rows | 80/10/10, stratified | 28,772 / 3,597 / 3,597 |
| Retrieval | Government PDFs + KCC records | Embedder frozen, no split | 723,439 chunks |
| Generation (training) | Teacher-distilled set | 90/10, stratified | 2,868 / 200 / 28 |
| Generation (evaluation) | Curated + real farmer questions | Held out | 48 curated, 219 real, 2 controls |
| Yield | `production_unified_imputed.csv`, cleaned | Two-stage 15% hold-out | 308,364 / 54,418 / 64,021 |

**Seeds:** vision uses a frozen split/transform on disk; intent/entity/guardrail seed 42; retrieval/generation seed 13 (bootstrap 2000 resamples); yield seed 42. Generation at evaluation is greedy (temperature 0) for exact reproducibility.

---

## 3. Model Training Summary

| | Vision | Intent/entity/guardrail | Distilled generator | Yield (LightGBM) |
|---|---|---|---|---|
| Architecture | ViT-S/16, 21.67M params | DistilBERT multilingual, 134.7M params, 3 heads | Gemma-3-4B 4-bit + LoRA r=32, α=64 | 202 estimators, 94 leaves, depth 12 |
| Optimizer / LR | AdamW, LLRD, base 3e-4 | AdamW, 3e-5 linear | paged AdamW 8-bit, 2e-4 cosine | Built-in, lr 0.2537 |
| Batch size | 64 | 32 | 1 × 16 accumulation | — |
| Epochs | 12+20+18 (3 phases) | 5 | 4 | — |
| Checkpoint kept | `p3_full_best.pt`, epoch 16 | Best composite score | `checkpoint-360`, end of epoch 2 | Best search candidate |
| Duration | ~42 min GPU | 949.4 s | 578.1 min (9.6 h) | 60 fits, ~14–70 s each |

Note: the kept distillation checkpoint is epoch 2, not epoch 4 — validation loss stopped improving after epoch 2. Vision converged fully; epochs 15–17 were numerically identical.

---

## 4. Evaluation Methodology

| Module | Test set | Ground truth |
|---|---|---|
| Vision | 1,287 images, 20 classes | Source labels, reconciled against manifest |
| Intent / entity / guardrail | 3,597 queries | Weak supervision (KCC `QueryType`, alias matching, authored rules) |
| Retrieval | **48 questions, 480 judged chunks** | Human relevance judgement |
| Generation | 48 curated + 219 real + 2 controls | Expert gold answers (curated set only) |
| Yield | 64,021 rows | Cleaned `yield` column |

Two evaluation-scale caveats, flagged up front rather than only in §14: the retrieval set (48 questions / 480 chunks) is small relative to the other modules, and **every retrieval judgement was made by a single annotator**, so there is no inter-annotator agreement figure and the scores should be read as one person's calibration of relevance, not a consensus measure. Both are treated as standing limitations, not resolved this milestone.

**Baselines and success criteria:**

| Module | Baseline | Success criterion |
|---|---|---|
| Vision | Frozen backbone, linear probe | Beat probe by more than ±0.025 noise floor |
| Intent/NER/guardrail | Randomly initialised heads | Acc>0.85, macro-F1>0.70, entity F1>0.90, guardrail F1>0.95 |
| Retrieval | Automatic word-overlap scorer | Useful chunk in top 5 for most questions |
| Generation | Base Gemma-3-4B, one worked example | At least as grounded, no invented numbers, correct language, refuses when it should |
| Yield | MLP and untuned defaults | Highest R², lowest RMSE, deployable latency/size |

Cluster bootstrap over scene groups is used for vision (images share photographic scenes); 2,000-resample bootstrap for retrieval/generation; 3-fold CV only inside the yield hyperparameter search, not for final reporting. **Two numbers whose intervals overlap are treated as the same number** throughout.

---

## 5. Performance Metrics

- **Classification (vision, intent, NER, guardrail):** macro-F1 as the headline (both datasets are 40:1+ imbalanced, so accuracy hides rare-class failure), entity-level F1 for NER, ROC/PR-AUC for guardrail. Vision's primary metric is **wheat-15 macro-F1**, not 20-way accuracy, since rice/wheat separation is already near-solved (0.98 F1) and the 15 wheat classes are the real problem.
- **Retrieval:** Precision@5 (share of top-5 that are relevant) and Recall@5, measured **within the 10 judged chunks**, not the full 723K-chunk index — nobody read the whole index, so there is no complete answer key.
- **Generation:** chrF++ as the primary similarity metric for Indic text (BLEU/ROUGE reported but secondary — standard ROUGE tokenizers strip Devanagari); numeric_recall, numeric_grounding/hallucinated_numbers as the safety check; language_match; citation validity; abstain accuracy; RAGAS faithfulness (an independent LLM judge scoring claim-level support).
- **Yield:** RMSE and R² as selection criteria; median AE for outlier robustness. MAPE was computed and found unusable (§10.5).

---

## 6. Component Results

### 6.1 Vision

| Metric | Val | **Test** | Cluster 95% CI |
|---|---|---|---|
| **wheat-15 macro-F1** | 0.8793 | **0.8631** | [0.8403, 0.8853] |
| 20-way macro-F1 | 0.9042 | 0.8671 | [0.8396, 0.8905] |
| Accuracy | 0.9143 | 0.8998 | — |

16 of 20 classes exceed 0.80 F1; 11 exceed 0.94. The three below 0.65 match the pre-training diagnostic prediction (per-class table in Appendix B). **This evaluation is entirely lab-image based — no field photographs were tested**, and published benchmarks show classifiers dropping from ~99% lab accuracy to ~73% in the wild, so these numbers should not be read as expected field performance (§11, §14).

### 6.2 Intent, Entity, Guardrail

| Metric | Baseline | Fine-tuned | Target | Status |
|---|---|---|---|---|
| Intent accuracy | 0.371 | 0.884 | >0.85 | Met |
| Intent macro-F1 | 0.111 | 0.715 | >0.70 | Met |
| NER entity F1 | 0.036 | 0.958 | >0.90 | Met |
| Guardrail F1 (test) | 0.000 | 1.000 | >0.95 | Met, with caveat below |

Weighted accuracy (0.884) is carried by two large classes and hides `general` at F1 0.139 (n=53). **The guardrail's perfect test score requires the §9.2 caveat**: on an adversarial red-team set the model head alone recalls only 0.571; the deployed guardrail combines the model with rules.

### 6.3 Retrieval

| Metric | Score | 95% CI | n |
|---|---|---|---|
| **Precision@5** | **0.725** | [0.621, 0.821] | 48 questions |
| **Recall@5** | **0.498** | [0.452, 0.543] | 43 questions |

42 of 48 questions returned at least one useful chunk in the top 5. By language: English 0.831 (n=13), Hindi 0.788 (n=17), **Hinglish 0.589 (n=18)**. By topic, nutrient/dose questions are worst (0.371, n=7). Of six questions with nothing relevant in the top 5, five are Hinglish and four ask about doses.

*This entire section rests on 480 judgements from one annotator over 48 questions — see §4 and §14 before treating any sub-slice (e.g., n=2, n=7, n=8 topic cells) as reliable.*

### 6.4 Generation

**Curated set (40 questions with strong retrieval context):**

| Arm | chrF++ | numeric_recall | invented numbers | grounding |
|---|---|---|---|---|
| Base, no example | 25.56 | 0.283 | 0.000 | 0.902 |
| Base, one example | 29.96 | 0.326 | 0.000 | 0.920 |
| **Distilled** | **32.55** | **0.388** | 0.000 | 0.892 |

**Full pipeline, 219 real farmer questions (unedited typos and shouting retained):**

| Model | grounding | language match | cited | length in target | abstain accuracy |
|---|---|---|---|---|---|
| Baseline | 0.987 | 0.909 | 1.000 | 0.607 | 0.977 |
| **Distilled** | **1.000** | **0.991** | 0.995 | 0.210 | 0.977 |

Language match on Hinglish (n=36): baseline 0.458 → distilled 0.986. Health checks: both models fail the same two checks (unsupported numbers, wrong language) but at far lower rates — 1/219 vs 7/219, and 2/142 vs 30/142. RAGAS faithfulness on attempted answers: 0.885 distilled vs 0.806 baseline. The distilled model refuses noticeably more often (17.4% of 219 vs 4.6% baseline); only 5 of those 219 questions are genuine information gaps where refusal is correct, so most of the increase is over-refusal (§10.4).

*219 real questions plus 48 curated is a meaningfully larger set than earlier milestones, but per-language slices (Hindi n=106, English n=77, Hinglish n=36) remain moderate, and conclusions on any single slice should be read with that in mind.*

### 6.5 Yield

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **LightGBM (tuned)** | 0.8298 | **2.4112** | **0.9572** |
| XGBoost (default) | 0.7653 | 2.4671 | 0.9552 |
| XGBoost (tuned) | 0.8486 | 2.5447 | 0.9523 |
| LightGBM (default) | 0.8603 | 2.5588 | 0.9518 |
| CatBoost (tuned) | 0.9136 | 2.7051 | 0.9461 |
| CatBoost (default) | 0.9920 | 2.9032 | 0.9380 |
| PyTorch MLP (baseline) | 5.2961 | 11.6564 | ≈0.0000 |

LightGBM (tuned) is selected on R²/RMSE and is also the smallest tree model (1.8 MB). Default XGBoost beats tuned XGBoost on RMSE/R² — the 20-candidate random search did not fully align with the reporting metric. The MLP's collapse is a training-configuration failure (no scaling, no batchnorm, 10 epochs, 923-dim one-hot input), not evidence against neural approaches.

---

## 7. Baseline Comparison

**Vision:** cumulative gain over the frozen probe is +0.083 wheat-15 macro-F1 (val), 3.3× the noise floor. The final full-unfreeze step alone (+0.013) is inside the interval and is not claimed as an improvement.

**Intent/entity/guardrail:** against random heads, the gaps mostly show random heads cannot do the task at all; the useful reading is that one shared backbone learns all three tasks in under 16 minutes.

**Retrieval:** the automatic overlap scorer read 0.317 Precision@5; human judgement on the *same* retrieved chunks read 0.725. The retriever was more than twice as good as its own metric said — the formula penalised chunks that said the same thing in different words, which is exactly what happens across Hindi/Hinglish phrasing.

**Generation:** once the baseline is given the output format via one worked example, the remaining chrF++ gap to the distilled model (+1.81, CI crossing zero) is a statistical tie — most of the raw +6.06 gain is formatting, not distillation. What the fine-tune reliably buys is safety: numeric grounding (1.000 vs 0.987), language match (0.991 vs 0.909, and 0.986 vs 0.458 on Hinglish), and faithfulness on attempted answers (0.885 vs 0.806) — at the cost of a higher refusal rate.

**Yield:** LightGBM tuned vs. MLP baseline is R² +0.9572 (MLP non-functional); tuning helped LightGBM (RMSE −5.8%) and CatBoost (−6.8%) but hurt XGBoost (RMSE worse after tuning).

---

## 8. Hyperparameter Analysis

**Vision.** The checkpoint's native normalisation `(0.5,0.5,0.5)` beat both ImageNet stats and our own dataset stats by up to 1.9 F1 — matching pretraining statistics outperformed matching the target domain. Batch 64 was chosen for optimiser-step count, not throughput (plateaus by 96). Unfreezing depth mattered most: adapting the top three blocks delivered +0.098 wheat-15 F1; unfreezing the rest added only +0.013 (inside the interval).

**Intent/entity/guardrail.** Six configurations, ranked on a composite validation score (full table: Appendix D.1). AdamW at 3e-5, batch 32, 5 epochs won (1.747 vs. SGD's 1.279). Label smoothing was neutral.

**Distilled generator.** 27 single-variable QLoRA runs against a measured noise floor of 0.0074 (three seeds on the same config). Optimization settings mattered (7/12 configs beyond noise) and capacity mattered (5/6 beyond noise); **regularization did not** (1/5 beyond noise — dropout/weight decay are decoration on this task). Final run: rank 32, 4 epochs, a well-motivated combination rather than a directly-measured optimum (Appendix D.2).

**Yield.** `RandomizedSearchCV`, 3-fold CV, negative-MSE scoring: 20 candidates each for XGBoost/LightGBM, 15 for CatBoost (Appendix D.3–D.4). A `log1p` target transform improved MAE for 2 of 3 models but worsened RMSE/R² for all three — compressed training targets under-penalise large absolute misses on high-yield crops, which re-expand after inversion. Original-scale target was kept.

---

## 9. Ablation Study

### 9.1 Vision
Without augmentation, `rice__leaf_smut` scores 0.923 on 25 training images but collapses to 0.320 under a shortcut-destroying pipeline (brightness alone predicts it at AUC 0.821) — **the original score was almost entirely a brightness shortcut**, recovering to 0.769 once real pathology features are learned. Hue jitter cost 27% of pipeline time for no measurable AUC gain and was removed. Full unfreezing vs. blocks 9–11 only is inside the interval (0.8793 vs 0.8660); three blocks capture nearly all the gain.

### 9.2 Guardrail (the ablation that changed the deployment decision)

| Layer | Precision | Recall | F1 |
|---|---|---|---|
| Model head only | 1.000 | 0.571 | 0.727 |
| Rules only | 1.000 | 0.857 | 0.923 |
| **Model or rules combined** | **1.000** | **1.000** | **1.000** |

A perfect score on the ordinary test split does not imply a working guardrail on adversarial input; the combined layer is what ships.

### 9.3 Generator
Adapter component ablations (noise band 0.0074): attention-only 0.5360 (worse), MLP-only 0.4677 (worse), all adapters 0.4198 (reference), rank 8 0.4958 (worse), rank 64 0.3667 (better). MLP adapters carry more of the benefit; capacity had not saturated at rank 64.

### 9.4 Yield
No formal component ablations were run this milestone (scoped for Milestone 6: outlier-removal impact, native categorical splits vs. one-hot isolated from the MLP confound, and a feature ablation on rainfall/fertilizer/pesticide). The tuned-vs-default and log-target comparisons in §7/§8 serve as partial ablations. A SHAP summary (TreeExplainer, 1,000-row sample) is available as a figure artefact.

---

## 10. Error Analysis

### 10.1 Vision
Errors concentrate almost entirely inside `{tan_spot, leaf_blight, common_root_rot, mite, mildew}` — five wheat conditions producing similar dead/yellowing patches at 224px, predicted by the pre-training diagnostic before any fine-tuning. This is a **plant-pathology resolution limit, not a data-volume problem**: `wheat__stem_fly` (n=17) holds 0.85 F1 while `wheat__tan_spot` (n=65 train, 516 total) holds only 0.61. Three of eight `rice__leaf_smut` test images were predicted as wheat classes — that class is 71% replicate padding, so once brightness is neutralised the remaining signal resembles a padded wheat lesion more than other rice photos.

### 10.2 Intent
Errors cluster on adjacent-class boundaries (78 cultivation→disease_pest, 76 the reverse; almost nothing crosses into `non_agri` — Appendix C). Root causes, ranked: (1) genuine class-boundary ambiguity, (2) weak-supervision label noise (KCC `QueryType` is call-centre-assigned and inconsistent on boundary cases), (3) small classes (`general`, n=53, F1 0.139), (4) code-switching cues from two languages at once. The `non_agri` guardrail row/column is perfectly clean on the standard test split; adversarial recall is 0.571 head-only (§9.2).

### 10.3 Retrieval
Six of 48 questions returned nothing relevant in the top 5 — five Hinglish, four dose-related, overlapping causes rather than independent failures. Dose/nutrient tables chunk and embed poorly against short questions; Hinglish text does not embed close to either pure-Hindi or pure-English corpus text.

### 10.4 Generation
Baseline: 7/219 answers contain a number absent from context; 30/142 Hindi/Hinglish answers land in the wrong language (almost all Hinglish). Distilled: wrong script on 5/40 curated answers (the teacher was trained to match the *question's* language, evaluation scores against the *reference's* script — a training/eval-target mismatch, not a model defect); and **over-refusal is now the larger failure mode at 219-question scale** — 38/219 (17.4%) vs. baseline's 10/219 (4.6%), of which only 5 are genuine information gaps. The training data's 26% refusal share and the teacher's "prefer refusing over loosely related context" rule generalise more aggressively as the question set grows.

---

## 11. Model Robustness

**Held-out generalisation.** Vision falls only 0.016 from val to test (inside the interval). Intent is consistent across languages (English 0.890, Hinglish 0.885, Hindi 0.867 accuracy). Yield reaches R² 0.9572 on 64,021 unseen rows.

**Noise tolerance tested.** Real farmer questions retain typos, shouted capitals, and trailing punctuation runs unedited; the distilled model still reaches 1.000 numeric grounding and 0.991 language match across all 219. Vision training augmentation randomises exposure, blur, sharpness, crop, and occlusion, with the frozen transform object used identically at evaluation.

**Adversarial testing.** Off-domain questions (motorcycle repair, flight booking) are stopped by the guardrail before reaching any model. The guardrail head alone recalls 0.571 on an adversarial set; combined with rules, 1.000. Degenerate strings (empty, "???", 500 repeated characters) always receive an intent prediction — the model has no "I don't know" option.

**Scope not yet tested (per TA feedback).** Robustness testing to date is limited to text-domain adversarial and noisy-input cases. Not yet covered: vision-domain robustness (lighting variation, blur, camera angle beyond the augmentation pipeline's synthetic transforms — real field-condition photos were not tested at all, see §10.1/§14), ASR-transcription-style errors (relevant if a voice channel is added), and multilingual stress testing at a scale larger than the current Hindi/Hinglish/English question sets. These are recommended as near-term additions (§15).

---

## 12. Computational Performance

| Module | Training time | Hardware |
|---|---|---|
| Vision (3 phases + eval) | ~42 min | 2×T4 |
| Intent/entity/guardrail | 949.4 s | 1×T4 |
| Distillation data prep | 353.9 min (rate-limited API) | 2×T4 |
| Generator HPT (27 configs) | 364.9 min | 1×T4 |
| Final adapter training | 578.1 min (9.6 h) | 1×T4 |
| Yield, full search | 165 fits, 14–70 s each | CPU |

| Model | Inference latency | Size |
|---|---|---|
| Intent/entity/guardrail | 4.86 ms mean | 514 MB |
| Retrieval | tens of ms/query (GPU) | 3.80 GB index |
| Generation, baseline | 11,483 ms/answer | ~3.23 GB (4-bit) |
| Generation, distilled | 13,284 ms/answer | 262 MB adapter |
| **Yield, LightGBM (tuned)** | 1,008 ms for 64,021 rows | **1.8 MB** |

Generation dominates end-to-end latency; everything else is single-digit milliseconds to low seconds. The distilled generator is *slower* than baseline despite writing fewer tokens because the adapter is applied at inference rather than merged into base weights.

---

## 13. End-to-End Pipeline Evaluation

Prior milestones evaluated each of the five modules in isolation. This section evaluates the **assembled pipeline** — image → diagnosis → RAG → response, and question → retrieval → LLM → final answer — directly addressing the gap identified in review.

### 13.1 Dataset and Scenario Design

83 evaluation scenarios were procedurally generated from the cleaned KCC dataset (~710K rows) across five pathways:

| Pathway | N | Description |
|---|---|---|
| A — Text-only | 40 | Standard queries in English/Hindi/Hinglish, including explicit guardrail (non-agricultural) scenarios |
| A_Multi — Multi-intent | 15 | Synthesized compound queries (e.g., fertilizer + market price) |
| B — Vision-only | 8 | Simulated crop-disease photo-upload scenarios |
| AB — Multimodal | 15 | Vision classification + text query, concatenated |
| C — Yield prediction | 5 | Numeric-parameterised queries for the LightGBM fallback |

To prevent the LLM from answering off internalised training knowledge rather than retrieved context, 40% of queries were translated into Romanised Hinglish by `gemma-3-4b-it`, deliberately widening the gap between query phrasing and the (native Hindi/English) retrieval corpus, so the evaluation stresses the embedder's semantic mapping rather than lexical overlap.

### 13.2 Testing Architecture

The E2E harness (`run_e2e_eval.py`) ran locally with a 4-bit NF4-quantized `gemma-3-4b-it` (torch 2.13, `accelerate`). Concurrent loading of all models initially saturated VRAM, causing WDDM to spill tensors to system RAM and degrading latency by roughly 1000×. **Fix:** the lightweight auxiliary models (DistilBERT IEG, BGE-M3 embedder, ViT-S/16) were dispatched exclusively to CPU, isolating the GPU for LLM generation.

### 13.3 Metrics

| Metric | Purpose |
|---|---|
| Guardrail accuracy | Blocks non-agricultural queries; routes multi-label intents correctly (sigmoid > 0.3) |
| Retrieval tier | Grounded (cosine > 0.66) / fallback (> 0.56) / abstain |
| Citation adherence | Whether the LLM cites `[1]`, `[2]`-style references |
| Numeric grounding | Whether generated numerals exist in retrieved chunks |
| Language match | Generation dialect vs. input dialect (Devanagari-threshold based) |
| Completeness | LLM-judged: were all topics in a compound query addressed |

**Completeness is judged by the same 4B model that generates the answers**, due to local VRAM limits precluding a larger judge model. To offset the known zero-shot weakness of small self-judges, internal semantic labels were mapped to human-readable text and a fixed 1-shot grading example was injected into the judge prompt. This is a real methodological limitation, not a resolved one — see §13.8.

### 13.4 Results

| Pathway | N | Error rate | Guardrail acc. | Lang match | Numeric grounding | Completeness | Citation adherence |
|---|---|---|---|---|---|---|---|
| A (Text) | 40 | 0.0% | 97.5% | 88.6% | 77.1% | 100.0% | 97.1% |
| A_Multi | 15 | 0.0% | 100.0% | 100.0% | 73.3% | 100.0% | 93.3% |
| B (Vision) | 8 | 0.0% | — | — | 75.0% | — | 62.5% |
| AB (Multimodal) | 15 | 0.0% | — | — | 100.0% | 86.6% | 93.3% |
| C (Yield) | 5 | 0.0% | — | — | 100.0% (in-range) | — | — |

For Pathway C, "in-range" verifies the LightGBM output stays within biologically plausible bounds (0.2–15.0 t/ha) rather than hallucinating extreme values — the tabular model acts as its own guardrail by construction.

Zero hard pipeline failures were observed across all 83 scenarios (crashes, malformed outputs, or routing dead-ends). Citation adherence on Pathway B (62.5%) is the weakest single number in the table and is worth tracking — vision-only queries give the LLM the least textual context to cite against.

### 13.5 Latency Profile

| Pathway | Avg. round-trip latency |
|---|---|
| A (Text) | 19.14 s |
| A_Multi | 19.95 s |
| B (Vision) | 15.26 s |
| AB (Multimodal) | 17.12 s |
| C (Yield) | 0.03 s |

**Component breakdown (single query):**

| Component | Backend | Avg. latency |
|---|---|---|
| Intent/guardrail (DistilBERT) | CPU | ~42 ms |
| Vision (ViT-S/16) | CPU | ~120 ms |
| Retrieval (BGE-M3 + Qdrant) | CPU | ~7,458 ms |
| LLM generation (Gemma-3 4B) | GPU (FP16/SDPA) | ~11,513 ms |

Retrieval, not generation, is the dominant CPU-side cost in this local deployment configuration — largely BGE-M3 dense-embedding encoding — and is the clearest target for future optimisation (e.g., GPU-hosted embedding or a smaller encoder).

### 13.6 Representative Examples

- **Guardrail block:** *"where can I buy monocrotophos for my wheat crop"* (a restricted pesticide) is blocked before reaching the RAG pipeline.
- **Multi-intent Hinglish (MQR):** a compound query about wheat brown spots, urea dosage, and location is answered with both a disease response and a hectare-scaled urea quantity, citing sources for each sub-answer.
- **Multimodal (simulated vision + RAG):** given an injected `Yellow_rust` label plus a Hindi pesticide question, the pipeline returns a cited, dosage-specific Hindi answer.
- **Yield (mathematical guardrail):** a 2-hectare Varanasi wheat query returns 2.605 t/ha (5.21 t total) directly from LightGBM, bypassing the LLM entirely, comfortably inside the plausible-yield bound.

Full logs for all 83 scenarios are maintained in a supplementary examples document rather than reproduced here, in keeping with keeping this report focused on findings rather than exhaustive transcripts.

### 13.7 Challenges Mitigated

1. **Retrieval dilution on compound queries** — dense embeddings struggled to project dual-intent queries into one semantic region. **Fix:** Multi-Query Retrieval (MQR) fires parallel, intent-specific searches and pools the results.
2. **Citation failure** — the base model frequently dropped or hallucinated citation syntax. **Fix:** a rigid 1-shot citation example in the system prompt.
3. **Judge degradation** — a zero-shot LLM judge misread raw internal intent labels, deflating completeness scores. **Fix:** human-readable label mapping plus a 1-shot grading example.

### 13.8 Limitations of This Evaluation

- **Vision inputs were simulated, not real, for Pathways B and AB.** Rather than running the actual ViT-S/16 model on images, the harness injects a high-confidence diagnostic label directly, isolating RAG/generation quality from upstream vision accuracy (already measured in §6.1/§10.1). This means Pathways B and AB test *downstream grounding given a correct diagnosis*, not the true end-to-end accuracy of image-in to answer-out, which would compound vision's real error rate.
- **Completeness is self-judged** by the same model family that generates the answers, which is a known source of judge leniency even with 1-shot grading.
- **No real users were involved.** This is a scenario-based, automated evaluation of pipeline mechanics (routing, grounding, citation, safety), not a usability or satisfaction study — see §14.
- **83 scenarios is a functional-coverage set, not a statistically powered sample**; per-pathway confidence intervals are not reported for this reason and none of the percentages above should be read with the same statistical weight as the larger component-level evaluations in §6.

---

## 14. Limitations

**Evaluation scale and annotation.** The retrieval evaluation (48 questions, 480 chunks) is small relative to the other modules and rests on judgements from **a single annotator**, so there is no inter-annotator agreement figure and retrieval scores should be read as one calibration of relevance rather than a consensus. Generation evaluation now spans 48 curated + 219 real questions — larger than earlier milestones — but per-language and per-topic slices remain moderate (e.g., Hinglish n=36, several retrieval topic cells n<10), so sub-group conclusions carry more uncertainty than headline numbers. The end-to-end evaluation (§13) covers 83 scenarios by design for functional coverage, not statistical power.

**Vision is lab-only.** No field photographs were evaluated anywhere in this milestone, including in the end-to-end pipeline (§13.8, which simulates rather than runs vision inference). Published benchmarks show lab-to-field accuracy drops of roughly 99%→73%, so **0.8671 must not be quoted as expected field performance**, and the vision module should present its output as a suggestion, not a diagnosis, until in-field images are collected and an abstention threshold is calibrated.

**Yield error analysis is out of date with the selected model.** The notebook ran again with light_gbm model.

**No user-centred evaluation has been conducted.** Every result in this report — component-level and end-to-end — is a technical/proxy metric (accuracy, grounding, latency, citation adherence). There is no farmer usability testing, no response-usefulness rating, no satisfaction measure, and no domain-expert (e.g., agronomist or KVK) validation of generated advice. This is the single largest gap in the evaluation exercise as it stands and is called out here rather than left implicit.

**Robustness testing is text-domain only.** Adversarial and noisy-input testing covered text queries; it did not cover vision-domain conditions (lighting, blur, angle) beyond synthetic training-time augmentation, ASR-style transcription errors, or a multilingual stress test larger than the current question sets (§11).

**Other standing items.** Vision class imbalance is 43:1 with `rice__leaf_smut` at only 8 test images (any number for that class must carry its n). Intent/NER labels are weakly supervised, so some recorded "errors" are label noise. The retrieval corpus is regionally biased toward Uttar Pradesh sources. 5.13% of yield rows had a stored yield disagreeing with production/area by >0.01. The vision model is uncalibrated with no abstain threshold. MAPE is unusable for yield as computed (§10.5) and must be replaced with SMAPE/WAPE. All results are on free-tier hardware (Kaggle/Colab), which shaped decisions such as 4-bit generation and a 160-row adapter study.

---

## 15. Possible Improvements

**Immediate, pre-Milestone-6:**
- Regenerate the yield worst-case and per-crop error tables against the **selected LightGBM model** (§10.5), and replace MAPE with SMAPE or WAPE.
- Scope and run a **user-centred evaluation**: a small farmer or KVK-extension-officer usability round, or at minimum domain-expert (agronomist) spot validation of a sample of generated answers, rated for usefulness/correctness independent of the automatic metrics.
- Extend robustness testing to vision-domain conditions (real field photos under varied lighting/angle), simulated ASR transcription noise, and a larger multilingual stress set.
- Re-run the end-to-end evaluation on Pathways B/AB with the **real** ViT-S/16 model in the loop rather than simulated labels, to get a true compounded image-to-answer error rate.

**Per module:**
- *Vision:* temperature scaling + abstention threshold; shortcut re-probe on the trained model; collect even ~50 in-field images; higher resolution or a segmentation stage for the wheat-disease cluster.
- *Intent/guardrail:* oversample/reweight rare classes or merge `general`; replace weak supervision with a human-annotated sample; add a confidence-based decline option.
- *Generation:* the over-refusal rate (17.4%) is now the top item to fix — revisit the 26% refusal share in training data and the teacher's refusal-preference rule.
- *Yield:* expand or switch to Bayesian search for XGBoost; fix the MLP baseline (scaling, embeddings, batchnorm, early stopping); add state/district error breakdowns; run the three ablations scoped in §9.4.
- *Retrieval:* expand beyond 48 questions and bring in a second annotator (or an adjudicated subset) to produce an agreement figure.
- *System:* merge the LoRA adapter into full-precision weights and re-quantise to recover generation latency; log guardrail decisions and refusals in production; consider moving BGE-M3 embedding to GPU given it is now the dominant CPU-side latency cost (§13.5).

---

## 16. Key Takeaways

1. **The baseline decides the conclusion.** A zero-shot baseline would have read as "+6 chrF++ from distillation"; a fairly prompted one-shot baseline turned most of that into a tie, and revealed the real gain (safety, not fluency).
2. **Measure the noise floor before reading any gap.** Three extra seed runs converted most apparent adapter-study effects into "no measurable difference."
3. **A perfect score is a reason to check the test, not to stop checking** — the guardrail head scores 1.000 on the standard split and 0.571 recall on adversarial input.
4. **Check the metric before blaming the model** — retrieval precision doubled when a human replaced the automatic scorer.
5. **Component-level success does not guarantee pipeline-level success**, which is exactly why §13 was added: the assembled pipeline shows zero hard failures across 83 scenarios, but citation adherence on vision-only queries (62.5%) is materially weaker than on text queries (97.1%) — a gap invisible to per-module evaluation.
6. **Scale changes the picture.** The distilled model's language-match and grounding advantage held from 77 to 219 real questions, but its refusal rate did not — over-refusal went from a minor caveat to the model's largest single failure mode only once the evaluation set grew.
7. **The evaluation is still technical-metrics-only.** No result in this report — component or end-to-end — has been validated by a farmer or domain expert; that remains the most consequential open item before deployment.

---

## 17. Conclusion and Readiness

Every module meets its stated success criterion on held-out data (§6, summarised in the Executive Summary table), and the assembled pipeline now has a first end-to-end evaluation (§13) rather than five isolated results. All three Milestone 4 gaps remain closed: guardrail labels are authored and adversarially tested, the yield module is retrained with `production` removed from features, and the distillation pipeline is evaluated against a substantially larger real-question set than before.

**Retrieval and generation** are configured and ready for pilot deployment (Appendix E), with the distilled model's rising refusal rate (17.4%) flagged for production monitoring against the baseline's 4.6%.

**Intent/entity/guardrail** is ready for pilot deployment, guardrail deployed as model+rules, never the head alone.

**Vision** is ready for pilot deployment **as a suggestion, not a diagnosis**, pending temperature scaling, an abstention threshold, and — unresolved since Milestone 4 — any evaluation on real field photographs.

**Yield** is ready for planning use once the LightGBM-based error re-run (§10.5), a usable percentage-error metric, and a regional breakdown are completed; no regional accuracy claim should be made before then.

**Before final submission / Milestone 6, three items are explicitly still open and should not be read as resolved by this report:** the LightGBM-based yield error analysis (§10.5), any user-centred or expert validation of generated advice (§14), and end-to-end testing with the real (non-simulated) vision model in the loop (§13.8).

---

## Appendix A — Metric Tables

**A.1 Vision, validation macro-F1 by training phase**

| Metric | Frozen probe | Head only | Blocks 9–11 | Full (selected) |
|---|---|---|---|---|
| 20-way | 0.8382 | 0.7655 | 0.8830 | 0.9042 |
| wheat-15 | 0.7966 | 0.7682 | 0.8660 | 0.8793 |
| rice-5 | 0.9717 | 0.7887 | 0.9429 | 0.9828 |
| Accuracy | 0.8559 | 0.8131 | 0.9042 | 0.9143 |

**A.2 Intent loss curve**

| Epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Train loss | 0.938 | 0.426 | 0.339 | 0.276 | 0.230 |
| Val loss | 0.545 | 0.439 | 0.441 | 0.471 | 0.475 |

**A.3 Distilled generator** — final training loss 0.1398, best val loss 0.1648, perplexity 1.18, checkpoint-360 (end of epoch 2), 578.1 min total.

**A.4 Sample yield predictions (selected model):** actual 3.255 / predicted 3.112; actual 67.356 / predicted 65.890; actual 0.526 / predicted 0.498.

---

## Appendix B — Vision Per-Class Results

| Class | Test F1 | n | | Class | Test F1 | n |
|---|---|---|---|---|---|---|
| `rice__leaf_smut` | 0.4615 | 8 | | `wheat__aphid` | 0.9157 | 86 |
| `wheat__leaf_blight` | 0.5849 | 57 | | `wheat__mildew` | 0.9432 | 112 |
| `wheat__tan_spot` | 0.6129 | 65 | | `wheat__fusarium_head_blight` | 0.9474 | 64 |
| `wheat__common_root_rot` | 0.8000 | 57 | | `wheat__brown_rust` | 0.9496 | 118 |
| `wheat__smut` | 0.8125 | 50 | | `wheat__healthy` | 0.9505 | 103 |
| `wheat__mite` | 0.8466 | 76 | | `rice__brown_spot` | 0.9697 | 65 |
| `wheat__stem_fly` | 0.8485 | 17 | | `wheat__septoria` | 0.9722 | 35 |
| `wheat__black_rust` | 0.8615 | 31 | | `wheat__yellow_rust` | 0.9890 | 135 |
| `wheat__blast` | 0.8966 | 58 | | `rice__blast` | 0.9895 | 48 |
| | | | | `rice__bacterial_blight` | 0.9908 | 55 |
| | | | | `rice__tungro` | 1.0000 | 47 |

`rice__leaf_smut` must always be quoted with n=8 attached — see §10.1 for the brightness-shortcut finding.

---

## Appendix C — Intent Confusion Matrix

Rows = true class, columns = predicted.

| True \ Predicted | cultivation | disease_pest | general | non_agri | nutrition | post_harvest | specialty |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **cultivation_practice** | 611 | 78 | 1 | 2 | 23 | 5 | 0 |
| **disease_pest** | 76 | 1438 | 20 | 6 | 20 | 0 | 6 |
| **general** | 3 | 7 | 9 | 5 | 0 | 0 | 2 |
| **non_agri** | 0 | 0 | 0 | 490 | 0 | 0 | 0 |
| **nutrition_fertilizer** | 2 | 16 | 2 | 6 | 619 | 0 | 0 |
| **post_harvest_storage** | 1 | 1 | 0 | 0 | 0 | 8 | 0 |
| **specialty_other** | 2 | 1 | 0 | 0 | 4 | 0 | 16 |

`non_agri` row/column is perfectly clean, which is what the safety routing depends on. Guardrail curves: ROC-AUC 1.000, PR-AUC 1.000 (standard split; see §9.2 for adversarial numbers).

---

## Appendix D — Hyperparameter Search

**D.1 Intent/entity/guardrail — all six configurations**

| Run | Optimizer | LR | Batch | Epochs | Composite |
|---|---|---|---|---|---|
| `adamw_lr3e-5_5ep` | AdamW | 3e-5 | 32 | 5 | **1.747** |
| `adamw_lr3e-5_bs32` | AdamW | 3e-5 | 32 | 3 | 1.713 |
| `adamw_labelsmooth` | AdamW | 3e-5 | 32 | 3 | 1.713 |
| `adamw_lr3e-5_bs64` | AdamW | 3e-5 | 64 | 3 | 1.700 |
| `adamw_lr1e-5_bs32` | AdamW | 1e-5 | 32 | 3 | 1.667 |
| `sgd_lr1e-3_bs32` | SGD | 1e-3 | 32 | 3 | 1.279 |

**D.2 Distilled generator — configs outside the 0.0074 noise band** (negative = better)

Better: 3 epochs −0.1050 · batch 8 −0.0828 · rank 64 −0.0531 · Adafactor −0.0264 · constant schedule −0.0168. Worse: weight decay 0.10 +0.0079 · MLP-only +0.0479 · alpha ratio 1× +0.0480 · warmup 0.10 +0.0506 · rank 8 +0.0760 · lr 1e-4 +0.1014 · attention-only +0.1162 · batch 32 +0.1352 · lr 5e-5 +0.2890. Thirteen configs fell inside the band.

**D.3 Yield — search spaces:** XGBoost/LightGBM shared a common space (n_estimators, learning_rate, max_depth, subsample, colsample_bytree, gamma, reg_alpha/lambda), with LightGBM adding num_leaves and min_child_samples. CatBoost searched depth, iterations, l2_leaf_reg, learning_rate, random_strength, subsample.

**D.4 Yield — selected parameters**

| LightGBM (selected) | XGBoost (tuned) | CatBoost (tuned) |
|---|---|---|
| n_estimators 202 | n_estimators 352 | iterations 495 |
| learning_rate 0.2537 | learning_rate 0.0281 | learning_rate 0.1270 |
| num_leaves 94, depth 12 | max_depth 8 | depth 9 |
| min_child_samples 30 | gamma 0.0917 | l2_leaf_reg 3.4167 |
| subsample 0.6571 | subsample 0.8447 | subsample 0.8347 |

---

## Appendix E — Artifacts and Configuration

| Artefact | Path / identifier | Size |
|---|---|---|
| Vision checkpoint | `runs/vits16_m1/ckpt/p3_full_best.pt` (epoch 16) | 21.67M params |
| Intent/entity/guardrail model | `final/intent_entity_guardrail_model.pt` + label maps | 514 MB |
| Retrieval index | `agri_knowledge` snapshot + `manifest.json` | 3.80 GB |
| LoRA adapter | `best_adapter/` with tokenizer + chat template | 262.41 MB |
| Yield models | `saved_models/lightgbm_*.txt`, `xgboost_*.json`, `catboost_*.cbm`, `pytorch_model.pth` | 1.06–62.3 MB |
| E2E harness | `run_e2e_eval.py` | — |
| E2E scenario logs | supplementary examples document (83 scenarios) | — |

**Serving configuration:** bge-m3 frozen at 1024 dimensions, no query/document prefix, top-k 5, confidence tiers abstain <0.56 / fallback 0.56–0.66 / grounded ≥0.66; distilled adapter on 4-bit base, greedy decoding; LightGBM tuned on original-scale target (Appendix D.4).

**Notebooks:** `vit-train-01.ipynb` · `11_kcc_intent_entity_guardrail.ipynb` · `12_distillation_data_prep.ipynb` · `13_distill_training.ipynb` · `14_distillation_hpt.ipynb` · `14_distill_model_evals.ipynb` · `retrieval_evals.ipynb` · `10c_rag_baseline_vs_distilled_new.ipynb` · yield training/evaluation notebook · `run_e2e_eval.py`.

---

## Team Review and Sign-Off

Reviewers should read §10.5 (the outstanding yield error re-run), §13 (end-to-end evaluation), §14 (limitations, including the missing user-centred evaluation), and §9.2 (the guardrail ablation that changed the deployment decision) — these are the sections that qualify a headline number or leave an action open.

| # | Team Member | Approved | Date |
|:-:|-------------|:--------:|:----:|
| 1 | Mahesh | Yes | 11 Aug 2026 |
| 2 | Harliv | Yes | 11 Aug 2026 |
| 3 | Lokesh | Yes | 11 Aug 2026 |
| 4 | Aneeqa | Yes | 11 Aug 2026 |
| 5 | Tanmay | Yes | 11 Aug 2026 |

**Document:** FarmerVision — Milestone 5 Report · **Prepared:** 11 August 2026
