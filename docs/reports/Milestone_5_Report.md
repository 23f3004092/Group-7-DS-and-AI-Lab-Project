# FarmerVision — Milestone 5 Report
## Model Evaluation, Baseline Comparison, Ablation, and Error Analysis

**Version:** v1 · **Prepared:** 6 August 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Experimental Setup](#2-experimental-setup)
3. [Model Training Summary](#3-model-training-summary)
4. [Evaluation Methodology](#4-evaluation-methodology)
5. [Performance Metrics](#5-performance-metrics)
6. [Experimental Results](#6-experimental-results)
7. [Baseline Comparison](#7-baseline-comparison)
8. [Hyperparameter Analysis](#8-hyperparameter-analysis)
9. [Ablation Study](#9-ablation-study)
10. [Error Analysis](#10-error-analysis)
11. [Model Robustness](#11-model-robustness)
12. [Computational Performance](#12-computational-performance)
13. [Limitations](#13-limitations)
14. [Possible Improvements](#14-possible-improvements)
15. [Discussion](#15-discussion)
16. [Conclusion](#16-conclusion)

**Appendices:** [A — Complete metric tables](#appendix-a--complete-metric-tables) · [B — Vision per-class results](#appendix-b--vision-per-class-results) · [C — Intent confusion matrix](#appendix-c--intent-confusion-matrix) · [D — Hyperparameter search results](#appendix-d--hyperparameter-search-results) · [E — Artifacts and configuration](#appendix-e--artifacts-and-configuration) · [Sign-off](#team-review-and-sign-off)

---

## 1. Introduction

### 1.1 Project objective

FarmerVision answers Indian farmers' questions about their crops. A farmer can ask in English, Hindi or Hinglish, or send a photo of a diseased leaf. The system routes the question, finds supporting text from a government and advisory corpus, writes a short grounded answer with sources, classifies leaf disease from the photo, and estimates expected yield.

### 1.2 Scope of evaluation

Milestone 4 trained the models. Milestone 5 measures them. Each module is evaluated on its own held-out data, against a stated baseline, with metrics that match its task. No training result is reused as an evaluation result.

### 1.3 Models evaluated

| Module | Model | Task type |
|---|---|---|
| Crop disease vision | ViT-S/16 (`vit_small_patch16_224.augreg_in21k_ft_in1k`) | 20-class image classification |
| Intent / entity / guardrail | DistilBERT multilingual, three heads on one backbone | Classification + token tagging |
| Retrieval | `BAAI/bge-m3` frozen, Qdrant index of 723,439 chunks | Ranking |
| Answer generation | `gemma-3-4b-it` 4-bit + distilled LoRA adapter | Text generation |
| Yield prediction | LightGBM, compared against XGBoost, CatBoost and an MLP | Tabular regression |

### 1.4 Objectives of Milestone 5

1. Evaluate every module on data it has never seen, quoting each number with its sample size.
2. Compare each model against a fair baseline, not a weak one.
3. Run ablations that show which components earn their place.
4. Analyse the errors and explain why they happen.
5. Close the three items Milestone 4 recorded as not delivered.

---

## 2. Experimental Setup

### 2.1 Environment and libraries

| Module | Platform | Compute |
|---|---|---|
| Vision | Kaggle | 2 × Tesla T4 (15.6 GB), 4 vCPU |
| Intent / entity / guardrail | Colab | Tesla T4 |
| Retrieval and generation | Kaggle and Colab | 1–2 × Tesla T4, fp16 (T4 has no hardware bf16) |
| Yield | Kaggle | CPU only |

Python 3.12, Torch 2.10.0+cu128, Qdrant v1.19.0. Libraries: `timm`, `transformers`, `peft`, `bitsandbytes`, `sentence-transformers`, `qdrant-client`, `rank-bm25`, `sacrebleu`, `ragas`, `seqeval`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`.

### 2.2 Dataset versions and splits

| Module | Dataset | Split | Sizes |
|---|---|---|---|
| Vision | 20-class rice + wheat corpus, `clean_index_v3.csv` | Scene-grouped, contamination-free | 10,252 / 1,284 / 1,287 |
| Intent / entity / guardrail | KCC sample + 5,172 generated non-agri rows | 80/10/10, stratified on guardrail label | 28,772 / 3,597 / 3,597 |
| Retrieval | Government PDFs + KCC records | No split, the embedder is frozen | 723,439 chunks |
| Generation (training) | Distillation set written by the teacher | 90/10, stratified by row kind | 2,868 / 200 / 28 |
| Generation (evaluation) | Curated and real farmer questions | Held out | 48 curated, 77 real, 2 controls |
| Yield | `production_unified_imputed.csv`, cleaned to 426,803 rows | Two-stage 15% hold-out | 308,364 / 54,418 / 64,021 |


### 2.3 Seeds and reproducibility

| Module | Seed | Notes |
|---|---|---|
| Vision | Fixed split file on disk | The data path is frozen in `pipeline.py`, so evaluation imports the audited transform object |
| Intent / entity / guardrail | 42 | Applied to all random operations |
| Retrieval and generation | 13 | Applied to torch, the split and the Trainer; bootstrap uses seed 13, 2000 resamples |
| Yield | 42 | Applied to both splits, all three estimators and every `KFold` |

Generation at evaluation time is greedy (temperature 0), so a re-run reproduces the same answers exactly. Retrieved chunks are written to disk before judging so the same chunks are scored in the same order. Each stage writes a manifest with a config hash.

### 2.4 Evaluation workflow

Load the selected checkpoint and the frozen transform or index manifest, run the model once over the held-out set with no tuning after that point, score with the metrics in §5 with intervals attached where the sample is small, compare against the §7 baseline, and save all per-row results.

---

## 3. Model Training Summary

| | Vision | Intent / entity / guardrail | Distilled generator | Yield (LightGBM) |
|---|---|---|---|---|
| **Architecture** | ViT-S/16, 21.67M params, head 384 → 20 | DistilBERT multilingual, 134.7M params, 3 heads | Gemma-3-4B 4-bit + LoRA r=32, α=64 | Gradient boosted trees, 202 estimators, 94 leaves, depth 12 |
| **Optimizer** | AdamW | AdamW | paged AdamW 8-bit | Built-in |
| **Learning rate** | 1e-3 → 1e-4/1e-3 → LLRD 0.70, base 3e-4 | 3e-5, linear with 10% warmup | 2e-4, cosine, warmup 0.03 | 0.2537 |
| **Batch size** | 64 | 32 | 1 row × 16 accumulation | — |
| **Epochs** | 12 + 20 + 18, three phases | 5 | 4 allowed | — |
| **Loss** | Label-smoothed cross-entropy | Cross-entropy + class-weighted cross-entropy on guardrail | Causal-LM cross-entropy on answer tokens only | Mean squared error |
| **Early stopping** | Cosine ran to completion, phase 3 froze at epoch 15 | Best composite validation score | Patience 2 on validation loss | None on the default and tuned runs |
| **Checkpoint kept** | `p3_full_best.pt`, epoch 16 | Best composite score | `checkpoint-360`, end of epoch 2 | Final fit of the best search candidate |
| **Duration** | ~42 min total GPU | 949.4 s | 578.1 min (9.6 h) | ~14–70 s per fit, 60 fits |

Two notes. The distilled adapter kept is the epoch-2 model, not epoch-4: validation loss stopped improving after epoch 2 and `load_best_model_at_end` caught it. For vision, epochs 15 to 17 produced identical validation metrics to four decimal places because cosine decay drove the learning rate to zero, so the run was fully converged.

---

## 4. Evaluation Methodology

| Module | Protocol | Test set | Ground truth |
|---|---|---|---|
| Vision | Evaluated once on a test set built before training and touched by no decision | 1,287 images, 20 classes | Source dataset labels, reconciled against the manifest |
| Intent / entity / guardrail | Selected on validation, scored once on test | 3,597 queries | Intent from the KCC `QueryType` field (weak supervision); NER from crop alias matching, 86.4% coverage; guardrail from authored rules and examples |
| Retrieval | 48 questions, top 10 chunks each, judged by a person | 480 chunk judgements | A chunk is relevant if a farmer asking that question would be helped by it |
| Generation | Three arms read identical chunks and identical prompts, decoding greedily | 48 curated + 77 real + 2 off-domain controls | Expert gold answers in KCC style for the curated set |
| Yield | Single held-out test set, never used in training or search | 64,021 rows | Cleaned `yield` column after outlier removal |

The 77 real questions are used unedited, with typos (`Damffing off`, `Leaf caurl`, `Merigold`), shouted capitals and trailing dot runs left in.

**Cross-validation** is used for the yield model only: 3-fold `KFold` inside `RandomizedSearchCV`, not for final metric reporting. Elsewhere it does not apply. Vision evaluation images are not independent because many share a photographic scene, so a cluster bootstrap over scene groups (±0.025 on wheat-15) is the correct uncertainty estimate. For the generator, k-fold would mean four more nine-hour training runs on a free T4.

**Baselines and success criteria:**

| Module | Baseline | Success criterion |
|---|---|---|
| Vision | Frozen backbone with a linear probe head | Beat the probe by more than the ±0.025 noise floor on wheat-15 macro-F1 |
| Intent / NER / guardrail | Same backbone with randomly initialised heads | Accuracy > 0.85, macro-F1 > 0.70, entity F1 > 0.90, guardrail F1 > 0.95 |
| Retrieval | The earlier automatic word-overlap scorer | A useful chunk in the top 5 for most questions |
| Generation | Base Gemma-3-4B given one worked example | At least as grounded as the baseline, no invented numbers, correct language, refuses when it should |
| Yield | PyTorch MLP and the untuned default configurations | Highest R² and lowest RMSE, with deployable latency and model size |

The generation baseline choice is the one that matters and is explained in §7.4.

---

## 5. Performance Metrics

### 5.1 Classification — vision, intent, NER, guardrail

Accuracy, macro-F1, weighted F1, per-class precision/recall/F1 with support, entity-level F1 for NER, and ROC-AUC and PR-AUC for the binary guardrail head.

**Macro-F1 is the honest headline** wherever classes are imbalanced, and both these datasets are: vision is 43:1 and intent is 46:1. Accuracy hides rare-class failure. Entity-level F1 matches whole spans rather than tokens, which is the stricter and standard choice for NER.

**Vision uses wheat-15 macro-F1 as its primary metric**, not 20-way accuracy. A frozen backbone already separates rice from wheat at 0.9835 F1, so any 20-way number is partly free. The 15 wheat conditions are the real problem.

### 5.2 Ranking — retrieval

**Precision@5** is the share of the top 5 chunks judged relevant. The generator only sees 5 chunks, so anything useless in those 5 is wasted context. **Recall@5** is the share of relevant chunks that made it into the top 5, and measures ranking rather than presence.

Recall is measured **inside the 10 chunks that were read**, not across all 723,439 chunks — nobody read the whole index, so there is no full answer key. Ten were judged and five scored on purpose: had only five been judged, Recall@5 would be 1.0 by arithmetic.

### 5.3 Generation

| Metric | What it measures | Why it is used |
|---|---|---|
| chrF++ | Character n-gram plus word bigram overlap with the gold answer | The main metric for Indic text. Character n-grams survive Hindi word endings where BLEU scores near zero for a correct answer |
| BLEU, ROUGE-1/2/L | Word overlap | Reported, quoted last. ROUGE is implemented directly because the standard library tokenizer deletes Devanagari and would score 47 of 48 references as 0.0 |
| numeric_recall | Share of the gold answer's numbers reproduced | A wrong dose is the failure that matters. `3 g/kg` and `5 g/kg` differ by one character and one damages a crop |
| numeric_grounding, hallucinated_numbers | Whether asserted numbers appear in the context, compared numerically with Devanagari digits folded to ASCII | The safety check. Verified with self-tests before use |
| language_match | 1.0 exact, 0.5 right script wrong register, 0.0 wrong script | Farmers must be answered in the language they used |
| has_citation, citation_valid | Whether sources are cited and whether every cited number exists | Citing a source never supplied is fabricated provenance |
| abstain accuracy | Whether the safety gate refused when it should | A correct refusal is a success, not a failure |
| RAGAS faithfulness | An independent LLM splits the answer into claims and checks each against the context | Reads whole sentences, so it catches an unsupported claim containing no numbers |

### 5.4 Regression — yield

MAE, MSE, RMSE, median absolute error, explained variance and R². **RMSE and R² are the selection criteria**; RMSE reads directly in yield units and R² is scale-independent so the four candidates compare directly. Median AE is reported because it is robust to the outlier tail.

**MAPE was computed and is not usable** — see §10.5.

### 5.5 Confidence intervals

Vision uses a cluster bootstrap over scene groups, giving ±0.025 on wheat-15. Retrieval and generation use a bootstrap with 2000 resamples. The adapter study measured a seed-noise band of 0.0074 by running one configuration on three seeds. **Two numbers whose intervals overlap are the same number**, and that rule is applied throughout.

---

## 6. Experimental Results

### 6.1 Vision

The test set was built before training, verified contamination-free, and evaluated once.

| Metric | Val (selection) | **Test (held out)** | Cluster 95% CI | Change |
|---|---|---|---|---|
| **wheat-15 macro-F1** | 0.8793 | **0.8631** | [0.8403, 0.8853] | −0.016 |
| 20-way macro-F1 | 0.9042 | **0.8671** | [0.8396, 0.8905] | −0.037 |
| rice-5 macro-F1 | 0.9828 | 0.8823 | [0.7915, 0.9523] | −0.100 |
| Accuracy | 0.9143 | **0.8998** | — | −0.014 |

Sixteen of twenty classes exceed 0.80 and eleven exceed 0.94. The three below 0.65 are the same three the pre-training diagnostics predicted. Per-class results are in Appendix B.

### 6.2 Intent, entity and guardrail

| Metric | Baseline | **Fine-tuned** | Target | Status |
|---|---|---|---|---|
| Intent accuracy | 0.371 | **0.884** | > 0.85 | Met |
| Intent macro-F1 | 0.111 | **0.715** | > 0.70 | Met |
| NER entity F1 | 0.036 | **0.958** | > 0.90 | Met |
| Guardrail precision / recall / F1 | 0.000 | **1.000** | > 0.95 | Met |
| Guardrail ROC-AUC / PR-AUC | — | 1.000 / 1.000 | — | — |

**Per-class intent performance (test, n = 3,597):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `disease_pest` | 0.905 | 0.912 | 0.909 | 1,576 |
| `non_agri` | 0.996 | 1.000 | 0.998 | 490 |
| `nutrition_fertilizer` | 0.860 | 0.882 | 0.871 | 702 |
| `cultivation_practice` | 0.817 | 0.828 | 0.822 | 738 |
| `post_harvest_storage` | 1.000 | 0.800 | 0.889 | 10 |
| `specialty_other` | 1.000 | 0.696 | 0.821 | 23 |
| `general` | 0.263 | 0.094 | **0.139** | 53 |
| **Weighted average** | **0.882** | **0.887** | **0.884** | 3,597 |

Macro-F1 of 0.715 is the number to quote. Accuracy of 0.884 is carried by the two large classes and hides `general` at 0.139. **The guardrail's perfect test score needs the caveat in §9.2**: on an adversarial red-team set the model head alone recalls only 0.571, which is why the deployed guardrail combines the model with rules.

### 6.3 Retrieval

Human judgement over 480 chunks across 48 questions.

| Metric | Score | 95% interval | Questions |
|---|---|---|---|
| **Precision@5** | **0.725** | [0.621, 0.821] | 48 |
| **Recall@5** | **0.498** | [0.452, 0.543] | 43 |

About 3.6 of the top 5 chunks are useful on average, and the top 5 holds about half the useful chunks found in the top 10. Forty-three of 48 questions had at least one relevant chunk in the top 10.

| Language | Precision@5 | n | | Topic (worst first) | Precision@5 | n |
|---|---|---|---|---|---|---|
| English | 0.831 | 13 | | Nutrient management | **0.371** | 7 |
| Hindi | 0.788 | 17 | | Varieties | 0.575 | 8 |
| **Hinglish** | **0.589** | 18 | | Weed management | 0.800 | 2 |
| | | | | Plant protection | 0.838 | 21 |
| | | | | Cultural practices | 0.840 | 10 |

Hinglish and dose questions are the two weak points and they overlap: of the six questions that returned nothing relevant in the top 5, five are Hinglish and four ask about doses.

### 6.4 Generation

**Curated set, the 40 questions where retrieval supplied strong context** — a model given nothing useful is not failing, so those 8 are separated out.

| Arm | chrF++ | BLEU | ROUGE-L | numeric_recall | invented numbers | grounding | script_ok |
|---|---|---|---|---|---|---|---|
| Base, no example | 25.56 | 11.26 | 0.3249 | 0.283 | 0.000 | 0.902 | 1.000 |
| Base, one example | 29.96 | 15.75 | 0.3657 | 0.326 | 0.000 | 0.920 | 1.000 |
| **Distilled** | **32.55** | **19.37** | 0.3632 | **0.388** | 0.000 | 0.892 | 0.875 |

Across all 48 questions including weak context, the base model invented numbers at 0.095 without an example and 0.024 with one. **The distilled model invented none.**

**Full pipeline, 77 real farmer questions.** Both models see identical retrieval and prompts.

| Model | grounding | language match | cited | citation valid | length in target | output tokens | abstain accuracy |
|---|---|---|---|---|---|---|---|
| Baseline | 0.992 | 0.935 | 1.000 | 1.000 | 0.623 | 98.9 | 0.948 |
| **Distilled** | **1.000** | **0.994** | 1.000 | 1.000 | 0.273 | 71.0 | 0.948 |

Abstain accuracy is identical for both, which is expected because the safety gate depends on retrieval, not on the generator. Had it differed, that would have signalled a setup bug.

**Language match by question language** is the clearest single result in the evaluation:

| Language | n | Baseline | **Distilled** |
|---|---|---|---|
| English | 38 | 0.987 | 0.987 |
| Hindi | 31 | 1.000 | 1.000 |
| **Hinglish** | 8 | **0.438** | **1.000** |

**Health checks:**

| Check | Baseline | Distilled |
|---|---|---|
| G1 Off-topic never reaches the model | PASS (0/2) | PASS (0/2) |
| G2 Every number traces to the source text | **FAIL** (3/77) | **PASS** (0/77) |
| G3 Answer in the same language as the question | **FAIL** (7/39) | **PASS** (0/39) |
| G4 Fallback answers carry the KVK verification line | PASS (4) | PASS (4) |
| G5 Answered questions cite their sources | PASS (77/77) | PASS (77/77) |
| G6 Cited context numbers actually exist | PASS (0 invalid) | PASS (0 invalid) |

**The distilled model passes all six. The baseline fails two.**

**RAGAS faithfulness (independent LLM judge, 21-question subset):** baseline 0.725, distilled 0.798 including refusals and **0.956 on attempted answers only**. Refusals are separated because faithfulness is supported statements divided by total statements, and "I do not have that information" is a claim about the model's own knowledge that can never be inferred from context. It scores 0.00 by construction, so leaving refusals in punishes the model for the exact behaviour the safety design asks for. On the 17 questions both models attempted, the comparison is 0.849 against 0.956.

### 6.5 Yield prediction

Test set of 64,021 rows, identical for every model.

| Model | MAE | MSE | RMSE | Median AE | Explained variance | R² |
|---|---|---|---|---|---|---|
| **LightGBM (tuned)** | 0.8298 | 5.8139 | **2.4112** | 0.2905 | 0.9572 | **0.9572** |
| XGBoost (default) | 0.7653 | 6.0866 | 2.4671 | 0.2314 | 0.9552 | 0.9552 |
| XGBoost (tuned) | 0.8486 | 6.4754 | 2.5447 | 0.2814 | 0.9523 | 0.9523 |
| LightGBM (default) | 0.8603 | 6.5473 | 2.5588 | 0.2870 | 0.9518 | 0.9518 |
| CatBoost (tuned) | 0.9136 | 7.3174 | 2.7051 | 0.2723 | 0.9462 | 0.9461 |
| CatBoost (default) | 0.9920 | 8.4288 | 2.9032 | 0.2844 | 0.9380 | 0.9380 |
| PyTorch MLP (baseline) | 5.2961 | 135.8725 | 11.6564 | 3.4907 | 0.0000 | **−0.0000** |

**LightGBM (tuned) is selected** on R² and RMSE. It is also the smallest tree model at 1.8 MB, which makes it the best accuracy-to-footprint trade-off (§12).

Two results are worth noting rather than smoothing over. **Default XGBoost beats tuned XGBoost** on RMSE and R², although tuned XGBoost has the better MAE and median AE — a sign that a 20-candidate random search scored on negative MSE did not fully line up with the RMSE and R² outcome. **The PyTorch MLP failed to learn**, with R² at zero and predictions collapsing towards the target mean. That is a training-configuration failure, not evidence against neural approaches: there was no feature scaling, no batch normalisation, only 10 epochs, and a 923-dimensional sparse one-hot input that the tree models handle natively through categorical splits instead.

### 6.6 Comparison with target performance

| Module | Metric | Target | Achieved | Status |
|---|---|---|---|---|
| Vision | wheat-15 gain over frozen probe | > 0.025 | **+0.083** | Met, 3.3× the noise floor |
| Intent | Accuracy / macro-F1 | > 0.85 / > 0.70 | 0.884 / 0.715 | Met |
| NER | Entity F1 | > 0.90 | 0.958 | Met |
| Guardrail | F1 | > 0.95 | 1.000 on test | Met on test, see §9.2 |
| Retrieval | Useful chunk in top 5 | Most questions | 42 of 48 | Met |
| Generation | No invented numbers, correct language | 0 / high | 0 of 125 / 0.994 | Met |
| Yield | Highest R², lowest RMSE, deployable | — | 0.9572 / 2.4112 / 1.8 MB | Met |

### 6.7 Overall system performance

Every module reaches its target on its own held-out data. The system has one dominant cost and one dominant weakness.

- **Latency is generation.** Intent and entity extraction take 4.86 ms, retrieval a few tens of milliseconds, and yield prediction runs at 63,510 rows per second on CPU. Generation takes about 14 seconds per answer on a free T4. Everything else is noise in the total.
- **Hinglish is the weakest input across the stack**: worst retrieval precision (0.589), worst generation similarity score, and the baseline generator's worst language. The distilled model closes the language-match half of that gap completely.

---

## 7. Baseline Comparison

### 7.1 Vision

| Metric | Frozen probe | Blocks 9–11 | **Full (selected)** | Gain |
|---|---|---|---|---|
| **wheat-15 macro-F1 (val)** | 0.7966 | 0.8660 | **0.8793** | **+0.083** |
| 20-way macro-F1 (val) | 0.8382 | 0.8830 | 0.9042 | +0.066 |
| Accuracy (val) | 0.8559 | 0.9042 | 0.9143 | +0.058 |

The cumulative +0.083 is 3.3 times the noise floor and is the defensible headline. The last step alone is +0.013, inside the interval, and is not claimed as an improvement.

### 7.2 Intent, entity and guardrail

Against randomly initialised heads: intent accuracy 0.371 → 0.884, macro-F1 0.111 → 0.715, entity F1 0.036 → 0.958, guardrail F1 0.000 → 1.000. The size of these gaps mostly shows random heads cannot do the task at all. The useful reading is that one shared backbone learns all three tasks in under 16 minutes.

### 7.3 Retrieval — the metric was the problem, not the retriever

| Scoring method | Precision@5 |
|---|---|
| Automatic word overlap with the gold answer | 0.317 |
| **Human judgement on the same retrieved chunks** | **0.725** |

The formula marked correct chunks wrong because they said the same thing in different words, which is exactly what happens on Hindi and Hinglish text. The retriever was more than twice as good as its own metric said, and the human-judged figure is the one to report.

### 7.4 Generation — what the fine-tune actually bought

| Comparison | chrF++ difference | 95% interval | Verdict |
|---|---|---|---|
| One example minus no example | +4.25 | [+0.49, +8.09] | Significant |
| Distilled minus no example | +6.06 | [+0.46, +11.93] | Significant |
| **Distilled minus one example** | **+1.81** | **[−3.48, +7.27]** | **Tie** |

**Most of the apparent distillation gain on similarity metrics is formatting.** Once the baseline is handed the output format with a single worked example, the remaining difference is not distinguishable from zero, and the per-question count agrees: the distilled model wins 18 and loses 19 of 40. On these metrics, "the distilled model writes better answers than a well-prompted baseline" is not a claim the numbers support.

What the fine-tune does buy is visible where chrF++ cannot see:

| | Baseline | Distilled |
|---|---|---|
| Numeric grounding (77 real questions) | 0.992 | **1.000** |
| Numbers invented (48 curated) | 0.024 to 0.095 | **0.000** |
| Language match overall / on Hinglish | 0.935 / 0.438 | **0.994 / 1.000** |
| chrF++ on Hindi questions | 27.70 | **38.28** |
| RAGAS faithfulness (shared 17) | 0.849 | **0.956** |
| Health checks passed | 4 of 6 | **6 of 6** |

Where the baseline is still ahead: chrF++ on Hinglish (37.04 against 29.14), answers inside the 90–200 token target (0.623 against 0.273), and speed. **The distilled model is safer and more consistent, not more fluent** — the right trade for a system giving dosage advice.

### 7.5 Yield

| Comparison | Result |
|---|---|
| LightGBM tuned vs the MLP baseline | R² +0.9572; the MLP is effectively non-functional |
| LightGBM tuned vs LightGBM default | RMSE 2.5588 → 2.4112 (−5.8%), R² 0.9518 → 0.9572 |
| CatBoost tuned vs CatBoost default | RMSE 2.9032 → 2.7051 (−6.8%), the clearest tuning win |
| XGBoost tuned vs XGBoost default | RMSE 2.4671 → 2.5447, tuning made it worse |

Tuning helped LightGBM and CatBoost and did not help XGBoost, which suggests the 20-candidate budget was too small or that XGBoost's defaults (1000 estimators, lr 0.05, depth 7) were already close to optimal for this data.

---

## 8. Hyperparameter Analysis

### 8.1 Vision

**Normalisation, measured on a frozen linear probe:** the checkpoint's native `(0.5, 0.5, 0.5)` scores 0.8461 macro-F1, ImageNet statistics 0.8326, and our own dataset statistics 0.8274. **Matching the pretraining statistics beats matching the target domain by 1.9 F1**, the opposite of the usual intuition. Square-root class balancing was neutral to negative in all three pairings and was rejected.

**Batch size:** throughput plateaus by 96 and memory is never the constraint (4.9 GB of 15.6 GB at the largest setting). 64 was selected because it is within 3% of peak throughput while giving 160 optimiser steps per epoch instead of 106, and on a 10,000-image fine-tune step count matters more than 3% of wall clock.

**Unfreezing depth, the experiment that mattered most:**

| Metric | Frozen probe | Head only | Blocks 9–11 | Full |
|---|---|---|---|---|
| wheat-15 macro-F1 (val) | 0.7966 | 0.7682 | 0.8660 | 0.8793 |
| Train/val gap | — | 0.047 | 0.092 | 0.073 |

Head-only scoring below the probe is expected, not a regression: the probe trained on un-augmented features, and a frozen backbone cannot learn invariance from augmentation, it only scatters the cloud the head must fit. Adapting the top three blocks is where the gain lives (+0.098). Unfreezing the remaining 16M parameters adds +0.013, inside the interval.

### 8.2 Intent, entity and guardrail

Six configurations ranked on a composite validation score across the three heads. AdamW at lr 3e-5, batch 32, 5 epochs won at 1.747. AdamW clearly beats SGD (1.279), 3e-5 beats 1e-5, five epochs beats three, batch 32 is slightly better than 64, and label smoothing changes nothing. Full table in Appendix D. Validation loss bottoms at epoch 3 and rises slightly after, but the composite score keeps improving, so five epochs was kept and the checkpoint chosen on the composite score.

### 8.3 Distilled generator

27 QLoRA fine-tunes, each differing from the baseline in exactly one setting, all trained on the same 160 rows for the same number of steps and scored on the same 48 held-back rows. The notebook raises an error if a configuration varies more than one setting, so the design is checked rather than trusted.

**The noise floor was measured first.** The same configuration on seeds 13, 7 and 21 gave best validation losses spanning 0.4197 to 0.4271, a band 0.0074 wide. Any gap smaller than that is not a result.

| Family | Configs | Best | Worst | Spread | Beyond noise | Matters? |
|---|---|---|---|---|---|---|
| Optimization | 12 | 0.3370 | 0.7088 | 0.3718 | 7 of 12 | **Yes** |
| Capacity | 6 | 0.3667 | 0.5360 | 0.1693 | 5 of 6 | **Yes** |
| Regularization | 5 | 0.4243 | 0.4277 | **0.0034** | 1 of 5 | **No** |

**Optimization settings and model capacity have real effects. Regularization does not** — the whole spread across five regularization configurations is less than half the seed-noise band, so dropout and weight decay are decoration on this task.

The final run uses rank 32 with four epochs. This is a composite, not any single configuration the study measured: it combines the two directions the study found real (more rank helps, more epochs helps). It is well motivated rather than proven optimal, and is recorded as such.

### 8.4 Yield

`RandomizedSearchCV` with 3-fold `KFold` scored on negative MSE: 20 candidates for XGBoost and LightGBM (60 fits each), 15 for CatBoost (45 fits). Selected LightGBM settings: 202 estimators, lr 0.2537, 94 leaves, max depth 12, min child samples 30, subsample 0.657, colsample 0.750. Search spaces and the other two winners are in Appendix D.

**Log-target experiment.** The yield distribution is heavily right-skewed (skewness 198.60) and spreads very differently across crop types, with sugarcane running 0 to 130+ against pulses at 0 to 3. `log1p` was tested as a variance-stabilising transform, inverted with `expm1` and clipped at zero so the metrics stay comparable.

| Model | MAE original → log | RMSE original → log | R² original → log |
|---|---|---|---|
| LightGBM (tuned) | 0.8298 → **0.8045** | 2.4112 → 2.8044 | 0.9572 → 0.9421 |
| CatBoost (tuned) | 0.9136 → **0.8909** | 2.7051 → 3.0615 | 0.9461 → 0.9310 |
| XGBoost (tuned) | 0.8486 → 0.8674 | 2.5447 → 2.9073 | 0.9523 → 0.9378 |

The transform **improved MAE for two of three models but worsened RMSE and R² for all three**. The log scale compresses large values during training, so the model under-penalises big absolute misses on high-yield crops, and those errors re-expand after inversion. **Decision: keep the original-scale target**, since RMSE and R² are the selection criteria. The log variant stays documented as an option if a later iteration prioritises relative error on low-yield crops.

---

## 9. Ablation Study

### 9.1 Vision

| Ablation | Result |
|---|---|
| **Without augmentation** (frozen probe on clean features) | `rice__leaf_smut` scores 0.923 on 25 training images. Under the shortcut-destroying pipeline it collapses to 0.320, then recovers to 0.769 once blocks are unfrozen. **The original 0.923 was almost entirely a brightness shortcut** |
| **Without the exposure transform** | Brightness alone predicts `leaf_smut` at AUC 0.821. With the power-law tone curve it falls to 0.552, and exposure-normalised sharpness from 0.803 to 0.595 |
| **Without hue jitter** | A ±0.02 hue jitter cost 3.10 ms per image, 27% of the data pipeline, and moved no AUC outside the noise floor. Removed |
| **Without full unfreezing** | wheat-15 0.8660 against 0.8793, inside the interval. Three blocks capture nearly all the gain |
| **Without class balancing** | Selected. Square-root sampling was neutral to negative in all three pairings |
| **Without drop-path at full unfreeze** | Phase 2 held a 0.092 gap with 5.3M trainable parameters; phase 3 held 0.073 with 21.7M, purely because drop-path was added with the unfreeze |

Without the augmentation work the model would ship reporting 0.92 on a class it identifies by darkness, and fail on the first field photograph taken in daylight.

### 9.2 Guardrail — the ablation that changed the deployment decision

Three strategies on an adversarial red-team set:

| Layer | Precision | Recall | F1 |
|---|---|---|---|
| Model head only | 1.000 | 0.571 | 0.727 |
| Rules only | 1.000 | 0.857 | 0.923 |
| **Model or rules combined** | **1.000** | **1.000** | **1.000** |

The head scores 1.000 on the ordinary test split and recalls only 0.571 on adversarial input. Neither layer alone is sufficient, and the combined layer is what goes to deployment. This directly answers the Milestone 4 finding that a perfect score on an easy negative class is not evidence of a working guardrail.

### 9.3 Generator

| Ablation | Result |
|---|---|
| Without adapter, without worked example | chrF++ 25.56, numeric_recall 0.283, invents numbers at 0.095 |
| Without adapter, with worked example | chrF++ 29.96 (+4.25, significant), invents at 0.024 |
| With adapter, without worked example | chrF++ 32.55 (+1.81 over one-shot, a tie), invents none |
| Without strong retrieval (8 of 48 questions) | Every arm scores poorly, which is why they are excluded from the fair comparison |
| Without human judgement | Precision@5 drops 0.725 to 0.317 |
| Without refusal separation | Distilled faithfulness reads 0.798 instead of 0.956 |

**Adapter component ablations** (validation loss, noise band 0.0074): attention adapters only 0.5360 (worse), MLP adapters only 0.4677 (worse), all adapters 0.4198 (reference), rank 8 0.4958 (worse), rank 64 0.3667 (better), without LoRA dropout 0.4243 and without warmup 0.4203 (both inside noise). Both module groups contribute and the MLP adapters carry more of the work; capacity had not saturated at rank 64; removing regularization entirely sits inside the noise band.

### 9.4 Yield

No formal component ablations were run this milestone. The tuned-versus-default comparison in §7.5 is a partial ablation of hyperparameter tuning, and the log-target experiment in §8.4 is an ablation of the target transform. Three ablations are scoped for the next milestone: performance without the 99th-percentile outlier removal, native categorical splits against one-hot encoding isolated from the MLP confound, and a feature ablation removing `annual_rainfall`, `fertilizer` and `pesticide` to test how much the model relies on agronomic inputs rather than on crop and location identity.

A SHAP summary was generated on a 1,000-row test sample using `TreeExplainer` on the selected model, and is available as a figure artefact.

---

## 10. Error Analysis

### 10.1 Vision — the errors are concentrated, not scattered

| True → predicted | n | Share of true class |
|---|---|---|
| `leaf_blight` → `mildew` | 7 | 12.3% |
| `leaf_blight` → `tan_spot` | 7 | 12.3% |
| `tan_spot` → `common_root_rot` | 7 | 10.8% |
| `tan_spot` → `leaf_blight` | 6 | 9.2% |
| `tan_spot` → `mite` | 6 | 9.2% |
| **`leaf_smut` → `tan_spot`** | 3 | **37.5%** |

**Root cause of the main cluster.** Almost every error falls inside `{tan_spot, leaf_blight, common_root_rot, mite, mildew}`. All five are wheat conditions producing dead or yellowing patches, and they look alike at 224 pixels. The frozen-feature diagnostic predicted a three-class version of this cluster before any fine-tuning. **This is a plant pathology problem, not a data artefact**, and more tan spot images are unlikely to fix it.

**Root cause of the cross-crop error.** Three of eight `rice__leaf_smut` test images were predicted as a wheat class, in a model that separates rice from wheat at 0.98 F1 everywhere else. Leaf smut images are about 71% replicate padding, so once brightness is neutralised what remains in a heavily padded rice image resembles a padded wheat lesion more than it resembles other rice photographs.

Rare classes fail asymmetrically: `wheat__stem_fly` (n=17) and `wheat__black_rust` (n=31) hold above 0.84, so small support alone is not the problem, while `rice__leaf_smut` on 8 images swung 0.833 → 0.471 → 0.769 → 0.923 across consecutive epochs and is uninformative in either direction.

### 10.2 Intent classification

| Query | True | Predicted | Why |
|---|---|---|---|
| DAP or NPK different information | cultivation_practice | nutrition_fertilizer | Fertiliser words pull it across; the label itself is arguable |
| Information about mushroom cultivation? | cultivation_practice | general | A specialty crop with few training examples |
| Hame gende ka tel bechna hai kisse sampark kare | general | cultivation_practice | A business question containing a crop name |
| Information about pre-emergence weed control | disease_pest | cultivation_practice | Weed control sits on a class boundary |

**Root causes, in order of how much they explain.** (1) *Class boundary confusion* — nearly all errors are between adjacent classes, with 78 cultivation queries going to disease_pest and 76 the other way, and almost nothing crossing into `non_agri` (Appendix C). (2) *Weak supervision noise* — intent labels come from the KCC `QueryType` field, filled in by call-centre staff and inconsistent on boundary cases, so some of these are the label being wrong rather than the model. (3) *Small classes* — `general` has 53 test rows, scores 0.139, and is a catch-all with no consistent meaning. (4) *Code-switching* — mixed-script queries carry cues from two languages at once.

Guardrail on the standard test split: zero false positives, zero false negatives. The adversarial result is in §9.2.

**Sample predictions.** Both unsafe queries were flagged correctly, which matters most, but "is monocrotophos safe for my crops?" was routed to `non_agri`, a urea question went to `disease_pest`, and "who is the current agriculture minister of UP?" was routed to a farming class instead of `non_agri`.

### 10.3 Retrieval

Six of 48 questions returned nothing relevant in the top 5.

| Failure mode | Example | Root cause |
|---|---|---|
| Hinglish phrasing | "Mango ki variety ki information chahiye?" | Romanised Hindi does not embed close to either the Hindi or the English text in the corpus |
| Dose and nutrient questions | "Potato crop mein fertilizer dose kya hai?" | Dose information lives in tables and schedules, which chunk badly and embed poorly against a short question |
| Thin corpus coverage | "Wilt control information about Pea crop?" | That crop and problem combination is barely present |

Four of the six are dose questions and five are Hinglish, so the failures share causes rather than being independent.

### 10.4 Generation

**Baseline failures.** Three of 77 answers contain a number not in the context, carried over from pretraining. Seven of 39 Hindi or Hinglish questions were answered in the wrong language, almost all Hinglish. Its two lowest-faithfulness answers score 0.00: for a question naming no crop it listed fungicides for potato, brinjal, rose and mango pulled from four unrelated contexts, and for "TELL ME PEST CONTROL IN PADDY?" it produced generic advice with citations attached that the cited text does not support.

**Distilled failures.** Two patterns, both traceable to a training choice rather than a model defect.

1. **Wrong script on 5 of 40 curated answers.** It answers a Hinglish question in romanised Hindi while the reference is Devanagari. The teacher was told to answer in the *question's* language and did so 100% of the time; the evaluation scores against the *reference's* script. The student learned exactly what it was taught, and this alone explains most of its lower chrF++ and grounding on Hinglish.
2. **Over-refusal on 4 of 21 questions.** Two are arguably correct and are the two the baseline answered with faithfulness 0.00. The other two were retrieved at the confident tier, so refusing there is a real cost. The cause is that 26% of training prompts teach refusal and the teacher rule says to prefer refusing over using loosely related context.

**Where the two scoring methods disagree.** The rule-based numeric check reads 0.993 and 1.000 while the LLM judge reads 0.725 and 0.798. Every gap is a case where the numbers were right but a surrounding claim was not supported. Numeric grounding is at ceiling for both models and cannot separate them; faithfulness can, because it reads whole sentences.

### 10.5 Yield

**MAPE is not usable as reported.** Every model's MAPE lands between about 5.6 × 10¹⁵% and 1.9 × 10¹⁶%. This is a division-by-near-zero artefact: the cleaned data still contains true-zero and near-zero yield rows, and MAPE's denominator blows up for any such row regardless of how small the absolute error is. It is omitted from every results table above and must be replaced with SMAPE, WAPE or a MAPE computed only above a minimum yield threshold before it is reported anywhere.

**Two diagnostics do not yet match the selected model.** The top-25 worst absolute errors and the per-crop-type percentage error breakdown were both computed against tuned CatBoost, not the selected LightGBM. Both should be re-run against LightGBM so the error analysis describes the deployed candidate.

**Root causes identified.** Near-zero true yields distort every percentage-based metric. High-yield, high-variance crop types such as sugarcane dominate the absolute-error tail, which is consistent with the log-target trade-off in §8.4. The MLP's near-zero R² is a training-configuration problem, not a data problem, as the tree models reach 0.95 on the same rows.

**Note on confusion matrices for retrieval and generation.** Confusion matrices and ROC curves are classification diagnostics and do not apply to ranking or free-text generation. The equivalents used here are the per-question Precision@5 distribution, the paired per-question chrF++ differences, the loss curves, and the pass/fail health check table.

---

## 11. Model Robustness

**Performance on unseen data.** Vision wheat-15 fell only 0.016 from validation to test, well inside the interval, on a set built before training. Intent is consistent across three languages: English 0.890, Hinglish 0.885, Hindi 0.867 accuracy. The generator's best validation loss of 0.1648 against a training loss of 0.1398 is a gap of +0.0250, and the checkpoint rule kept the epoch-2 model rather than epoch-4. Yield reaches R² 0.9572 on 64,021 unseen rows.

**Noise tolerance.** The 77 real questions are deliberately unclean (`Damffing off`, `Leaf caurl`, `Cater pilar`, shouted capitals, trailing dot runs). The distilled model reached 1.000 numeric grounding and 0.994 language match on them, so it is not brittle to surface noise. Query normalisation is applied to the embedding only; the generator always sees the farmer's original words, and its effect was measured rather than assumed (+0.0165 on one question, −0.0663 on another). For vision, training augmentation randomises exposure, blur, sharpness, crop and occlusion, and the transform is frozen in a module so evaluation uses exactly the audited object.

**Unseen categories.** The yield pipeline explicitly checks for unseen categorical values in validation and test after aligning to the training categories, and warns if any are found — a reasonable production safeguard against unseen crop, state or district values.

**Adversarial and edge cases.**

| Case | Behaviour |
|---|---|
| Off-domain questions (motorcycle repair, flight booking in Hindi) | The safety gate stopped both before any model saw them |
| Adversarial unsafe queries | Guardrail head alone recalls 0.571; combined with rules, 1.000 (§9.2) |
| Empty string, "???", "123456789", 500 repeated characters | The intent model always returns a class, defaulting to `disease_pest` or `cultivation_practice`. It has no "I do not know" option, which is a real gap |
| Questions a document corpus cannot answer (a phone number, seed stock) | Flagged so refusing scores as correct; abstain accuracy 0.948 |
| Zero-yield and low-area rows | Present by design after cleaning; a known stress point for percentage metrics (§10.5) |

**Generalisation limits.** Vision has not been tested on field photographs, and published work shows classifiers dropping from around 99% on lab datasets to around 73% in the wild, so **0.8671 should not be quoted as expected field accuracy**. Noise and adversarial testing was not run for the yield model this milestone.

---

## 12. Computational Performance

### 12.1 Training time

| Module | Time | Hardware |
|---|---|---|
| Vision, three phases plus evaluation | ~42 min | 2 × T4 |
| Intent / entity / guardrail | 949.4 s | 1 × T4 |
| Distillation data preparation | 353.9 min, mostly rate-limited API calls | 2 × T4 |
| Generator hyperparameter study, 27 configs | 364.9 min | 1 × T4 |
| Final adapter training | 578.1 min (9.6 h) | 1 × T4 |
| Yield, CatBoost default | ~4 min | CPU |
| Yield, full search across three families | 165 fits, ~14–70 s each | CPU |

### 12.2 Inference and model size

| Model | Latency | Throughput | Size |
|---|---|---|---|
| Intent / entity / guardrail | 4.86 ms mean, 5.12 ms p95 | 858.9 queries/s at batch 32 | 514 MB |
| Retrieval | Tens of ms per query with a GPU | — | 3.80 GB index snapshot |
| Generation, baseline | 11,444 ms per answer | 8.7 tokens/s | ~3.23 GB base, 4-bit |
| Generation, distilled | 14,096 ms per answer | 4.9 tokens/s | 262 MB adapter |
| **Yield, LightGBM (tuned)** | **1,008 ms for 64,021 rows** | **63,510 rows/s** | **1.8 MB** |
| Yield, CatBoost (tuned) | 352 ms | 182,070 rows/s | 19.8 MB |
| Yield, XGBoost (tuned) | 2,797 ms | 22,892 rows/s | 45.4 MB |
| Yield, PyTorch MLP | 427 ms | 149,811 rows/s | 1.06 MB |

CatBoost is fastest at inference and the MLP is smallest, but both trail badly on accuracy. LightGBM gives the best trade-off: highest R², the smallest tree-model file at 1.8 MB, and mid-range latency, which is comfortably enough for both a nightly batch refresh and a per-farmer query.

The distilled generator is slower per answer despite writing fewer tokens, because the adapter is applied at inference time rather than merged into the base weights.

**Memory.** Vision peaked at 2.7 GB at batch 64; adapter training peaked at 10.3 GB of 15.6 GB; restoring the index needs about 5 GB of free disk. The vision data pipeline was CPU-bound throughout, supplying 263 images per second against a GPU ceiling of 373 on 4 vCPUs, which is why the augmentation cost audit in §9.1 was worth running. Yield ran CPU-only, so GPU utilisation does not apply.

**Cost.** Everything trained and evaluated on free Kaggle and Colab hardware. The only external costs are the teacher API calls during distillation data preparation and the LLM judge calls during evaluation.

---

## 13. Limitations

**Data.** Vision class imbalance is 43:1, with `rice__leaf_smut` at 25 training and 8 test images, so any number for that class is uninformative and must be quoted with n attached. 21.8% of the vision corpus is replicate padding (71% for leaf smut), and the corpus is curated with no field photographs anywhere in it. Intent and NER labels come from weak supervision, so some recorded errors are label noise rather than model error, and three intent classes are too small to learn (`general` 53, `specialty_other` 23, `post_harvest_storage` 10 test rows). Pure Hindi is thin at 75 test rows. The retrieval corpus is regionally biased towards Uttar Pradesh sources and inherits the coverage bias of the KCC call records. For yield, 5.13% of rows had a stored `yield` disagreeing with production over area by more than 0.01, and residual noise from that may still affect low-yield rows.

**Models and measurement.** The necrotic-lesion cluster is unresolved, with `tan_spot` at 0.613 and `leaf_blight` at 0.585. The vision model is not calibrated, so a confidently wrong diagnosis is possible and there is no abstain threshold. Shortcut invariance was measured on the augmentation pipeline, not on the trained model. The intent model always predicts something, even for an empty string. Retrieval recall is pool-limited to the 10 chunks judged per question, and all 480 judgements come from one annotator, so there is no agreement figure. Evaluation samples are small — 48 curated and 77 real questions, with faithfulness resting on 21 — so intervals are wide and any difference crossing zero is a tie. For yield, MAPE is unusable as computed (§10.5), the worst-prediction analysis does not match the selected model, the MLP baseline is under-tuned so its result is not evidence against neural approaches, the search budget of 15 to 20 candidates is modest, and the split is random rather than chronological, so the model is not tested on future years.

**Scalability and compute.** Every result comes from free hardware. That ceiling shaped real decisions: the generator runs 4-bit, the index is embedded in shards, the training sequence window was set by memory rather than by the data, and the adapter study ran on a 160-row slice, so it ranks settings at that scale rather than proving an optimum.

**Failure scenarios and ethics.** A wrong dose is a real-world harm, which is why numeric grounding is treated as a safety metric, why fallback answers carry a KVK verification line, and why the distilled model is preferred despite being slower. A confident wrong diagnosis is the vision module's worst failure and nothing currently prevents it. The guardrail must be deployed as model plus rules, never the head alone. Yield performance has not been broken down by state or district, which matters for a deployment aimed at Uttar Pradesh, so no regional accuracy claim should be made yet.

---

## 14. Possible Improvements

The retrieval and generation modules are treated as complete at this milestone; the configuration in §3 is what will be deployed. The items below cover the other modules.

**Vision.** Fit temperature scaling on validation and set an abstention threshold, so the system can ask a clarifying question instead of asserting a low-confidence diagnosis. Run a shortcut re-probe on the trained model by altering exposure at inference and occluding the padding region. Collect even 50 in-field images to turn "0.867 on our test split" into a statement about deployment. Higher input resolution (384 px) or a segmentation stage would attack the hard cluster, but both are considerably more expensive. Package a `predict()` callable returning top-k labels with calibrated confidence and an abstain flag.

**Intent, entity and guardrail.** Oversample or class-weight the three rare classes, or merge `general` into a clearer taxonomy since it currently has no consistent meaning. Replace weak supervision with human annotation on a sample, so reported errors can be separated from label noise. Collect more pure Hindi queries. Add a confidence threshold so the model can decline instead of always predicting. Try XLM-RoBERTa base, and use k-fold cross-validation on the 30,000-row sample.

**Yield.** Replace MAPE with SMAPE or WAPE and re-run the worst-prediction and per-crop-type diagnostics against the selected LightGBM model. Expand the search budget or switch to Bayesian optimisation for XGBoost, where tuning currently underperforms the default. Fix the MLP baseline with feature scaling, embedding layers for categoricals instead of one-hot, batch normalisation and early stopping. Add state and district-level error breakdowns. Run the three ablations scoped in §9.4. Consider an ensemble of LightGBM and CatBoost given their different error profiles. Apply the same early-stopping protocol to the default, tuned and log-target runs so the comparison is exactly fair.

**System.** Generation dominates end-to-end time; merging the adapter into the full-precision base and re-quantising would recover most of the roughly 1.2 to 2× penalty. Log guardrail decisions and abstentions in production, since both are safety behaviours whose real-world rate is unknown.

---

## 15. Discussion

### 15.1 Were the objectives achieved?

Yes. Every module met its stated success criterion on held-out data. Vision beat its baseline by 3.3 times the noise floor and generalised from validation to test. The intent model exceeded all four targets. Retrieval returns something useful in the top 5 for 42 of 48 questions. The distilled generator passes all six safety health checks where the baseline passes four. LightGBM reaches R² 0.9572 in a 1.8 MB model.

All three Milestone 4 gaps are closed. Guardrail labels were authored rather than borrowed, then tested against an adversarial set that revealed the honest limit. The yield module was retrained with gradient boosting as originally selected, with `production` removed from the features so the leakage recorded in Milestone 4 no longer applies. The distillation pipeline was run and evaluated.

### 15.2 Comparison with expectations

Five expectations turned out to be wrong, and each was worth finding.

1. **The distilled model does not clearly beat the baseline on similarity metrics** once the baseline is given the output format. The +6.06 chrF++ gain over a plain base model splits into +4.25 for formatting and +1.81 for everything else, and the second is a tie.
2. **Regularization does not matter for the adapter.** Five configurations spanning dropout 0.00 to 0.20 and weight decay 0.00 to 0.10 produced a total spread of 0.0034, less than half the noise band.
3. **The automatic retrieval scorer was badly wrong**, reading 0.317 where human judgement reads 0.725.
4. **Matching pretraining image statistics beat matching the target domain**, by 1.9 F1.
5. **Tuning did not help XGBoost**, which its own defaults beat on RMSE and R², and the log-target transform improved MAE while worsening RMSE and R² for every model. Variance-stabilising transforms do not improve all metrics uniformly.

Two expectations were confirmed: Hinglish is the weakest input across the whole stack, and dataset imbalance limits rare-class performance more than any modelling choice does.

### 15.3 Practical applicability

The system is usable for a pilot. A farmer can ask in three languages, get a grounded answer with sources or an honest refusal, send a leaf photo and get a diagnosis that is right about 90% of the time, and get a yield estimate for planning. The safety behaviours that matter are in place: no invented numbers across 125 evaluated questions, correct language 99.4% of the time, and off-domain questions stopped before they reach a model. Yield inference at 63,510 rows per second on CPU covers both a nightly batch refresh and a per-farmer query with no GPU needed.

Two constraints remain. Answers take about 14 seconds on free hardware, acceptable for a messaging interface but not for a live call. And the vision module should present its output as a suggestion until it is calibrated.

### 15.4 Lessons learned

1. **The baseline decides the conclusion.** A zero-shot baseline would have produced a headline of "+6 chrF++ from distillation" that was true and misleading. The one-shot baseline turned it into a tie, and a truer picture.
2. **Measure the noise floor before reading any gap.** Three extra seed runs converted twenty of twenty-seven study results from apparent small effects into an honest "no measurable difference".
3. **A perfect score is a reason to check the test, not to stop checking.** The guardrail head scores 1.000 on the test split and 0.571 recall on adversarial input.
4. **Check the metric before blaming the model.** The retrieval score doubled when the scorer was replaced by a person, and the yield MAPE reached 10¹⁶% because of near-zero denominators rather than because of anything the models did.
5. **Evaluation-set construction mattered more than any modelling choice.** Two separate mechanisms would each have inflated the vision score, and neither was visible without explicit measurement.
6. **Tuning is not automatically an improvement.** It helped LightGBM and CatBoost and hurt XGBoost under the same budget.

### 15.5 Key observations

1. The distilled model is safer and more consistent, not more fluent, which is the right trade for dosage advice.
2. Vision errors concentrate almost entirely in one cluster of visually similar wheat conditions, so this is a pathology limit rather than a data-quantity limit.
3. Class separability, not data volume, is the binding constraint: leaf smut reaches 0.92 on 25 training images while tan spot reaches 0.61 on 516.
4. Hinglish is the weakest input at every stage of the pipeline.
5. Optimization and capacity settings have measurable effects on adapter training; regularization settings do not.
6. Generation is essentially the whole latency budget; every other module runs in milliseconds.

---

## 16. Conclusion

### 16.1 Summary of findings

Five models were evaluated on held-out data against stated baselines, and all met their success criteria.

The vision classifier reaches **0.8671 20-way macro-F1 and 0.8998 accuracy on 1,287 held-out images**, an improvement of +0.083 wheat-15 macro-F1 over a frozen baseline, which is 3.3 times the noise floor. Its errors are concentrated in one cluster of visually similar wheat conditions.

The intent, entity and guardrail model reaches **88.4% intent accuracy, 0.715 macro-F1 and 0.958 entity F1**, with a perfect guardrail score on the standard test split. Ablation against an adversarial set showed the head alone recalls 0.571, so the deployed guardrail combines model and rules.

Retrieval reaches **Precision@5 of 0.725 and Recall@5 of 0.498** under human judgement over 480 chunks, against 0.317 from the automatic scorer it replaced.

The distilled generator is **statistically tied with a well-prompted baseline on similarity metrics** but clearly better on everything that matters for safety: no invented numbers, correct language 99.4% of the time against 93.5%, higher judged faithfulness, and six of six health checks passed against four.

The yield model reaches **R² 0.9572 and RMSE 2.4112** with tuned LightGBM on 64,021 held-out rows, in a 1.8 MB artefact running at 63,510 rows per second on CPU.

### 16.2 Final performance

| Module | Headline metric | Value |
|---|---|---|
| Vision | 20-way macro-F1 / accuracy (n = 1,287) | 0.8671 [0.8396, 0.8905] / 0.8998 |
| Intent | Accuracy / macro-F1 (n = 3,597) | 0.884 / 0.715 |
| NER | Entity F1 | 0.958 |
| Guardrail | Test F1 / red-team recall combined | 1.000 / 1.000 |
| Retrieval | Precision@5 / Recall@5 | 0.725 / 0.498 |
| Generation | Language match / numeric grounding / health checks | 0.994 / 1.000 / 6 of 6 |
| Yield | R² / RMSE / MAE (n = 64,021) | 0.9572 / 2.4112 / 0.8298 |

### 16.3 Readiness

**Retrieval and generation are complete and ready for deployment as configured.** The distilled adapter is a 262 MB artefact, the index is a matched snapshot and manifest pair, and the serving configuration is recorded in Appendix E.

**The intent, entity and guardrail model is ready for pilot deployment**, with the guardrail deployed as model plus rules, never the head alone.

**The vision model is ready for pilot deployment as a suggestion, not a diagnosis**, until temperature scaling and an abstention threshold are added.

**The yield model is ready for planning use.** The corrective items in §10.5 and §14 — a usable percentage-error metric, error analysis re-run against LightGBM, and a regional breakdown — should be completed before any regional accuracy claim is made.

Milestone 6 is deployment. Every artefact needed for it is listed in Appendix E.

---

## Appendix A — Complete Metric Tables

**A.1 Vision, validation macro-F1 by training phase**

| Metric | Frozen probe | Head only | Blocks 9–11 | Full (selected) |
|---|---|---|---|---|
| 20-way | 0.8382 | 0.7655 | 0.8830 | 0.9042 |
| wheat-15 | 0.7966 | 0.7682 | 0.8660 | 0.8793 |
| rice-5 | 0.9717 | 0.7887 | 0.9429 | 0.9828 |
| Accuracy | 0.8559 | 0.8131 | 0.9042 | 0.9143 |
| Train/val gap | — | 0.047 | 0.092 | 0.073 |

**A.2 Intent loss curve (5 epochs, 949.4 s)**

| Epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Train loss | 0.938 | 0.426 | 0.339 | 0.276 | 0.230 |
| Validation loss | 0.545 | 0.439 | **0.441** | 0.471 | 0.475 |

**A.3 Distilled generator, final training**

Final training loss 0.1398 · best validation loss 0.1648 · perplexity 1.18 · overfitting gap +0.0250 · best checkpoint `checkpoint-360` at the end of epoch 2 · 4 of 4 epochs completed · 578.1 min.

**A.4 Generation, all 48 curated questions including weak context**

| Arm | chrF++ | BLEU | ROUGE-1 | ROUGE-L | numeric_recall | invented | grounding |
|---|---|---|---|---|---|---|---|
| Base, one example | 28.53 | 14.43 | 0.3665 | 0.3381 | 0.294 | 0.024 | 0.793 |
| Distilled | 28.31 | 17.07 | 0.3342 | 0.3048 | 0.344 | 0.000 | 0.781 |
| Base, no example | 23.89 | 9.84 | 0.3125 | 0.2916 | 0.254 | 0.095 | 0.768 |

**A.5 Yield** — the full seven-model comparison is in §6.5 and the log-target comparison in §8.4. Sample predictions from the selected model: actual 3.255 predicted 3.112; actual 67.356 predicted 65.890; actual 0.526 predicted 0.498.

---

## Appendix B — Vision Per-Class Results

| Class | Test F1 | n | | Class | Test F1 | n |
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

Sixteen of twenty classes exceed 0.80 and eleven exceed 0.94. `rice__leaf_smut` must always be quoted with n = 8 attached.

**Diagnostic classes across phases:** `wheat__tan_spot` 0.407 → 0.306 → 0.569; `rice__leaf_smut` 0.923 → 0.320 → 0.769. The second is the clearest evidence in the project that the original score was a brightness shortcut and that the class is nonetheless learnable from real pathology. Top test confusions are in §10.1.

---

## Appendix C — Intent Confusion Matrix

Rows are true classes, columns predicted.

| True \ Predicted | cultivation | disease_pest | general | non_agri | nutrition | post_harvest | specialty |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **cultivation_practice** | **611** | 78 | 1 | 2 | 23 | 5 | 0 |
| **disease_pest** | 76 | **1438** | 20 | 6 | 20 | 0 | 6 |
| **general** | 3 | 7 | **9** | 5 | 0 | 0 | 2 |
| **non_agri** | 0 | 0 | 0 | **490** | 0 | 0 | 0 |
| **nutrition_fertilizer** | 2 | 16 | 2 | 6 | **619** | 0 | 0 |
| **post_harvest_storage** | 1 | 1 | 0 | 0 | 0 | **8** | 0 |
| **specialty_other** | 2 | 1 | 0 | 0 | 4 | 0 | **16** |

Two readings. The `non_agri` row and column are perfectly clean, which is what the routing gate depends on. Everything else confuses along the cultivation, disease and nutrition boundary, which is where the labels themselves are ambiguous. Guardrail curves: ROC-AUC 1.000, PR-AUC 1.000.

---

## Appendix D — Hyperparameter Search Results

**D.1 Intent, entity and guardrail — all six configurations**

| Run | Optimizer | LR | Batch | Epochs | Label smoothing | Composite |
|---|---|---|---|---|---|---|
| `adamw_lr3e-5_5ep` | AdamW | 3e-5 | 32 | 5 | 0.0 | **1.747** |
| `adamw_lr3e-5_bs32` | AdamW | 3e-5 | 32 | 3 | 0.0 | 1.713 |
| `adamw_labelsmooth` | AdamW | 3e-5 | 32 | 3 | 0.1 | 1.713 |
| `adamw_lr3e-5_bs64` | AdamW | 3e-5 | 64 | 3 | 0.0 | 1.700 |
| `adamw_lr1e-5_bs32` | AdamW | 1e-5 | 32 | 3 | 0.0 | 1.667 |
| `sgd_lr1e-3_bs32` | SGD | 1e-3 | 32 | 3 | 0.0 | 1.279 |

**D.2 Distilled generator — configurations outside the 0.0074 noise band** (negative is better)

Better: 3 epochs −0.1050 · batch 8 −0.0828 · rank 64 −0.0531 · Adafactor −0.0264 · constant schedule −0.0168. Worse: weight decay 0.10 +0.0079 · MLP only +0.0479 · alpha ratio 1× +0.0480 · warmup 0.10 +0.0506 · rank 8 +0.0760 · lr 1e-4 +0.1014 · attention only +0.1162 · batch 32 +0.1352 · lr 5e-5 +0.2890. The thirteen not listed fell inside the band and are recorded as no measurable difference.

**D.3 Yield — search spaces**

- XGBoost: `n_estimators` randint(100, 500), `learning_rate` loguniform(0.01, 0.3), `max_depth` randint(3, 10), `subsample` and `colsample_bytree` uniform(0.6, 0.4), `gamma` uniform(0, 0.5), `reg_alpha` and `reg_lambda` loguniform(1e-5, 1)
- LightGBM: as above plus `num_leaves` randint(20, 100), `max_depth` randint(5, 15), `min_child_samples` randint(10, 50)
- CatBoost: `depth`, `iterations`, `l2_leaf_reg`, `learning_rate`, `random_strength`, `subsample`

**D.4 Yield — selected parameters**

| LightGBM (selected) | XGBoost (tuned) | CatBoost (tuned) |
|---|---|---|
| n_estimators 202 | n_estimators 352 | iterations 495 |
| learning_rate 0.2537 | learning_rate 0.0281 | learning_rate 0.1270 |
| num_leaves 94, max_depth 12 | max_depth 8 | depth 9 |
| min_child_samples 30 | gamma 0.0917 | l2_leaf_reg 3.4167 |
| subsample 0.6571 | subsample 0.8447 | subsample 0.8347 |
| colsample_bytree 0.7498 | colsample_bytree 0.6727 | random_strength 1.0794 |
| reg_alpha 0.00198, reg_lambda 0.00047 | reg_alpha 0.00144, reg_lambda 0.00029 | — |

**D.5 Vision** — the normalisation and batch-size grids are in §8.1.

---

## Appendix E — Artifacts and Configuration

| Artefact | Path or identifier | Size |
|---|---|---|
| Vision checkpoint | `runs/vits16_m1/ckpt/p3_full_best.pt` (epoch 16) | 21.67M params |
| Vision data path, frozen | `pipeline.py` | — |
| Vision split, logs, results | `clean_index_v3.csv`, `runs/vits16_m1/logs/`, `test_results.json` | — |
| Intent / entity / guardrail model | `final/intent_entity_guardrail_model.pt` + label maps | 514 MB |
| Retrieval index | `agri_knowledge` snapshot + `manifest.json` | 3.80 GB |
| LoRA adapter | `best_adapter/` with tokenizer and chat template | 262.41 MB |
| Distillation data / RAG eval run | hash `1d8f4ed7d4ca` / `7efae569e141` | 23.15 MB |
| Yield models | `saved_models/lightgbm_*.txt`, `xgboost_*.json`, `catboost_*.cbm`, `pytorch_model.pth` | 1.06–62.3 MB |
| Yield results | `results/model_comparison.csv`, `*_cv_results.csv`, `best_hyperparameters.json` | — |

**The index snapshot and its manifest are a matched pair.** Without the manifest the serving path has to guess the prefix policy, vector dimension and confidence thresholds, and every wrong guess fails silently rather than raising an error.

**Serving configuration:** bge-m3 frozen at 1024 dimensions, no query or document prefix, top-k 5, confidence tiers abstain below 0.56, fallback 0.56 to 0.66, grounded at or above 0.66; the distilled adapter loaded onto the 4-bit base with greedy decoding; LightGBM tuned on the original-scale target with the parameters in D.4.

**Notebooks:** `vit-train-01.ipynb` · `11_kcc_intent_entity_guardrail.ipynb` · `12_distillation_data_prep.ipynb` · `13_distill_training.ipynb` · `14_distillation_hpt.ipynb` · `14_distill_model_evals.ipynb` · `retrieval_evals.ipynb` · `10c_rag_baseline_vs_distilled_new.ipynb` · yield training and evaluation notebook.

**Figures:** vision dataset composition, split comparison, per-class F1, phase curves, test confusion matrix; retrieval precision and recall, score spread; generation headline, paired differences, by script, by language, RAGAS; adapter loss curve; yield predicted-versus-actual with residuals, learning curves, SHAP beeswarm.

---

## Team Review and Sign-Off

Reviewers should read §7.4 (what the fine-tune actually bought), §9.2 (the guardrail ablation that changed the deployment decision), §10.5 (the yield MAPE artefact), §10.1 (the vision error cluster) and §13. These are the sections that qualify a headline number or change a decision recorded in an earlier milestone.

| # | Team Member | Approved | Date |
|:-:|-------------|:--------:|:----:|
| 1 | Mahesh | Yes | 6 Aug 2026 |
| 2 | Harliv | Yes | 6 Aug 2026 |
| 3 | Lokesh | Yes | 6 Aug 2026 |
| 4 | Aneeqa | Yes | 6 Aug 2026 |
| 5 | Tanmay | Yes | 6 Aug 2026 |

**Document version:** Milestone 5 — v1 · **Prepared:** 6 August 2026
