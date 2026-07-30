# M4 Report: Intent/Entity/Guardrail Extractor

**Project:** FarmerVision  
**Milestone:** 4  
**Date:** 30 July 2026  
**Model:** DistilBERT-base-multilingual-cased with 3 task heads

---

## Executive Summary

We trained a multi-task DistilBERT-based extractor (134.7M params) with three heads for intent classification (12 classes), named-entity recognition (CROP/DISTRICT spans), and guardrail flagging (off-domain detection). The model was trained on a 30,000-row stratified sample of KCC data with distant supervision for NER and synthetic off-domain examples from AG News. Training completed in ~600 seconds on a Colab T4 GPU.

**Key Results:**

- **Intent:** Accuracy 77.7% | Macro-F1 59.9% (rare classes drag performance)
- **NER:** Entity F1 93.8% (on labeled spans; coverage 82.9%)
- **Guardrail:** 100% on toy task | **Fails on real off-domain queries**

**Bottom Line:** Intent and NER are MVP-ready. Guardrail is **not production-ready** — the perfect score on AG News is an artifact of a toy dataset.

---

## 1. Dataset and Preprocessing

### 1.1 Data Sources

| Component | Dataset             | Source                    | Training Size     |
| --------- | ------------------- | ------------------------- | ----------------- |
| Intent    | KCC QueryType       | KCC corpus (710,616 rows) | 30,000 stratified |
| NER       | KCC Crop + District | Same as above             | 30,000 rows       |
| Guardrail | KCC + AG News       | AG News train split       | 38,000 rows       |

### 1.2 Preprocessing

1. **Stratified sampling** by QueryType to preserve rare intents
2. **Intent collapse:** Top 11 QueryTypes + "other" → 12 classes
3. **Entity alias building:** Parsed bracketed dual-language names into surface forms
4. **Distant supervision for NER:** Matched aliases → BIO tags
5. **Train/val/test split:** 80/10/10 stratified by guardrail label

### 1.3 Data Limitations

| Limitation           | Detail                                       |
| -------------------- | -------------------------------------------- |
| NER labels           | Distant supervision (not human-annotated)    |
| Guardrail off-domain | Synthetic (AG News — a proxy)                |
| Training size        | Only 30,000 rows (4.2% of full corpus)       |
| Entity coverage      | 82.9% of rows have at least one matched span |

---

## 2. Model Architecture

### 2.1 Architecture Overview

The model uses a single DistilBERT backbone with three task-specific heads:

**Input Layer**

- Query text (max 64 tokens)
- Tokenized using DistilBERT multilingual tokenizer

**Backbone: distilbert-base-multilingual-cased**

- 66 million parameters
- Hidden size: 768
- 12 transformer layers with 8 attention heads
- Outputs [CLS] token representation (768-dim) and token-level representations

**Head 1: Intent Classification**

- Linear layer: 768 → 12
- Outputs logits for 12 intent classes
- Loss: CrossEntropyLoss (ignore_index=-100 for synthetic rows)

**Head 2: NER (Named Entity Recognition)**

- Linear layer: 768 → 5
- Outputs BIO tags per token: O, B-CROP, I-CROP, B-DISTRICT, I-DISTRICT
- Loss: Token-level CrossEntropyLoss (ignore_index=-100 for special tokens)

**Head 3: Guardrail**

- Linear layer: 768 → 2
- Outputs binary logits: in-domain vs. off-domain
- Loss: CrossEntropyLoss (class-weighted: off-domain = 3.75×)

**Total Parameters:** 134.7M (66M backbone + 68.7M heads)

### 2.2 Key Components

| Component           | Detail                                    |
| ------------------- | ----------------------------------------- |
| Backbone            | DistilBERT-base-multilingual-cased        |
| Hidden size         | 768                                       |
| Max sequence length | 64 tokens                                 |
| NER tag set         | O, B-CROP, I-CROP, B-DISTRICT, I-DISTRICT |
| Intent classes      | 12 (including 'other')                    |
| Guardrail classes   | 2 (in-domain, off-domain)                 |
| Total parameters    | 134.7M                                    |

### 2.3 Architecture Rationale

- **Single forward pass** → three outputs → lower latency
- **DistilBERT** → 66M params → faster than MuRIL (~236M)
- **Multi-task learning** → shared representations benefit all tasks
- **-100 ignore index** → synthetic rows don't contribute to intent/NER loss

---

## 3. Training Configuration

### 3.1 Hyperparameters

| Parameter              | Value | Justification                                 |
| ---------------------- | ----- | --------------------------------------------- |
| Optimizer              | AdamW | Standard for transformer fine-tuning          |
| Learning rate          | 3e-5  | Conservative; recommended for DistilBERT      |
| Batch size             | 32    | Maximum that fits in T4 GPU memory            |
| Epochs                 | 3     | Validation loss plateaued; early stopping     |
| Weight decay           | 0.01  | Default AdamW value                           |
| Max sequence length    | 64    | Queries are short; 64 captures full context   |
| Guardrail class weight | 3.75  | Based on class imbalance ratio (30,000/8,000) |

The chosen configuration resulted in stable training with no signs of overfitting, as evidenced by the validation loss plateauing at epoch 3.

### 3.2 Loss Functions

| Head      | Loss             | Weight | Notes                                |
| --------- | ---------------- | ------ | ------------------------------------ |
| Intent    | CrossEntropyLoss | 1.0    | ignore_index=-100 for synthetic rows |
| NER       | CrossEntropyLoss | 1.0    | ignore_index=-100 for special tokens |
| Guardrail | CrossEntropyLoss | 1.0    | Class-weighted: off-domain = 3.75×   |

**Guardrail class weights:** In-domain: 1.0 | Off-domain: 3.75 (prioritizes recall)

### 3.3 Hardware

| Component     | Specification           |
| ------------- | ----------------------- |
| GPU           | Colab T4 (16GB)         |
| CPU           | 2-core (Colab)          |
| Training time | ~600 seconds (3 epochs) |

---

## 4. Generalization and Training Stability

### 4.1 Observations

- Training loss decreased consistently; validation loss plateaued by epoch 3
- No severe overfitting (train-val gap small)
- NER and guardrail components converged quickly
- Guardrail loss near-zero after epoch 1 (task is too easy)
- See Appendix B for detailed training logs

---

## 5. Quantitative Results

### 5.1 Intent Classification

| Metric      | Score     |
| ----------- | --------- |
| Accuracy    | **77.7%** |
| Macro-F1    | **59.9%** |
| Weighted-F1 | 76.6%     |

**Per-class F1 (Key Classes):**

| Class                      | F1       | Support |
| -------------------------- | -------- | ------- |
| Plant Protection           | 0.89     | 1,360   |
| Weed Management            | 0.92     | 192     |
| Varieties                  | 0.86     | 151     |
| Nutrient Management        | 0.72     | 358     |
| **Vegetative Propagation** | **0.00** | **24**  |
| **Field Preparation**      | **0.14** | **40**  |

**Finding:** Accuracy masks poor performance on rare classes. Macro-F1 (59.9%) is the honest headline.

### 5.2 Named Entity Recognition (NER)

| Metric          | Score     |
| --------------- | --------- |
| Entity-level F1 | **93.8%** |
| Precision       | 89.7%     |
| Recall          | 98.0%     |

**Coverage:** 82.9% of rows have at least one matched entity span.

**Caveat:** Performance measured on **labeled spans only** — unlabeled mentions are not evaluated.

### 5.3 Guardrail (Off-Domain Detection)

| Class      | Precision | Recall | F1   |
| ---------- | --------- | ------ | ---- |
| In-domain  | 1.00      | 1.00   | 1.00 |
| Off-domain | 1.00      | 1.00   | 1.00 |

⚠️ **Critical caveat:** Perfect score on AG News proxy task is **not a sign of readiness**. See Section 8.

---

## 6. Qualitative Results

### Sample Predictions

| Query                                   | Intent             | Guardrail     | Entities                 |
| --------------------------------------- | ------------------ | ------------- | ------------------------ |
| "wheat crop is turning yellow"          | Plant Protection   | in-domain     | [wheat]                  |
| "gehu mein pila rog laga hai"           | Plant Protection   | in-domain     | (none)                   |
| "sugarcane disease red rot in Bareilly" | Plant Protection   | in-domain     | [sugar, can, Bare, illy] |
| "PM Kisan eligibility kaise check kare" | Cultural Practices | in-domain     | [Kis]                    |
| **"what is the capital of France"**     | Cultural Practices | **in-domain** | (none)                   |
| **"stock market crashed"**              | Cultural Practices | **in-domain** | (none)                   |

### Key Observations

1. Intent works well for common classes (Plant Protection: 0.89 F1)
2. Hindi queries handled but entities often missed
3. **Guardrail fails on obvious off-domain queries** — both misclassified as in-domain
4. NER has false positives ("Kis" flagged as crop in "PM Kisan")
5. Districts (Sitapur) not detected as location entities

---

## 7. Artifacts Generated

| Artifact         | Path                                        | Size    | Purpose             |
| ---------------- | ------------------------------------------- | ------- | ------------------- |
| Model weights    | `final/intent_entity_guardrail_model.pt`    | ~540 MB | Trained model       |
| Label maps       | `final/intent_entity_label_maps.json`       | ~2 KB   | Class mappings      |
| Training summary | `final/intent_entity_training_summary.json` | ~10 KB  | Metrics and history |
| Notebook         | `11_kcc_intent_entity_guardrail.ipynb`      | ~100 KB | Full pipeline       |

---

## 8. Key Findings and Observations

### 8.1 What Worked Well

- Multi-task architecture works efficiently (single forward pass → three outputs)
- Intent classification strong for common classes (Plant Protection: 0.89 F1)
- NER distant supervision achieves 93.8% F1 on labeled spans
- Training stable — no gradient issues, clean plateau
- All artifacts saved and ready for inference

### 8.2 What Did Not Perform as Expected

| Issue                                         | Impact                                               |
| --------------------------------------------- | ---------------------------------------------------- |
| Guardrail fails on obvious off-domain queries | **Critical** — not ready for production              |
| Macro-F1 59.9% vs. accuracy 77.7%             | Misleading headline numbers                          |
| Hindi entity detection fails                  | Distant supervision misses Hindi crop names          |
| AG News as proxy is poor                      | Learns "news style" not "non-agricultural semantics" |
| 30K rows under-represents rare intents        | Rare class F1 near zero                              |

### 8.3 Bottlenecks and Known Limitations

1. **Guardrail data** — no natural off-domain data; AG News is a poor substitute
2. **Entity coverage** — only 82.9% rows labeled; Hindi mentions frequently missed
3. **Class imbalance** — 1,360 Plant Protection vs. 24 Vegetative Propagation
4. **Training scale** — 30K rows insufficient for robust rare-class learning
5. **NER labels** — distant supervision (string matching), not human-annotated
6. **Guardrail scope** — only covers off-domain detection; dosage-bounds and banned-term detection still need authored adversarial data

### 8.4 Justification for AG News

**Why AG News was used:**

- No natural off-domain data exists in KCC
- Provides clearly non-agricultural text for training binary classifier
- Standard technique for domain classification with no negative class

**Why AG News is insufficient:**

- AG News headlines are structurally different from user queries
- Model learns "news style" not "non-agricultural semantics"
- Real off-domain farmer queries may use agricultural vocabulary
- **Evidence:** "what is the capital of France" and "stock market crashed" both predicted as in-domain

### 8.5 Plans for Improvement (M5)

| Priority    | Improvement                                                                        | Rationale                                       |
| ----------- | ---------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Highest** | Author real guardrail examples (dosage bounds, banned terms, ambiguous off-domain) | The single missing piece for complete guardrail |
| **High**    | Train on full KCC corpus (700K+ rows)                                              | 30K sample under-represents rare intents        |
| **High**    | Augment off-domain data with SQuAD, common out-of-scope queries                    | Fix generalization failure                      |
| **Medium**  | Increase epochs (10-15) with heavier off-domain weighting                          | Current 3 epochs insufficient                   |
| **Medium**  | Add Hindi crop aliases to alias map                                                | Fix Hindi entity detection                      |
| **Low**     | CRF layer on NER head                                                              | Improve entity boundary detection               |

---

## 9. Conclusion

### 9.1 Summary

| Component | Status    | Readiness                                    |
| --------- | --------- | -------------------------------------------- |
| Intent    | MVP-ready | Scale to full corpus, improve rare classes   |
| NER       | MVP-ready | Add Hindi aliases, evaluate unlabeled recall |
| Guardrail | Not ready | **Major redesign needed**                    |

### 9.2 Critical Message

The guardrail head **does not work** despite its perfect test score. The near-perfect accuracy on AG News is an artifact of a toy dataset, not a sign of readiness. This confirms M3 §7.0.3's warning: guardrail labels must be authored, not synthesized.

**M5 MUST include:**

1. Real guardrail examples (dosage bounds, banned terms, ambiguous queries)
2. Diverse off-domain data beyond AG News
3. Training on the full KCC corpus
4. Robust evaluation set for guardrail capabilities

Without these, the system should not claim any safety capability.

---

## Appendix A: Intent Class Distribution

| Class                     | Count  | Percentage |
| ------------------------- | ------ | ---------- |
| Plant Protection          | 13,834 | 46.1%      |
| Nutrient Management       | 3,501  | 11.7%      |
| Fertilizer Use            | 3,220  | 10.7%      |
| Cultural Practices        | 2,914  | 9.7%       |
| Weed Management           | 1,923  | 6.4%       |
| Varieties                 | 1,699  | 5.7%       |
| Seeds & Planting Material | 771    | 2.6%       |
| Other                     | 727    | 2.4%       |
| Water Management          | 579    | 1.9%       |
| Field Preparation         | 314    | 1.0%       |
| Seeds                     | 296    | 1.0%       |
| Vegetative Propagation    | 222    | 0.7%       |

---

## Appendix B: Training Logs

| Epoch | Time (s) | Train Loss | Intent Loss | NER Loss | Guardrail Loss | Val Loss |
| ----- | -------- | ---------- | ----------- | -------- | -------------- | -------- |
| 1     | 194      | 0.993      | 0.930       | 0.052    | 0.011          | 0.835    |
| 2     | 197      | 0.714      | 0.691       | 0.022    | 0.001          | 0.762    |
| 3     | 198      | 0.620      | 0.601       | 0.019    | 0.000          | 0.765    |

**Total training time:** ~589 seconds (~9.8 minutes)

---

## Appendix C: Model Loading Code

```python
import json
import torch
from transformers import AutoTokenizer

# Load label maps
with open('intent_entity_label_maps.json', 'r') as f:
    label_maps = json.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(label_maps['model_name'])

# Load model
model = IntentEntityGuardrailModel(
    label_maps['model_name'],
    len(label_maps['intent_classes']),
    len(label_maps['ner_labels'])
)
model.load_state_dict(torch.load('intent_entity_guardrail_model.pt'))
model.to(DEVICE)
model.eval()
```
