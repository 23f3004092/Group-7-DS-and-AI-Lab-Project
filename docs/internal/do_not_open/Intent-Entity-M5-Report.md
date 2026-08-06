# Milestone 5 Report: Intent, Entity, and Guardrail Extractor

---

## 1. Introduction

### 1.1. Project Objective

The objective of this project is to develop and evaluate a robust, multi-task Natural Language Processing (NLP) pipeline for processing farmer queries in a multilingual setting (English, Hinglish, and Hindi). The core component is a single neural model that performs three crucial classification and extraction tasks simultaneously:

1. **Intent Classification:** Categorizing a query into one of seven agricultural domains:
   - `cultivation_practice`
   - `disease_pest`
   - `general`
   - `non_agri`
   - `nutrition_fertilizer`
   - `post_harvest_storage`
   - `specialty_other`

2. **Named Entity Recognition (NER):** Extracting the primary crop entity mentioned in the query.

3. **Guardrail Flagging:** Detecting unsafe or dangerous advice queries, such as those suggesting banned substances or extreme dosages.

### 1.2. Scope of Evaluation

This report presents the comprehensive evaluation of the final multi-task NLP model. The evaluation includes quantitative performance metrics, hyperparameter analysis, qualitative analysis of model outputs, error analysis, and robustness checks. The evaluation is performed on a held-out test set that was not used during training or hyperparameter tuning.

### 1.3. Models Evaluated

| Model      | Description                                                                              |
| ---------- | ---------------------------------------------------------------------------------------- |
| Baseline   | Pretrained backbone (distilbert-base-multilingual-cased) with randomly initialized heads |
| Fine-tuned | Best configuration trained on full dataset                                               |

### 1.4. Objectives of Milestone 5

- To conduct a thorough and unbiased evaluation of the final model
- To compare hyperparameter configurations and identify the optimal setup
- To provide detailed analysis of the model's strengths and weaknesses
- To report performance metrics critical for assessing model readiness for deployment
- To identify potential failure modes and areas for future improvement

---

## 2. Experimental Setup

### 2.1. Hardware and Software Environment

| Component            | Specification                      |
| -------------------- | ---------------------------------- |
| Hardware             | Tesla T4 GPU                       |
| Python Version       | 3.12.13                            |
| PyTorch Version      | 2.10.0+cu128                       |
| Transformers Version | 5.0.0                              |
| scikit-learn Version | 1.6.1                              |
| Random Seed          | 42                                 |
| Backbone             | distilbert-base-multilingual-cased |
| Total Parameters     | 134,743,308                        |

### 2.2. Dataset Versions and Preprocessing

| Component          | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| Source Data        | `kcc_cleaned_all_crops.csv` (710,616 agricultural queries)       |
| Training Sample    | 30,000 examples stratified by QueryType                          |
| Intent Mapping     | 7-class taxonomy                                                 |
| NER Labeling       | Rule-based alias matching with manual additions (86.4% coverage) |
| Non-Agri Data      | 5,172 generated examples                                         |
| Guardrail Labeling | Rule-based flags for banned substances and dosage patterns       |

### 2.3. Training/Validation/Test Split

| Split      | Count  | Percentage |
| ---------- | ------ | ---------- |
| Training   | 28,772 | 80%        |
| Validation | 3,597  | 10%        |
| Test       | 3,597  | 10%        |

The split was stratified by the `guardrail_label`.

### 2.4. Random Seeds and Reproducibility

A fixed random seed (`SEED = 42`) was set for all random operations to ensure complete reproducibility.

---

## 3. Model Training Summary

### 3.1. Final Model Architecture

- **Backbone:** `distilbert-base-multilingual-cased` (134.7M parameters)
- **Intent Head:** Linear layer mapping `[CLS]` token embedding to 7 intent classes
- **NER Head:** Linear layer mapping token embeddings to 3 NER labels (`O`, `B-CROP`, `I-CROP`)
- **Guardrail Head:** Linear layer mapping `[CLS]` token embedding to binary safe/unsafe classification

### 3.2. Hyperparameters

| Parameter         | Value                  |
| ----------------- | ---------------------- |
| Optimizer         | AdamW                  |
| Learning Rate     | 3e-5                   |
| Batch Size        | 32                     |
| Epochs            | 5                      |
| Weight Decay      | 0.01                   |
| Label Smoothing   | 0.0                    |
| Gradient Clipping | 1.0                    |
| Scheduler         | Linear with 10% warmup |

### 3.3. Loss Function

| Task      | Loss Function                           |
| --------- | --------------------------------------- |
| Intent    | Cross-Entropy with ignore_index=-100    |
| NER       | Cross-Entropy with ignore_index=-100    |
| Guardrail | Weighted Cross-Entropy (class-balanced) |

### 3.4. Training Duration

| Metric               | Value                               |
| -------------------- | ----------------------------------- |
| Training Time        | 949.4 seconds (~15.8 minutes)       |
| Checkpoint Selection | Based on validation composite score |

### 3.5. Loss Curve

| Epoch | Train Loss | Val Loss |
| ----- | ---------- | -------- |
| 1     | 0.938      | 0.545    |
| 2     | 0.426      | 0.439    |
| 3     | 0.339      | 0.441    |
| 4     | 0.276      | 0.471    |
| 5     | 0.230      | 0.475    |

The validation loss bottomed at epoch 3 and slightly increased through epochs 4-5, indicating potential overfitting, but the composite validation score continued to improve.

---

## 4. Evaluation Methodology

### 4.1. Evaluation Protocol

1. Model trained using selected hyperparameters
2. Performance monitored on validation set after each epoch
3. Best checkpoint saved based on composite score
4. Final evaluation performed on held-out test set

### 4.2. Test Dataset Description

- **Size:** 3,597 examples
- **Composition:** Real agricultural queries, synthetic non-agri, and authored guardrail examples

**Class Distribution:**

| Class                | Support |
| -------------------- | ------- |
| disease_pest         | 1,576   |
| cultivation_practice | 738     |
| nutrition_fertilizer | 702     |
| non_agri             | 490     |
| general              | 53      |
| specialty_other      | 23      |
| post_harvest_storage | 10      |

### 4.3. Ground Truth Preparation

| Task      | Method                                            |
| --------- | ------------------------------------------------- |
| Intent    | Derived from QueryType mapping (weak supervision) |
| NER       | Heuristic-based crop alias matching               |
| Guardrail | Rule-based flags + generated examples             |

### 4.4. Baseline Models

- **Pretrained Backbone with Untrained Heads:** Demonstrates value of fine-tuning
- **Rule-based Guardrail:** Pattern-matching for banned substances/dosages

### 4.5. Success Criteria

| Metric          | Target |
| --------------- | ------ |
| Intent Accuracy | > 80%  |
| Intent Macro F1 | > 0.70 |
| NER Entity F1   | > 0.90 |
| Guardrail F1    | > 0.95 |

---

## 5. Performance Metrics

### 5.1. Intent Classification Metrics

| Metric              | Definition                                                           | Justification                                                |
| ------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------ |
| Accuracy            | Proportion of correctly classified queries                           | General measure of overall correctness                       |
| Macro F1 Score      | Harmonic mean of precision and recall, averaged across all classes   | Treats all classes equally; critical for imbalanced datasets |
| Per-Class Precision | Of queries predicted as class X, how many actually belong to class X | Identifies false positives per class                         |
| Per-Class Recall    | Of queries belonging to class X, how many were correctly identified  | Identifies false negatives per class                         |
| Per-Class F1        | Harmonic mean of precision and recall for each class                 | Balanced measure per class                                   |

### 5.2. NER (Crop Extraction) Metrics

| Metric          | Definition                                               | Justification                                        |
| --------------- | -------------------------------------------------------- | ---------------------------------------------------- |
| Entity-Level F1 | Span-based evaluation matching true entity spans exactly | More strict than token-level; standard for NER tasks |

### 5.3. Guardrail Metrics

| Metric    | Definition                                                  | Justification                        |
| --------- | ----------------------------------------------------------- | ------------------------------------ |
| Precision | Of flagged unsafe queries, how many are actually unsafe     | Minimizes false positives            |
| Recall    | Of actually unsafe queries, how many were correctly flagged | Minimizes false negatives            |
| F1 Score  | Harmonic mean of precision and recall                       | Balanced measure                     |
| PR-AUC    | Area under precision-recall curve                           | Robust for imbalanced datasets       |
| ROC-AUC   | Area under ROC curve                                        | Measures ability to separate classes |

---

## 6. Experimental Results

### 6.1. Quantitative Performance Tables

| Metric              | Baseline | Fine-tuned | Target | Status |
| ------------------- | -------- | ---------- | ------ | ------ |
| Intent Accuracy     | 0.371    | **0.884**  | > 0.85 | Met    |
| Intent Macro F1     | 0.111    | **0.715**  | > 0.70 | Met    |
| NER Entity F1       | 0.036    | **0.958**  | > 0.90 | Met    |
| Guardrail Precision | 0.000    | **1.000**  | > 0.95 | Met    |
| Guardrail Recall    | 0.000    | **1.000**  | > 0.95 | Met    |
| Guardrail F1        | 0.000    | **1.000**  | > 0.95 | Met    |

### 6.2. Per-Class Performance

| Class                | Precision | Recall    | F1        | Support   |
| -------------------- | --------- | --------- | --------- | --------- |
| cultivation_practice | 0.817     | 0.828     | 0.822     | 738       |
| disease_pest         | 0.905     | 0.912     | 0.909     | 1,576     |
| general              | 0.263     | 0.094     | 0.139     | 53        |
| non_agri             | 0.996     | 1.000     | 0.998     | 490       |
| nutrition_fertilizer | 0.860     | 0.882     | 0.871     | 702       |
| post_harvest_storage | 1.000     | 0.800     | 0.889     | 10        |
| specialty_other      | 1.000     | 0.696     | 0.821     | 23        |
| **Weighted Average** | **0.882** | **0.887** | **0.884** | **3,592** |
| **Macro Average**    | **0.834** | **0.745** | **0.778** | **3,592** |

### 6.3. Visualizations

#### Confusion Matrix (Intent Classification)

### 6.3. Visualizations

#### Confusion Matrix (Intent Classification)

| True \ Pred              | cultivation | disease_pest | general | non_agri | nutrition | post_harvest | specialty |
| :----------------------- | :---------: | :----------: | :-----: | :------: | :-------: | :----------: | :-------: |
| **cultivation_practice** |   **611**   |      78      |    1    |    2     |    23     |      5       |     0     |
| **disease_pest**         |     76      |   **1438**   |   20    |    6     |    20     |      0       |     6     |
| **general**              |      3      |      7       |  **9**  |    5     |     0     |      0       |     2     |
| **non_agri**             |      0      |      0       |    0    | **490**  |     0     |      0       |     0     |
| **nutrition_fertilizer** |      2      |      16      |    2    |    6     |  **619**  |      0       |     0     |
| **post_harvest_storage** |      1      |      1       |    0    |    0     |     0     |    **8**     |     0     |
| **specialty_other**      |      2      |      1       |    0    |    0     |     4     |      0       |  **16**   |

#### Guardrail ROC and PR Curves

| Metric  | Value |
| ------- | ----- |
| ROC-AUC | 1.000 |
| PR-AUC  | 1.000 |

---

## 7. Baseline Comparison

### 7.1. Baseline Model Performance

The baseline model (pretrained backbone with randomly initialized heads) achieves:

- Intent Accuracy: 37.1%
- Intent Macro F1: 0.111
- NER Entity F1: 0.036
- Guardrail F1: 0.000

### 7.2. Improvement Achieved

| Metric          | Baseline | Fine-tuned | Improvement |
| --------------- | -------- | ---------- | ----------- |
| Intent Accuracy | 0.371    | 0.884      | +138%       |
| Intent Macro F1 | 0.111    | 0.715      | +544%       |
| NER Entity F1   | 0.036    | 0.958      | +2,561%     |
| Guardrail F1    | 0.000    | 1.000      | N/A         |

### 7.3. Discussion

The dramatic improvement demonstrates that fine-tuning is essential for the backbone to adapt to the agricultural domain, and the multi-task learning approach effectively shares representations across tasks.

---

## 8. Hyperparameter Analysis

### 8.1. Hyperparameters Explored

| Parameter       | Values Tested | Best Value |
| --------------- | ------------- | ---------- |
| Optimizer       | AdamW, SGD    | AdamW      |
| Learning Rate   | 1e-5, 3e-5    | 3e-5       |
| Batch Size      | 32, 64        | 32         |
| Epochs          | 3, 5          | 5          |
| Weight Decay    | 0.01          | 0.01       |
| Label Smoothing | 0.0, 0.1      | 0.0        |

### 8.2. Effect of Parameter Changes

| Configuration                | Intent Macro F1 | NER Entity F1 | Composite Score |
| ---------------------------- | --------------- | ------------- | --------------- |
| adamw_lr3e-5_bs32 (3 epochs) | 0.753           | 0.960         | 1.713           |
| adamw_lr1e-5_bs32            | 0.730           | 0.937         | 1.667           |
| adamw_lr3e-5_bs64            | 0.747           | 0.953         | 1.700           |
| **adamw_lr3e-5_5ep**         | **0.785**       | **0.962**     | **1.747**       |
| sgd_lr1e-3_bs32              | 0.477           | 0.802         | 1.279           |
| adamw_labelsmooth            | 0.753           | 0.960         | 1.713           |

### 8.3. Key Insights

1. **Optimizer:** AdamW significantly outperforms SGD
2. **Learning Rate:** 3e-5 is better than 1e-5
3. **Epochs:** 5 epochs improves performance over 3 epochs
4. **Label Smoothing:** Does not provide additional benefit
5. **Batch Size:** 32 is slightly better than 64

---

## 9. Ablation Study (Guardrail Layer)

### 9.1. Methodology

The guardrail ablation study evaluates three strategies on an adversarial red-team set:

- **Model Only:** Uses only the trained guardrail head
- **Rules Only:** Uses only rule-based pattern matching
- **Combined:** Uses OR logic (model OR rules)

### 9.2. Results

| Layer      | Precision | Recall | F1    |
| ---------- | --------- | ------ | ----- |
| Model Only | 1.000     | 0.571  | 0.727 |
| Rules Only | 1.000     | 0.857  | 0.923 |
| Combined   | 1.000     | 1.000  | 1.000 |

### 9.3. Discussion

- The **combined approach** achieves perfect recall and precision (F1 = 1.000)
- The **combined approach** is recommended for production deployment
- The **model only** approach has lower recall, missing some unsafe queries
- The **rules only** approach has perfect precision but lower recall

---

## 10. Error Analysis

### 10.1. Misclassified Examples

| Query                                           | True Intent          | Predicted Intent     | Analysis                                              |
| ----------------------------------------------- | -------------------- | -------------------- | ----------------------------------------------------- |
| DAP or NPK different information                | cultivation_practice | nutrition_fertilizer | Fertilizer query misclassified as nutrition           |
| Information about mushroom cultivation ?        | cultivation_practice | general              | Cultivation query routed to general class             |
| Hame gende ka tel bechna hai kisse sampark kare | general              | cultivation_practice | Business query misclassified as agricultural practice |
| Information about variety of onion?             | disease_pest         | cultivation_practice | Variety query misclassified as cultivation            |
| Information about pre-emergence weed control... | disease_pest         | cultivation_practice | Weed control misclassified as cultivation             |

### 10.2. Root-Cause Analysis

1. **Class Boundary Confusion:** Errors occur between semantically close classes
2. **Ambiguous Queries:** Queries containing keywords from multiple categories
3. **Small Class Sizes:** Rare classes have poor performance due to insufficient training data
4. **Distant Supervision Noise:** Intent labels may contain errors
5. **Multilingual Challenges:** Code-switching can confuse the model

### 10.3. Guardrail Performance

- **False Positives:** 0
- **False Negatives:** 0

The guardrail shows perfect performance on the test set.

---

## 11. Model Robustness

### 11.1. Language-wise Performance

| Language | Count | Intent Accuracy | Intent Macro F1 |
| -------- | ----- | --------------- | --------------- |
| English  | 2,121 | 0.890           | 0.757           |
| Hinglish | 1,396 | 0.885           | 0.775           |
| Hindi    | 75    | 0.867           | 0.894           |

The model maintains consistent performance across all three languages.

### 11.2. Edge Cases

| Input       | Case              | Model Prediction     |
| ----------- | ----------------- | -------------------- |
| ""          | Empty string      | disease_pest         |
| "wheat"     | Single crop word  | cultivation_practice |
| "a" \* 500  | Very long input   | disease_pest         |
| "???"       | Meaningless input | disease_pest         |
| "123456789" | Numeric input     | cultivation_practice |

The model always makes a prediction, defaulting to `disease_pest` or `cultivation_practice` when uncertain.

---

## 12. Computational Performance

| Metric                    | Value                     |
| ------------------------- | ------------------------- |
| Single Query Mean Latency | 4.86 ms                   |
| Single Query p95 Latency  | 5.12 ms                   |
| Batch 32 Throughput       | 858.9 qps                 |
| Model Size                | 514.0 MB                  |
| Total Parameters          | 134.7 M                   |
| Training Time             | 949.4 seconds (~15.8 min) |

**Deployment Considerations:**

- **Fast Inference:** < 5ms per query suitable for real-time applications
- **Model Size:** 514 MB manageable for cloud deployment
- **Throughput:** 858 queries per second supports high-volume use cases

---

## 13. Limitations

### 13.1. Dataset Limitations

| Limitation          | Description                                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| Distant Supervision | Intent and NER labels from weak supervision                                    |
| Class Imbalance     | `general`, `post_harvest_storage`, `specialty_other` classes have few examples |
| Language Coverage   | Limited pure Hindi training data                                               |

### 13.2. Model Limitations

| Limitation       | Description                                          |
| ---------------- | ---------------------------------------------------- |
| Overfitting      | Validation loss increased after epoch 3              |
| Guardrail Scope  | Only covers dangerous queries                        |
| Default Behavior | Makes predictions even with no meaningful input      |
| Weak Performance | Poor performance on rare classes (general: 0.139 F1) |

### 13.3. Bias and Ethical Considerations

- **Regional Bias:** May not generalize to all agricultural contexts
- **Language Bias:** Performance may differ across languages
- **Guardrail Overreach:** False positives could prevent legitimate advice

---

## 14. Possible Improvements

### 14.1. Data Improvements

1. Address class imbalance with oversampling or class-weighted loss
2. Refine NER labels with human-annotated data
3. Expand non-agri dataset with more diverse queries
4. Collect more pure Hindi training data

### 14.2. Model Improvements

1. Explore alternative backbones (xlm-roberta-base, mBERT)
2. Use ensemble methods for better robustness
3. Implement cost-sensitive learning for guardrail
4. Add confidence thresholds and deferral system

### 14.3. Training Improvements

1. Use k-fold cross-validation
2. Explore better regularization techniques
3. Try cosine decay learning rate scheduling

---

## 15. Discussion

### 15.1. Objectives Achievement

| Objective                             | Result | Status      |
| ------------------------------------- | ------ | ----------- |
| Intent Classification (>85% accuracy) | 88.4%  | ✅ Exceeded |
| Intent Macro F1 (>0.70)               | 0.715  | ✅ Exceeded |
| NER Entity F1 (>0.90)                 | 0.958  | ✅ Exceeded |
| Guardrail F1 (>0.95)                  | 1.000  | ✅ Exceeded |

### 15.2. Practical Applicability

The model is computationally efficient and performs well on core tasks, making it suitable for real-time agricultural chatbot applications. The `non_agri` intent and guardrail head make the system safe and manageable for large-scale deployment.

### 15.3. Lessons Learned

1. Hyperparameter tuning significantly impacts performance
2. Class imbalance is a major challenge
3. Guardrail integration is critical for safety
4. Multilingual backbone provides strong performance

---

## 16. Conclusion

### 16.1. Summary of Findings

1. **Intent Classification:** 88.4% accuracy, 71.5% Macro F1
2. **Entity Extraction:** 95.8% Entity F1
3. **Guardrail:** Perfect performance (Precision=1.0, Recall=1.0)
4. **Multi-Lingual Robustness:** Consistent across languages
5. **Computational Efficiency:** < 5 ms per query

### 16.2. Final Performance Summary

| Metric          | Performance | Target | Status |
| --------------- | ----------- | ------ | ------ |
| Intent Accuracy | 88.4%       | > 85%  | ✅ Met |
| Intent Macro F1 | 71.5%       | > 0.70 | ✅ Met |
| NER Entity F1   | 95.8%       | > 0.90 | ✅ Met |
| Guardrail F1    | 100%        | > 0.95 | ✅ Met |

### 16.3. Readiness for Deployment

The model is **ready for pilot deployment** with the following considerations:

1. Monitor guardrail performance in production
2. Implement deferral system for low-confidence predictions
3. Collect more data for rare intent classes
4. Implement active learning for continuous improvement

---

## Appendices

### Appendix A: Complete Metric Tables

| Metric              | Baseline | Fine-tuned |
| ------------------- | -------- | ---------- |
| Intent Accuracy     | 0.371    | 0.884      |
| Intent Macro F1     | 0.111    | 0.715      |
| NER Entity F1       | 0.036    | 0.958      |
| Guardrail Precision | 0.000    | 1.000      |
| Guardrail Recall    | 0.000    | 1.000      |
| Guardrail F1        | 0.000    | 1.000      |

### Appendix B: Hyperparameter Search Results

| Run ID            | Optimizer | LR   | Batch | Epochs | WD   | LS  | Val Score |
| ----------------- | --------- | ---- | ----- | ------ | ---- | --- | --------- |
| adamw_lr3e-5_5ep  | AdamW     | 3e-5 | 32    | 5      | 0.01 | 0.0 | **1.747** |
| adamw_lr3e-5_bs32 | AdamW     | 3e-5 | 32    | 3      | 0.01 | 0.0 | 1.713     |
| adamw_labelsmooth | AdamW     | 3e-5 | 32    | 3      | 0.01 | 0.1 | 1.713     |
| adamw_lr3e-5_bs64 | AdamW     | 3e-5 | 64    | 3      | 0.01 | 0.0 | 1.700     |
| adamw_lr1e-5_bs32 | AdamW     | 1e-5 | 32    | 3      | 0.01 | 0.0 | 1.667     |
| sgd_lr1e-3_bs32   | SGD       | 1e-3 | 32    | 3      | 0.01 | 0.0 | 1.279     |

### Appendix C: Sample Predictions

| Query                                          | Intent               | Entities    | Flag |
| ---------------------------------------------- | -------------------- | ----------- | ---- |
| mere tamatar ke paudho me keede lag gaye hain  | disease_pest         | ['tamatar'] | 0    |
| dhaan ki fasal me urea kitna daale             | disease_pest         | ['dhaan']   | 0    |
| gehu ki buwai ka sahi samay kya hai            | cultivation_practice | ['gehu']    | 0    |
| should I use a double dose of pesticide?       | disease_pest         | []          | 1    |
| is monocrotophos safe for my crops?            | non_agri             | []          | 1    |
| how to apply for a loan for my tractor?        | non_agri             | []          | 0    |
| who is the current agriculture minister of UP? | nutrition_fertilizer | []          | 0    |

### Appendix D: Model Configuration

```json
{
  "model_name": "IEGModel",
  "backbone": "distilbert-base-multilingual-cased",
  "n_intents": 7,
  "n_ner": 3,
  "selected_run_id": "adamw_lr3e-5_5ep",
  "hyperparameters": {
    "optimizer": "adamw",
    "learning_rate": 3e-5,
    "batch_size": 32,
    "epochs": 5,
    "weight_decay": 0.01,
    "label_smoothing": 0.0
  },
  "intent_classes": [
    "cultivation_practice",
    "disease_pest",
    "general",
    "non_agri",
    "nutrition_fertilizer",
    "post_harvest_storage",
    "specialty_other"
  ]
}
```
