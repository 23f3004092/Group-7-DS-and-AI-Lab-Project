# Notebook E — Training-Pipeline Design (Augmentation, Normalization, Imbalance, Robustness)

**Stage:** Milestone 3 (design documented now, in Milestone 2)
**Input:** the final prepared dataset from Notebook D — 12,859 images, 20 classes, 256×256 letterboxed, split into train/val/test with `label_to_idx.json`.
**Scope of this document:** what happens to the data *at training time*, why each step exists, and exactly how it will run. None of these steps modify the saved dataset — they run live inside the data loader / training loop.

---

## 1. Why these steps are train-time, not preprocessing

The prepared dataset holds **one clean, deterministic copy per image**. The steps below are deliberately *not* baked into it, for three reasons:

- **Augmentation must be random every epoch** — the model should see a different transformed version of each image on every pass; baking it in would freeze that randomness.
- **Some steps apply only to the training split** — validation and test must stay clean, or evaluation becomes dishonest.
- **Imbalance handling depends on the final split counts** — which only existed after Notebook D.

So this is a pipeline definition, not a data-writing step.

---

## 2. Step 1 — Resize/Crop + Normalization

This is the deterministic backbone applied to **every** image (all splits).

**What it does:**
- Images arrive at 256×256 (already letterboxed in preprocessing).
- **Train:** a random 224×224 crop (adds mild positional variation).
- **Val/Test:** a fixed 224×224 **center** crop (deterministic — no randomness in evaluation).
- **All splits:** convert to tensor, then normalize with **ImageNet mean/std** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

**Why:**
- 224×224 is the native input size for our chosen backbone (EfficientNet-B0), which is ImageNet-pretrained — so we match both the resolution and the normalization statistics the backbone was trained on, or transfer learning degrades.
- Cropping from 256→224 (rather than resizing) gives a small "free" augmentation on the train split while keeping val/test reproducible.

```python
# illustrative
train_tf = Compose([RandomCrop(224), ..., ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
eval_tf  = Compose([CenterCrop(224),      ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
```

---

## 3. Step 2 — Data Augmentation (train split only)

**What it does:** applies random transforms to each training image, re-rolled every epoch, so the model learns disease features rather than memorizing exact photos. Val/test get **none** of this.

**Baseline augmentation (all training classes):**
- Random horizontal + vertical flip (leaves have no inherent orientation)
- Random rotation (±15–20°)
- Color jitter (brightness / contrast / saturation) — simulates the lighting variation of real farmer phone photos
- The random 224 crop from Step 1
- Optionally: mild Gaussian blur / random erasing for robustness to occlusion and focus

**Rationale — the lab-to-field gap.** The core project risk is that a model trained on clean images collapses on messy real-world phone photos. Augmentation is the primary defense: by showing the model flipped, rotated, differently-lit versions, it's forced to generalize to conditions it will actually face in a farmer's field.

**Heavier augmentation for the rare / risky classes** (this is where two deferred issues get addressed):
- **`leaf_smut` (40 images)** — strongest augmentation, since it has the fewest unique samples. Heavy augmentation multiplies the *effective* variety the model sees.
- **`rice__tungro`** — heavier **background-oriented** augmentation (aggressive random crops, color/background jitter) specifically to attack its background/framing shortcut (see Step 4).
- **`wheat__stem_fly` (172)** — elevated augmentation as the smallest wheat class.

---

## 4. Step 3 — Class-Imbalance Handling

After the split, the training set ranges from `leaf_smut` (24) and `stem_fly` (138) up to `yellow_rust` (1,081) — roughly a 45:1 spread, driven by `leaf_smut`. Left unhandled, the model becomes biased toward majority classes: overall accuracy looks fine while rare classes quietly get poor recall. There are **two standard approaches**, and they are not mutually exclusive.

### Approach A — Weighted Loss (class-weighted cross-entropy)

**What:** each class gets a weight in the loss function, inversely proportional to its frequency, so mistakes on rare classes are penalized more heavily.

```python
# weights computed from the TRAIN split only
weights = total_train / (num_classes * per_class_train_counts)
criterion = CrossEntropyLoss(weight=weights.to(device))
```

**Pros:** one line; doesn't change how data flows; every image still seen once per epoch; interacts cleanly with augmentation.
**Cons:** for *extreme* minorities (leaf_smut at 24), simply up-weighting 24 images can make the loss spiky and the gradient noisy — the model still only *sees* 24 distinct leaf_smut images per epoch.

### Approach B — Weighted Sampler (WeightedRandomSampler)

**What:** rebalances each *batch* by sampling rare-class images more often, so the model sees roughly balanced classes per batch.

```python
sample_weights = 1.0 / per_class_train_counts[train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train), replacement=True)
loader  = DataLoader(train_ds, sampler=sampler, ...)
```

**Pros:** directly fixes what the model *sees*; much better for extreme minorities — leaf_smut appears far more often per epoch.
**Cons:** oversampling means the same few leaf_smut images repeat many times per epoch → **this only works well paired with strong augmentation**, or the model overfits those exact copies. It also changes what "one epoch" means (some images seen many times, others rarely).

### Recommendation

**Use both, in a light combination:** a weighted sampler to lift the extreme minorities (`leaf_smut`, `stem_fly`) toward visibility, *plus* a mild weighted loss to smooth the rest — and crucially, the heavier per-class augmentation from Step 2 so the oversampled rare images aren't just memorized. This is the standard robust recipe for a long-tailed dataset with one very small class. We'll start here and tune based on the **per-class recall** we see on the validation set (not overall accuracy).

**Evaluation metric note:** because of the imbalance, the model is judged on **macro-F1 + per-class recall + a confusion matrix**, never plain accuracy — plain accuracy would hide rare-class failure.

---

## 5. Step 4 — Tungro Robustness Check

This addresses the single most important shortcut flagged across the whole EDA.

**The problem (recap).** In rice Set 1, `Tungro` was photographed differently from every other class: soil background, whole-plant framing. Preprocessing removed the *dimension* half of the shortcut (letterbox to 256²), but the *background/framing* half survives — a CNN can still "cheat" by learning "soil background → Tungro" instead of the actual viral yellowing symptoms. A model doing this shows great validation accuracy but **collapses when a farmer photographs Tungro against green foliage**, or labels any soil photo as Tungro.

**What the check does:**
1. **Explicit robustness test set** — evaluate Tungro recall specifically on Tungro images against *non-soil* backgrounds (and check non-Tungro images on soil aren't misclassified as Tungro). If accuracy holds there, the shortcut is beaten.
2. **Saliency / Grad-CAM inspection** — visualize where the model "looks." If the heatmap lights up the soil/background instead of the leaf lesions for Tungro, that's the shortcut in action.

**Mitigation if it fails** (escalating):
- Heavier background augmentation for Tungro (already staged in Step 2): aggressive random crops that cut out background, background randomization, strong color jitter.
- If still failing: **segmentation/masking** — use a UNet or simple thresholding to mask out the background so only the leaf remains before classification. This is the "probably necessary" intervention flagged in the project summary.

This check runs *after* the first training run — it's a diagnostic, not a training step — and determines whether the segmentation work is needed.

---

## 6. Steps Remaining Before Final Training (the bridge checklist)

Once the four steps above are designed, here's everything that still stands between now and a full, trustworthy training run. In order:

1. **Publish the final dataset** — save Notebook D's `final/` as a Kaggle Dataset (`crop-disease-prepared-256`) and attach it read-only to Notebook E. *(Data side is then fully frozen.)*

2. **Implement + smoke-test the input pipeline** — write the actual `Dataset`/`DataLoader` with the transforms above, load `label_to_idx.json`, and **visualize a batch of augmented images** to confirm they look right (not over-distorted, labels correct, normalization sane). Cheap step, catches most pipeline bugs.

3. **Instantiate the model + transfer-learning setup** — load ImageNet-pretrained EfficientNet-B0, replace the classifier head with a 20-class head, decide the freeze/unfreeze schedule (typically: train the head first, then unfreeze the backbone at a lower learning rate).

4. **Wire up loss, optimizer, scheduler** — class-weighted loss + sampler from Step 3, an optimizer (e.g. AdamW), and a learning-rate schedule. Compute class weights **from the train split only** (a common leakage mistake is using full-dataset counts).

5. **Sanity checks before the long run:**
   - **Overfit a tiny batch** — a correct pipeline should drive loss to ~0 on a handful of images; if it can't, something's broken.
   - **Confirm the split is respected** — no val/test images leaking into the train loader.
   - **Verify per-class weights** and that the sampler produces balanced batches.

6. **Short baseline run** — a few epochs end-to-end to confirm the whole thing trains, loss decreases, and val macro-F1 moves. This validates the pipeline before committing to a long run.

7. **Set up evaluation reporting** — macro-F1, per-class precision/recall, and a confusion matrix on val (and finally test). This is the dashboard the rest of training is tuned against.

8. **Then: full training run** (Milestone 3 proper) — followed by the Tungro robustness check (Step 4) and any resulting mitigation.

**Summary of what's frozen vs. pending:** the *dataset* is done and immutable after step 1. Steps 2–7 are pipeline/model wiring and validation — no more data changes. Only after the baseline run (step 6) and its checks do we launch the real training.

---

## 7. Summary

Notebook E turns the frozen 12,859-image dataset into a live training pipeline: deterministic resize+normalize for all splits, random augmentation for training only (heavier for `leaf_smut`, `tungro`, `stem_fly`), imbalance handled by a weighted-sampler + weighted-loss combination backed by strong augmentation, and a post-training Tungro robustness check with segmentation as the fallback. Everything here is applied on-the-fly — the prepared dataset itself is never modified. The bridge checklist in Section 6 is the remaining path to a full training run.
