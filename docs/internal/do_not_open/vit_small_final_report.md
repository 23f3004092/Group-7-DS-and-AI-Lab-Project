# Crop Disease Classification — ViT-S/16 Training Pipeline
## Final Report: Data Integrity, Augmentation Policy, Split Construction, Training, and Evaluation

**Component:** Computer-vision module of the Multi-Source Agentic RAG framework (Milestone 1)
**Backbone:** `vit_small_patch16_224.augreg_in21k_ft_in1k` (timm 1.0.26)
**Environment:** Kaggle — 2 × Tesla T4 (15.6 GB, sm_75), 4 vCPU, 33.7 GB RAM
**Status:** complete — three training phases run, single held-out test evaluation performed
**Notebook:** `vit-train-01.ipynb`, cells A1 → E1
**Headline test result:** 20-way macro-F1 **0.8671** [0.8396, 0.8905], accuracy **0.8998**, n = 1,287

---

## 0. Purpose and Framing

This report documents everything done **before** a single gradient step of supervised fine-tuning. That ordering is deliberate. On a 12,859-image dataset assembled from three separate public sources, the dominant risk is not optimisation — it is that the model learns something real about the *dataset* and nothing about *plant pathology*, and that our evaluation set is too contaminated to notice.

Every decision below was made against a measurement, not an intuition. Where a hypothesis was tested and turned out to be wrong, that is recorded too, because a negative result that cost a cell of compute is still a result worth citing in the write-up.

The work divides into four phases:

| Phase | Cells | Question answered |
|---|---|---|
| **A** — Data integrity | A1–A5 | Is the data what the manifest claims, and is anything leaking? |
| **B** — Augmentation policy | B1, B1b, B1c | Which non-pathological cues can predict the label, and can we destroy them? |
| **C** — Baseline & split | B2, C1–C4 | What does the frozen backbone already know, and is our val set honest? |
| **B3/B4** — Pipeline engineering | B3, B3b-1…4, B4 | Can we feed the GPU fast enough, without weakening the augmentation? |

---

## 1. Dataset

Twenty classes across two crops: 5 rice diseases and 15 wheat conditions (13 diseases, 1 pest-damage class, 1 healthy). All images pre-processed to 256×256 RGB JPEG by an upstream preparation step. Three source datasets are merged:

- `rice_s1_nirmalsankalana`
- `rice_s2_vbookshelf`
- `wheat_kushagra3204`

![Dataset composition and exclusion funnel](fig1_dataset.png)

The imbalance is severe and it is not gradual — it is one catastrophic outlier plus a mild gradient. `rice__leaf_smut` has **25 training images**; `wheat__yellow_rust` has 1,081. That is 43:1, and it means every per-class metric for leaf_smut sits on a sample small enough that a single misclassification moves F1 by more than 0.1. This constraint shapes several later decisions.

---

## 2. Phase A — Data Integrity

### 2.1 Cell A1 — Structural reconciliation

**What we did.** Verified the on-disk file tree against `master_manifest.csv` and `label_to_idx.json`, then fully decoded every one of the 12,859 files to catch truncation (with `LOAD_TRUNCATED_IMAGES = False`, so truncation raises rather than silently returning a partial image).

**Why.** A count mismatch or a truncated JPEG discovered at epoch 40 costs a training run. Discovered here it costs 90 seconds.

**Result — completely clean:**

| Check | Result |
|---|---|
| Class directories identical across splits | ✅ 20/20, matching `label_to_idx` keys |
| Manifest ↔ on-disk count reconciliation | ✅ zero delta on all 60 (split, class) cells |
| Geometry | ✅ 12,859 / 12,859 exactly 256×256 |
| Colour mode | ✅ 12,859 / 12,859 RGB |
| Extension vs true format | ✅ 12,859 `.jpg` → JPEG (no mislabelled PNGs) |
| Zero-byte / unreadable / truncated | ✅ 0 |

### 2.2 Cell A2 — Leakage scan and letterbox geometry

**What we did.** In a single pass: MD5 of every file, a dependency-free 64-bit perceptual hash (2-D DCT, top-left 8×8 block, thresholded at the median with the DC term dropped), and a constant-colour bar detector.

**Why pHash rather than MD5 alone.** Exact duplicates are the easy case. The dangerous case is the *same photograph* re-encoded, re-cropped by a few pixels, or saved at a different quality — byte-identical hashing misses these entirely, and they leak just as effectively.

**Results:**

| Check | Result |
|---|---|
| Cross-split exact (MD5) duplicates | 0 ✅ |
| Within-split exact duplicates | 8 |
| Near-duplicates, pHash Hamming ≤ 5, train↔val | 8 ⚠️ |
| Near-duplicates, train↔test | 10 ⚠️ |
| Near-duplicates, val↔test | 1 ⚠️ |
| Manifest `group_id` spanning >1 split | 0 ✅ |

The upstream `group_id` field was doing its job for exact grouping, but 19 near-duplicate pairs crossed a split boundary regardless. This was the first hint that the published split could not be taken at face value.

The constant-bar detector reported a mean padding fraction of **0.24%** across the training split, with only 25 images above 40%. That number turned out to be badly wrong, which A3b resolves.

### 2.3 Cell A3 — Adjudicating the near-duplicates

**What we did.** Recomputed pHashes retaining pair indices, rendered every one of the 19 cross-split pairs side by side for visual adjudication, and tabulated them by Hamming distance and label agreement.

**Findings.**

- The 8 within-split exact duplicates concentrate in two classes: `wheat__black_rust` (2) and `wheat__blast` (12 rows / 6 pairs). Within-split duplication inflates the effective weight of those samples but does not leak, so they were left in place.
- Of the 19 cross-split near-duplicate pairs, **4 are cross-label** — e.g. `train/rice__leaf_smut/leaf_smut_0092.jpg` at Hamming 2 from `val/rice__brown_spot/brown_spot_0022.jpg`. Two near-identical images carrying different labels is either a labelling error or a pHash collision on near-blank content. Visual inspection showed the latter: most cross-label pairs involve heavily padded images where the perceptual hash is dominated by the padding, not the leaf.
- `rice__tungro` accounts for 8 of the 19 pairs on its own — a strong signal of burst photography that reappears decisively in C2.

The high-padding subset (n=62) had fill colours clustered near **0** (black) with a handful near **255** (white), confirming these are genuine letterbox bars — but only for that small subset.

### 2.4 Cell A3b — The padding measurement was wrong

**What we did.** Replaced the constant-colour bar detector with a robust one that also detects **replicated rows** (a row identical to its neighbour within tolerance), then re-ran across the whole training split.

**Why this matters.** A preprocessing pipeline that pads with `replicate` or `reflect` produces bars that are *not uniform in colour* — they are copies of the edge pixels. A detector looking for flat colour sees texture and reports "no padding" while a third of the frame is synthetic.

**Result — the headline finding of Phase A:**

> Mean training-split padding: **0.24% → 21.80%**, a 90× correction.

![Padding by source and by class](fig2_padding.png)

| Source dataset | mean pad % | median | max |
|---|---|---|---|
| `rice_s1_nirmalsankalana` | 7.3 | 0.4 | 55.2 |
| `rice_s2_vbookshelf` | **74.7** | 77.4 | 91.4 |
| `wheat_kushagra3204` | 24.0 | 25.0 | 93.8 |

Per class, `rice__leaf_smut` is **71.3% padding on average** (median 76.3%, and *zero* images with no padding). Every leaf_smut image is roughly three-quarters synthetic pixels.

**Why this is a shortcut, not just a cosmetic issue.** If one class is systematically 71% replicate-padded and the others are not, a convolutional or attention model can identify that class from the *statistics of the padding region* without ever looking at the leaf. It would score well on val and fail completely on a field photograph. This finding is what motivated the entire augmentation audit in Phase B.

### 2.5 Cell A4 — Aspect-ratio forensics

**What we did.** Reconstructed the implied original aspect ratio from the detected content box, then attempted to locate the raw pre-256 source images for direct comparison.

**Findings.**

- The dataset is overwhelmingly square: 99.5% of images have implied AR < 1.05:1. Only 31 images (0.24%) exceed 1.5:1, and 14 have a content dimension below 64 px — degenerate slivers, almost all `wheat__brown_rust`.
- **The decisive negative result:** all 120 `rice_s2_vbookshelf` images measure *exactly* 1.00:1 under the constant-bar detector, despite their originals being known wide leaf strips. Combined with A3b's 74.7% robust padding for that source, this establishes that the preparation step **did not letterbox with black bars — it padded by replication**, which the naive detector cannot see.
- No raw sources were mounted in the session, so the source-vs-prepared comparison could not be completed. This is a documented gap, not a resolved question.

### 2.6 Cell A5 — Exclusions and training statistics

**Exclusion policy, and the reasoning behind each rule:**

1. **Degenerate slivers** — content dimension < 64 px *or* padding > 60%. At that point the leaf occupies too few real pixels to carry a diagnosis; the image is mostly synthetic. **19 removed.**
2. **Cross-split near-duplicates** — remove the **eval-side** member, never the train-side. Removing a training image loses signal; removing an eval image only shrinks the test set slightly. Pairs where either member was already a sliver were skipped, since those matches are padding artefacts rather than true duplicates. **17 removed.**

**Total: 36 images (0.28%).** Splits became train 10,261 / val 1,281 / test 1,281.

One consequence flagged immediately: `rice__leaf_smut` val dropped from 8 to **7 images**, below the `MIN_EVAL = 8` floor. Recorded as a known weakness rather than papered over.

**Per-channel statistics, computed on the training split only** (never on val or test — that would be a subtle form of leakage):

| | R | G | B |
|---|---|---|---|
| Dataset mean | 0.3530 | 0.3835 | 0.2540 |
| Dataset std | 0.2782 | 0.2850 | 0.2388 |
| ImageNet mean | 0.485 | 0.456 | 0.406 |

Maximum absolute deviation from ImageNet: **0.152 in mean, 0.061 in std**. The dataset is markedly darker and greener than ImageNet — large enough that "which normalisation?" became an empirical question rather than a default, tested in B2.

**Class-conditional image statistics — the shortcut inventory:**

| Class | mean brightness | mean sharpness (Laplacian var) |
|---|---|---|
| `rice__leaf_smut` | **0.2196** (darkest) | 0.0132 |
| `rice__brown_spot` | **0.4357** (brightest) | 0.0024 |
| `wheat__healthy` | 0.2782 | 0.0381 |
| `rice__blast` | 0.4119 | **0.0022** (blurriest) |
| `wheat__healthy` / `wheat__fusarium_head_blight` | — | **0.038** (sharpest) |

Two exploitable non-pathological cues emerged:

1. **Brightness identifies `rice__leaf_smut`** — it is the darkest class in the dataset by a clear margin.
2. **Sharpness separates rice from wheat** — the rice sources are systematically blurrier (Laplacian variance spanning a 25× range across classes).

A third hypothesis, that `wheat__mildew` was brightness-separable, was **tested and rejected**: z = −0.28, well inside normal variation.

---

## 3. Phase B — Augmentation Policy and Shortcut Destruction

The governing principle: an augmentation policy is only defensible if you can *measure* that it destroys the shortcut it was chosen to destroy. Each candidate policy was scored with the same two metrics, both AUCs where **0.50 means the cue carries no information about the label** and 1.00 means it is perfectly predictive.

![Shortcut destruction across four policy versions](fig3_shortcuts.png)

### 3.1 Cell B1 — Baseline policy (v1)

Geometric + photometric stack: `RandomResizedCrop(224, scale 0.65–1.0, ratio 0.80–1.25, bicubic)` → `HFlip(0.5)` → `RandomAffine(15°, translate 0.08, scale 0.92–1.08)` → `ColorJitter(0.35, 0.30, 0.25, hue 0.02)` → `GaussianBlur(p=0.35, σ 0.1–1.6)` → `RandomAdjustSharpness(2.5, p=0.35)` → `ToTensor` → `Normalize` → `RandomErasing(p=0.15)`.

The **two-sided sharpness randomisation** (blur *and* sharpen, each at p=0.35) is the deliberate part: one-sided blurring would shift every class in the same direction and preserve the ordering. Randomising in both directions collapses the ordering.

**Results:**

| Audit | Before | After v1 |
|---|---|---|
| Sharpness → rice-vs-wheat (AUC) | 0.833 | **0.604** |
| Per-class sharpness spread (max/min) | 25.0× | **3.7×** |
| leaf_smut brightness gap | +0.1183 | +0.1046 |
| Brightness → leaf_smut (AUC) | 0.821 | **0.776** ❌ |

The sharpness shortcut was substantially destroyed. The brightness shortcut barely moved.

**Diagnosis.** `ColorJitter(brightness=0.35)` samples a *multiplicative* factor in [0.65, 1.35] with expectation 1.0. It widens the brightness distribution of every class equally but cannot move any class's *mean* toward any other. The between-class gap survives untouched. **A multiplicative jitter is structurally incapable of destroying an absolute-exposure shortcut.**

### 3.2 Cell B1b — `RandomExposure` (v2), and why it fell short

Introduced a transform that rescales each image so its mean luminance is a fresh random draw from a target range — targeting the mean directly rather than jittering around it.

**Results:** bright_auc 0.821 → 0.771 (v1) → **0.718** (v2). Real progress, but the prediction was 0.50–0.58. Two causes, both diagnosed from the numbers rather than guessed:

**Cause 1 — `p = 0.7`.** 30% of draws are a no-op, so 30% of pairs retain the original gap. Working through the mixture: 0.49 of pairs get both images randomised (AUC 0.5), 0.09 get neither (0.821), 0.42 are mixed. That alone predicts ≈ 0.58.

**Cause 2 — `clamp_(0,1)` is asymmetric, and this is the real culprit.** Check `rest_mean`: 0.3423 against a target-range midpoint of 0.35 — bright images landed exactly where intended, because they need a scale factor near 1.0 and barely clip. Now `smut_mean`: expected 0.314, observed **0.2703**. Backing out the achieved mean gives ~0.288, not 0.35. Scaling a 0.25-mean image to 0.50 requires factor 2.0, which drives its bright pixels into the ceiling; the clamp eats the top and the mean never arrives.

> **The dark class structurally cannot reach the bright end of the target range.** Multiplicative scaling with a clamp is not exposure adjustment — it is exposure adjustment *plus a highlight blowout that correlates with how dark the image started*, which is exactly the signal we were trying to erase.

### 3.3 Cell B1c — `RandomExposureGamma` (v3), adopted

Replaced multiplicative scaling with a **power-law tone curve**, `x → x^γ`:

```python
class RandomExposureGamma(torch.nn.Module):
    """Monotone and bijective on [0,1] => no clipping, so a dark image
    can actually reach the bright end. γ found by bisection on a strided
    pixel subsample; mean(x**γ) is monotone DECREASING in γ."""
    def __init__(self, target=(0.18, 0.55), p=0.9, iters=10, eps=1e-3, stride=31):
```

Three properties make this work where v2 failed:

1. **Monotone and bijective on [0,1]** — no clipping is possible, so no exposure-correlated highlight destruction.
2. **Order-preserving** — pixel rankings are unchanged, so lesion boundaries and relative contrast survive; only the tone curve moves.
3. **γ found by bisection** (10 iterations) on a strided subsample (every 31st pixel), making the search cost ~2.5 ms/image rather than a full-image optimisation.

Raised `p` from 0.7 to 0.9, closing the no-op leak from cause 1.

Also introduced an **exposure-normalised sharpness metric**, `Laplacian_var / mean²`, because raw Laplacian variance scales with overall exposure — brightening an image inflates its apparent sharpness. Without normalisation, the exposure transform would appear to fix the sharpness shortcut as a side effect, which would be an artefact of the metric rather than a real result.

**Results:**

| Config | sharp_auc | sharpn_auc (normalised) | sharpn_ratio | bright_auc | bright_gap | smut_mean | rest_mean |
|---|---|---|---|---|---|---|---|
| base (no aug) | 0.833 | 0.803 | 37.6× | 0.821 | 0.1183 | 0.2500 | 0.3682 |
| v2 (mult+clamp) | 0.617 | 0.621 | 5.5× | 0.745 | 0.0760 | 0.2632 | 0.3392 |
| **v3 (gamma)** | 0.611 | **0.595** | **4.4×** | **0.552** | **0.0207** | **0.3460** | 0.3667 |

The brightness shortcut is destroyed: AUC 0.821 → 0.552, and `smut_mean` (0.3460) now sits within 0.02 of `rest_mean` (0.3667). The class that was the darkest in the dataset by a wide margin is no longer distinguishable by brightness.

Both shortcuts identified in A5 are now measured as neutralised. **This is the primary methodological result of the work so far.**

---

## 4. Phase C — Frozen Baseline and Split Construction

### 4.1 Cell B2 — Normalisation selection by linear probe

**What we did.** Extracted 384-d features from the frozen ViT-S/16 backbone under **three** normalisation candidates simultaneously (one forward pass per candidate, features cached to disk as float16), then trained a linear head on each and compared.

**Why a probe rather than a judgement call.** The checkpoint's native statistics are `mean = std = (0.5, 0.5, 0.5)` — AugReg-style, *not* classic ImageNet. Our dataset statistics differ from ImageNet by 0.152 in mean. Three defensible answers existed, so we measured instead of arguing.

**Grid (frozen backbone, 60 epochs, val split):**

| norm | balance | macro-F1 | acc | best_ep | min class F1 |
|---|---|---|---|---|---|
| **native (0.5)** | **none** | **0.8461** | 0.8564 | 13 | 0.507 |
| native | sqrt | 0.8453 | 0.8587 | 26 | 0.521 |
| imagenet | none | 0.8326 | 0.8470 | 47 | 0.496 |
| dataset | none | 0.8274 | 0.8470 | 14 | 0.478 |
| imagenet | sqrt | 0.8236 | 0.8423 | 57 | 0.460 |
| dataset | sqrt | 0.8216 | 0.8423 | 32 | 0.429 |

**Decisions taken:**

- **Native (0.5, 0.5, 0.5) normalisation adopted**, beating dataset statistics by +1.9 F1 points. The intuition that "matching your own dataset statistics is better" is wrong here — matching the *pretraining* statistics preserves more of the transferred representation than matching the target domain does.
- **No class balancing at the probe stage.** `sqrt` sampling was neutral-to-negative on macro-F1 in every pairing, though it did raise the minimum per-class F1 slightly. Revisit at fine-tuning, where the dynamics differ.
- Majority-class accuracy floor: 0.105. The probe is far above chance.

**Immediate red flag.** `rice__leaf_smut` scored **0.923** with only 24 training images, while `wheat__tan_spot` scored 0.507 with 517. A class with 20× less data scoring 0.4 higher is not a success — it is a symptom. This drove C1.

### 4.2 Cell C1 — Probe diagnostics

Four investigations:

**(1) Val-set size per class.** `rice__leaf_smut` has 7 val images; `wheat__stem_fly` has 17. Any per-class F1 computed on 7 samples has a resolution of roughly ±0.15 — the leaf_smut number is close to meaningless.

**(2) Feature-space nearest-neighbour leakage.** For each val image, cosine similarity to its nearest training neighbour:

| Threshold | val images | % |
|---|---|---|
| > 0.90 | 423 | 33.0% |
| > 0.95 | 171 | **13.4%** |
| > 0.98 | 36 | 2.8% |
| > 0.995 | 0 | 0.0% |

Per class, `rice__tungro` has a **mean** NN similarity of 0.9682 — its average val image is nearly identical to some training image. The top pairs are consecutive filenames (`tungro_01163` ↔ `tungro_01162`, `blast_01465` ↔ `blast_01470`), the unmistakable signature of burst photography of the same physical leaf.

**(3) Is crop identity free?** A binary rice-vs-wheat probe on frozen features scores **macro-F1 0.9835, accuracy 0.9906**. Crop identity is essentially free from ImageNet-pretrained features. The nominal 20-way problem is therefore really *a solved 2-way problem plus a 5-way and a 15-way problem*, and any 20-way macro-F1 is inflated by the free part.

**(4) The honest sub-problems:**

| Sub-problem | classes | train / val | macro-F1 |
|---|---|---|---|
| rice | 5 | 1,740 / 219 | **0.9642** |
| wheat | 15 | 8,521 / 1,062 | **0.8017** |

This reframed the entire project: **wheat-15 is the real task.** Rice is nearly solved by a frozen backbone.

**Confusion structure** — the errors are not scattered, they form one tight cluster:

| true → pred | n | % of true class |
|---|---|---|
| `leaf_blight` → `tan_spot` | 10 | 18.2% |
| `tan_spot` → `common_root_rot` | 10 | 15.4% |
| `common_root_rot` → `tan_spot` | 6 | 10.5% |
| `mite` → `tan_spot` | 6 | 8.0% |
| `smut` → `blast` | 6 | 12.0% |

A `{tan_spot, leaf_blight, common_root_rot}` confusion triangle dominates. These are all necrotic-lesion wheat foliar conditions that are genuinely hard to separate visually — this is a real pathology problem, not an artefact.

### 4.3 Cell C2 — Scene components and the scale of contamination

**What we did.** Built a similarity graph over all 12,823 images (cosine on frozen features), took connected components at two thresholds, and treated each component as a **scene** — a set of photographs of the same physical subject.

**Results at cos > 0.95:** 2,392 edges → 10,934 components, 9,609 singletons, largest component 23 images.

| Class | images | scenes | scenes/image | largest cluster |
|---|---|---|---|---|
| `rice__tungro` | 461 | 207 | **0.449** | 16 |
| `rice__bacterial_blight` | 554 | 320 | 0.578 | 6 |
| `rice__blast` | 477 | 277 | 0.581 | 12 |
| `rice__brown_spot` | 645 | 397 | 0.616 | 6 |
| `wheat__yellow_rust` | 1,351 | 1,348 | 0.998 | 2 |

`rice__tungro`'s 461 images are only **207 distinct scenes** — each scene photographed roughly twice. The rice sources are burst-captured; the wheat sources are not.

**The contamination number:**

> At cos > 0.95, **391 eval images (15.3% of val+test) share a scene with an image in another split.** At cos > 0.98, 95 images (3.7%).

**Direct measurement of the inflation:**

| Eval subset | n | 20-way F1 | wheat | rice |
|---|---|---|---|---|
| all | 1,281 | 0.8358 | 0.6627 | 0.6942 |
| sim ≤ 0.98 | 1,245 | 0.8349 | 0.6627 | 0.6927 |
| sim ≤ 0.95 | 1,110 | 0.8244 | 0.6582 | 0.6839 |
| sim ≤ 0.90 | 858 | **0.7649** | 0.6123 | 0.7550 |

Progressively removing near-duplicates from the eval set monotonically lowers the score. That monotone relationship is what a contaminated evaluation looks like.

*A methodological note worth recording:* the 0.8358 here differs from B2's 0.8461 because B2 selected the best epoch **on the val set**. Selecting the checkpoint on the same split you report is worth about +0.01 of free inflation. Later cells use a fixed 60-epoch schedule.

### 4.4 Cell C3 — Group-aware re-split (v2), and the trap it fell into

**What we did.** Discarded the published split entirely and rebuilt it with connected components as **atomic units** — every image in a scene goes to the same split, so a scene cannot span a boundary by construction. Assignment was greedy on per-class deficit, processing **largest groups first**. 20 components spanning multiple labels were assigned by dominant label.

**Contamination: eliminated.**

| Check | Result |
|---|---|
| Groups spanning >1 split (cos > 0.95) | **0** |
| Groups spanning >1 split (cos > 0.98) | **0** |
| Val images with train NN cos > 0.95 | **0** |
| Test images with train NN cos > 0.95 | **0** |
| Val NN-to-train cosine, median | 0.8605 |

**But the baseline went up, not down:**

| Split | 20-way | wheat-15 | rice-5 |
|---|---|---|---|
| original | 0.8358 | 0.7953 | 0.9718 |
| **v2** | **0.8723** | **0.8509** | 0.9441 |

Removing leakage should *lower* a score. It rose by +3.7 points. That contradiction is what C4 exists to resolve — and taking the 0.8723 at face value would have been the single biggest error available in this project.

### 4.5 Cell C4 — Diagnosing the trap, and the corrected split (v3)

**The diagnosis.** "Largest groups first" packed the biggest clusters into val and test to hit the count targets quickly. The result:

| v2 val | value |
|---|---|
| Nominal size | 1,284 images |
| Distinct scenes | 1,080 |
| **Effective size** | **84% of nominal** |
| `rice__tungro` | 46 images in **19** scenes; largest cluster = 30.4% of the class |
| `wheat__blast` | 58 images in 43 scenes; largest cluster 17.2% |

An eval set where 30% of a class is one repeated scene is *easier* than a diverse one: getting that scene right earns 14 correct predictions. **The v2 split traded leakage for concentration** — it removed cross-split contamination while creating within-eval redundancy that inflated the score by a different mechanism.

**Cluster-level uncertainty.** Image-level bootstrap treats every eval image as independent, which is false when 14 of them are the same leaf. Resampling *groups* instead:

| Metric | F1 | image CI | **cluster CI** | dedup (1 img/scene) |
|---|---|---|---|---|
| 20-way | 0.8723 | ±0.0206 | **±0.0232** | 0.8726 ± 0.0020 |
| wheat-15 | 0.8509 | ±0.0216 | **±0.0224** | 0.8443 ± 0.0021 |
| rice-5 | 0.9441 | ±0.0491 | **±0.0617** | 0.9649 ± 0.0037 |

The rice-5 interval is ±0.06 — that metric can barely distinguish anything.

**The fix: reverse the packing order.** Fill val and test from **singletons and smallest groups first**, leaving multi-image clusters for train, where redundancy is harmless (it only reweights the loss slightly).

![Split comparison and evaluation integrity](fig4_splits.png)

**v3 results:**

| | v2 val | **v3 val** | v2 test | **v3 test** |
|---|---|---|---|---|
| Images | 1,284 | 1,284 | 1,286 | 1,287 |
| Scenes | 1,080 | **1,283** | 1,087 | **1,285** |
| Effective size | 84% | **100%** | 85% | **100%** |
| Worst-class largest cluster | 30.4% | **16.7%** | 23.5% | **12.5%** |
| Cross-split groups (0.95 / 0.98) | 0 / 0 | **0 / 0** | — | — |

**v3 is 100% effective and zero-contaminated simultaneously.** Final sizes: **train 10,252 / val 1,284 / test 1,287.**

**The baseline we will actually report:**

| Metric | macro-F1 | cluster 95% CI | dedup |
|---|---|---|---|
| 20-way | **0.8382** | [0.8136, 0.8577] | 0.8382 ± 0.0000 |
| **wheat-15** | **0.7966** | [0.7698, 0.8188] | 0.7966 ± 0.0000 |
| rice-5 | 0.9717 | [0.9115, 0.9965] | 0.9717 ± 0.0000 |
| accuracy | 0.8559 | | |

The dedup standard deviation is exactly 0.0000 because there is essentially one image per scene — nothing left to resample. That is the arithmetic confirmation that the eval set is now clean.

**Per-class frozen baseline on v3:**

![Per-class F1 on the v3 split](fig5_perclass.png)

The three targets are unambiguous:

| Class | F1 | train n | Why |
|---|---|---|---|
| `wheat__tan_spot` | **0.407** | 516 | Centre of the necrotic-lesion confusion triangle. Not a data-quantity problem. |
| `wheat__black_rust` | 0.582 | 252 | Confusable with brown/yellow rust; moderate data. |
| `wheat__leaf_blight` | 0.607 | 452 | The other arm of the tan_spot triangle. |

Note the counter-intuitive pair: `rice__leaf_smut` scores 0.923 on **25 training images** while `wheat__tan_spot` scores 0.407 on 516. Data volume is not the binding constraint — **class separability is.** (The leaf_smut number rests on 6 val images and should not be over-interpreted; one error moves it by ~0.15.)

---

## 5. Phase B3/B4 — Pipeline Engineering

### 5.1 Cell B3 — `pipeline.py`

Froze the data path into an importable module: `RandomExposureGamma`, `build_transforms()`, `CropDS`, `make_loader()`. Writing it to a file rather than leaving it in notebook cells means the training cells import exactly the object that was audited — no risk of a redefined cell silently changing the transform mid-experiment.

The eval transform is `CenterCrop(224)` on a 256 image = crop_pct 0.875, deliberately matching the B2 probe so that the frozen baseline and the fine-tuned model are evaluated identically.

### 5.2 Cell B3b-1 — Per-stage CPU profiling

![Per-stage cost and epoch-time comparison](fig6_throughput.png)

Full train transform: **11.146 ms/image**, i.e. 90 img/s per core.

| Stage | ms/img | % of full |
|---|---|---|
| **ColorJitter with hue=0.02** | **5.692** | **51.1%** |
| ColorJitter without hue | 2.044 | 18.3% |
| GaussianBlur (at p=1.0) | 3.847 | 34.5% |
| RandomExposureGamma (i10, s31) | 2.533 | 22.7% |
| RandomAffine | 2.070 | 18.6% |
| RandomAdjustSharpness | 2.014 | 18.1% |
| RRC bicubic | 1.281 | 11.5% |
| RRC bilinear | 1.033 | 9.3% |
| jpeg decode | 0.467 | 4.2% |
| ToTensor | 0.328 | 2.9% |
| RandomErasing | 0.208 | 1.9% |

> **A hue jitter of ±0.02 — roughly 7° of hue rotation, visually almost nothing — costs 3.65 ms/image, a third of the entire pipeline.** torchvision implements it as a full RGB→HSV→RGB round-trip in Python-level tensor operations. This is the worst cost-to-effect ratio in the chain by a wide margin.

### 5.3 Cell B3b-2 — GPU ceiling

| batch size | img/s | s/epoch (GPU-bound) | peak memory |
|---|---|---|---|
| 32 | 341 | 30.0 | 1.6 GB |
| 64 | 360 | 28.5 | 2.7 GB |
| **96** | **373** | **27.4** | 3.8 GB |
| 128 | 368 | 27.9 | 4.9 GB |

Throughput plateaus by bs=96 and 15.8 GB of T4 memory is never close to a constraint. **bs=64 retained** — it is within 3% of peak throughput and gives 160 optimiser steps per epoch versus 106 at bs=96, which matters more on a 10k-image fine-tune than 3% of wall clock.

### 5.4 Cell B3b-3 — Loader throughput

A decoded uint8 cache (2.02 GB, built in 23 s) replaced repeated JPEG decoding, and worker counts were swept.

| transform | workers | img/s | s/epoch |
|---|---|---|---|
| full_tf | 2 | 139 | 73.5 |
| full_tf | 3 | 153 | 67.2 |
| full_tf | 4 | 188 | 54.6 |
| fast_tf | 2 | 205 | 50.0 |
| fast_tf | 3 | 225 | 45.6 |
| fast_tf | 4 | **263** | **38.9** |

**We are CPU-bound in every configuration** — the GPU can consume 373 img/s and the best loader supplies 263.

*A process note:* an earlier version of this benchmark ran for over an hour before being interrupted. Cause: `persistent_workers=True` on every benchmarked loader without teardown, leaving ~22 worker processes competing for 4 vCPUs, plus the 2 GB decoded array being built twice. The rewritten version splits the work into three cells with explicit `_shutdown_workers()` and RAM logging between stages. Recorded because it is a real and easily repeated failure mode.

### 5.5 Cell B3b-4 — The veto test

Before adopting any speed-up, a pre-registered check: **does trimming augmentation cost us the shortcut kills we spent three cells earning?** Three variants compared on both cost and both AUCs:

| variant | ms/img | img/s | s/epoch | bright_auc | smut_mean | rest_mean | sharp_auc |
|---|---|---|---|---|---|---|---|
| `full_tf` (bicubic + hue + gamma i10/s31) | 11.51 | 188 | 54.5 | 0.549 | 0.3466 | 0.3666 | 0.584 |
| **`mid_tf` (hue removed only)** | **8.41** | **243** | **42.1** | **0.560** | 0.3443 | 0.3673 | 0.605 |
| `fast_tf` (bilinear + no hue + gamma i8/s61) | 7.63 | 263 | 39.0 | 0.558 | 0.3447 | 0.3672 | 0.581 |

Two-point fit on the measured anchors: `wall_ms/img = 0.82 + 0.391 × cpu_ms` — about 2.6 effective cores with ~0.8 ms/img of fixed IPC overhead, implying a hard ceiling near 1,220 img/s that we are nowhere near.

**Three conclusions:**

1. **The v3 result replicated independently.** `bright_auc = 0.549` here, measured with different sampling from B1c's 0.552. The brightness shortcut is genuinely dead, not an artefact of one sample.
2. **Hue was pure cost.** 3.10 ms/image (27% of the pipeline) and all three AUCs are indistinguishable — the spread across variants is inside the ±0.02 noise floor.
3. **A hypothesis was tested and refuted.** The coarser gamma search (`iters=8, stride=61`) was flagged as risky, since a coarser bisection should undershoot the target mean on dark images. It does not: `smut_mean` is 0.3447 under `fast_tf` versus 0.3466 under `full_tf`. A difference of 0.002 on a quantity we moved by 0.07 means the bisection had already converged well before iteration 8. Clean negative result, worth recording.

**Decision: adopt `mid_tf`.** `fast_tf` is 3.1 s/epoch faster — roughly 3 minutes across the entire project — but its bilinear resampling and coarser exposure search touch something the audit does not measure: whether fine lesion texture survives. `wheat__tan_spot` at 0.407 is the class this whole effort exists to fix, and interpolation quality is not worth trading for three minutes on an argument the audit cannot support either way.

### 5.6 Cell B4 — Patch and verification

Two changes to `pipeline.py`, applied by **appending a wrapper** rather than editing the original function, so the audited object and the shipped object provably cannot diverge:

1. **Hue removed from `ColorJitter`** — the single provably-free change.
2. **`CropDS` switched to a contiguous uint8 array.** Not for speed (decode is only 4%), but for fork behaviour: a Python list of byte-strings has each object's refcount touched on every `__getitem__`, which dirties the memory page and defeats copy-on-write, duplicating the 2 GB cache across all four workers. A numpy block that is never written stays genuinely shared.

**Verification results — all checks passed:**

| Check | Result |
|---|---|
| `ColorJitter` hue | ✅ `None`; b=(0.65,1.35) c=(0.7,1.3) s=(0.75,1.25) |
| Cache build | 27 s; train 2.02 GB (10252, 256, 256, 3), val 0.25 GB |
| Steps per epoch @ bs=64 | 160 |
| Batch shape / dtype | (64, 3, 224, 224) float32, labels 0–19 |
| Train batch stats | mean −0.2820, std 0.5582, range [−1.000, +1.000] |
| Val batch stats | mean −0.2044, std 0.5193 |
| Eval transform deterministic | ✅ True |
| Train transform stochastic | ✅ True |
| **Throughput** | **235 img/s → 43.6 s/epoch** (predicted 243 / 42.1) |
| Copy-on-write behaviour | ✅ RAM 8.1 → 7.9 GB during 4-worker iteration (no per-worker duplication) |

Measured throughput landed within 3% of the model's prediction, and the RAM trace confirms the cache is shared rather than copied four times.

---

## 6. Current State

### 6.1 Frozen decisions

| Decision | Value | Justified by |
|---|---|---|
| Backbone | `vit_small_patch16_224.augreg_in21k_ft_in1k` | — |
| Normalisation | native (0.5, 0.5, 0.5) | B2 grid, +1.9 F1 over dataset stats |
| Class balancing (probe) | none | B2 grid; revisit at fine-tuning |
| Train transform | `mid_tf` (v3 gamma, no hue) | B3b-4 veto test |
| Eval transform | CenterCrop(224), crop_pct 0.875 | matches B2 probe |
| Split | `clean_index_v3.csv` | C4 — 0% contamination, 100% effective |
| Batch size | 64 | B3b-2; 160 steps/epoch |
| Workers | 4 | B3b-3 sweep |
| **Primary selection metric** | **wheat-15 macro-F1** | C1 — rice is nearly free |

### 6.2 Numbers to beat

| Metric | Frozen baseline | Cluster 95% CI |
|---|---|---|
| **wheat-15 macro-F1** | **0.7966** | [0.7698, 0.8188] |
| 20-way macro-F1 | 0.8382 | [0.8136, 0.8577] |
| rice-5 macro-F1 | 0.9717 | [0.9115, 0.9965] |
| accuracy | 0.8559 | — |
| `wheat__tan_spot` F1 | 0.407 | — |

Any improvement smaller than about **±0.025** on wheat-15 is inside the confidence interval and must not be claimed as a result.

### 6.3 Artefacts on disk

| File | Contents |
|---|---|
| `pipeline.py` | Transforms, dataset, loader — the frozen data path |
| `clean_index_v3.csv` | 12,823 rows: `path` (physical) + `split` (logical v3) + `y` + scene IDs |
| `split_plan_v3.csv` | Per-class targets vs achieved |
| `exclusions.csv` | 36 removed images with reason |
| `geometry_hashes.csv`, `phash.npy` | MD5 + pHash + letterbox geometry |
| `near_dup_pairs.csv`, `padding_robust.csv` | Phase-A evidence |
| `meta_with_components.csv` | Scene assignments at both thresholds |
| `aug_config.json` | Full augmentation config + all audit tables |
| `probe_results.json`, `probe_diagnostics.json` | B2 grid, C1 diagnostics |
| `feats/X_native_{train,val,test}.npy` | Cached 384-d frozen features |

### 6.4 Known limitations

These are recorded honestly rather than smoothed over:

1. **Scene definition is mildly circular.** Groups are defined by cosine similarity in the *same* frozen backbone we are about to fine-tune. It is the standard approach and errs conservative (over-grouping costs a little eval data, never leaks), but it is not an independent measurement.
2. **`rice__leaf_smut` remains statistically hollow** — 25 train / 6 val / 8 test. Its F1 has a resolution of roughly ±0.15. It should be reported with the count attached, always.
3. **Raw source images were never located**, so the padding correction in A3b is inferred from bar structure rather than confirmed against originals.
4. **Padding is not augmented away.** We destroyed the brightness and sharpness shortcuts; we did **not** destroy the 71%-padding signature of `rice__leaf_smut`. `RandomResizedCrop(scale 0.65–1.0)` partially disrupts it, but this remains an untested exposure.
5. **Train/eval exposure distributions differ by design** — train batches centre at −0.282 in normalised space, val at −0.204, because eval images are not exposure-randomised. This is intentional (the training distribution is strictly wider and covers eval), but worth watching if a train/val gap appears that the loss curves do not explain.
6. **The upstream preparation step is a black box.** We characterised its output thoroughly but never saw its code.

---

## 7. Training — Three Phases

The schedule was progressive: freeze everything, then unfreeze the top quarter, then unfreeze all with regularisation. Each phase changed **one** structural thing so that any gain was attributable.

Selection metric throughout: **wheat-15 macro-F1** on val, chosen in C1 because a frozen backbone already separates rice from wheat at 0.9835 F1, making the 20-way number partly free.

| Phase | Trainable | Config | Epochs | Time |
|---|---|---|---|---|
| **D1** — head only | 7.7K (0.04%) | frozen backbone, lr 1e-3, cosine | 12 | 9 min |
| **D2** — blocks 9–11 | 5.33M (25%) | lr 1e-4 backbone / 1e-3 head, 2-epoch warmup, grad-clip 1.0 | 20 | 15 min |
| **D3** — full | 21.67M (100%) | LLRD 0.70 (base 3e-4), drop-path 0.10, 1-epoch warmup | 18 | 15 min |

### 7.1 Phase 1 — the shortcut confirmation

Phase 1 exists to validate the loop and to produce a controlled comparison against the C4 frozen probe. The probe was trained on **un-augmented** features; phase 1 is the first run where the shortcut-destroying augmentation is active at training time. Everything else is identical.

| Metric | probe (C4) | phase 1 | delta |
|---|---|---|---|
| 20-way | 0.8382 | 0.7655 | **−0.0727** |
| wheat-15 | 0.7966 | 0.7682 | −0.0284 |
| rice-5 | 0.9717 | 0.7887 | −0.1830 |
| accuracy | 0.8559 | 0.8131 | −0.0428 |

The drop is expected and was predicted in advance. Two mechanisms:

**(1) The shortcut prediction was confirmed, decisively.**

> `rice__leaf_smut`: **0.923 → 0.320**, delta −0.603.

The C4 probe scored a class with 25 training images higher than classes with twenty times more data, and that anomaly was flagged in advance as a symptom rather than a success. Under exposure randomisation the class collapsed. **The 0.923 was essentially the brightness shortcut in full.** This single class accounts for 41% of the 20-way drop.

Without the B1c gamma transform, that model ships, reports 0.92 on leaf_smut, and fails on the first field photograph taken in daylight.

**(2) A second open risk closed in the good direction.** Every leaf_smut image is ~71% replicate padding (A3b), and we never destroyed that signature — it was listed as the top open exposure. If padding were carrying the class, leaf_smut would have held near 0.92 after brightness was neutralised. It did not. **Padding alone is not sufficient to identify the class.**

**(3) Frozen backbone + heavy augmentation underperforms structurally.** The head trains on augmented features and is evaluated on clean ones; because the backbone cannot update, augmentation cannot teach invariance — it only scatters the cloud the head must fit. The easy classes confirm this reading: `yellow_rust` −0.013 and `tungro` −0.011 barely moved, while marginal classes absorbed the loss.

Train/val 0.860 / 0.813 — a healthy 0.047 gap, no overfitting. Val plateaued from epoch 5.

### 7.2 Phase 2 — unfreezing blocks 9–11

![Phase 2 training dynamics](fig7_phase2_curves.png)

| Metric | probe | phase 1 | **phase 2** | vs p1 | vs probe |
|---|---|---|---|---|---|
| 20-way | 0.8382 | 0.7655 | **0.8830** | +0.1175 | +0.0448 |
| **wheat-15** | 0.7966 | 0.7682 | **0.8660** | +0.0978 | **+0.0694** |
| rice-5 | 0.9717 | 0.7887 | 0.9429 | +0.1542 | −0.0288 |
| accuracy | 0.8559 | 0.8131 | 0.9042 | +0.0911 | +0.0483 |

+0.069 on wheat-15 over the frozen probe, against a cluster CI of ±0.025 — nearly three times the noise floor. Adapting the top three blocks was the correct intervention.

**Both diagnostic classes answered:**

| | probe | p1 | **p2** | reading |
|---|---|---|---|---|
| `wheat__tan_spot` | 0.407 | 0.306 | **0.569** | the confusion triangle *is* separable with adapted features |
| `rice__leaf_smut` | 0.923 | 0.320 | **0.769** | recovered *without* the shortcut — the class has real learnable features |

The leaf_smut arc is the strongest single result in the project: phase 1 proved the original score was a shortcut, phase 2 proved the class is nonetheless learnable from actual pathology. Both halves were needed to make either claim.

![Per-class progression across phases](fig8_perclass_progress.png)

The shape of the gains matters — the four broken classes absorbed nearly everything, and only `rice__tungro` moved backwards (−0.010, noise). That is what a genuine capacity increase looks like, as opposed to a metric drifting up uniformly.

**But phase 2 ended memorising.** Train accuracy 0.9943; train loss 0.6515 against a label-smoothing floor of ~0.62. The generalisation gap widened from 0.047 to 0.090, and val stopped moving at epoch 12 — epoch 18 beat epoch 12 by 0.0004, pure noise selection. This diagnosis is what determined the phase 3 configuration.

### 7.3 Phase 3 — full fine-tune with regularisation

![Phase 3 analysis](fig9_phase3.png)

Two coupled changes, justified as a matched pair: unfreezing everything is the capacity increase, and drop-path is the counterweight that makes it safe. Layer-wise LR decay at 0.70 gave block 11 ≈ 2.1e-4, block 9 ≈ 1.0e-4 (continuous with phase 2, so the warm start was undisturbed), and the patch embedding ≈ 2.9e-6 — effectively frozen, protecting the generic edge and texture detectors that ImageNet learned from a million images.

**Drop-path did its job:** the gap held at **0.073** while trainable parameters went from 5.3M to 21.7M (phase 2 ended at 0.092 with a quarter of the parameters).

| Metric | probe | p1 | p2 | **p3** | vs p2 |
|---|---|---|---|---|---|
| 20-way | 0.8382 | 0.7655 | 0.8830 | **0.9042** | +0.0212 |
| **wheat-15** | 0.7966 | 0.7682 | 0.8660 | **0.8793** | +0.0133 |
| rice-5 | 0.9717 | 0.7887 | 0.9429 | 0.9828 | +0.0399 |
| accuracy | 0.8559 | 0.8131 | 0.9042 | 0.9143 | +0.0101 |

**The phase 2 → phase 3 gain is not statistically significant.** +0.0133 on wheat-15 sits inside the ±0.025 cluster CI. By the rule pre-registered in C4, this is reported as *comparable to phase 2*, not better. Phase 3 won on val and is therefore the selected checkpoint, which is legitimate — but the defensible claim is the **cumulative** one: **+0.083 wheat-15 over the frozen baseline, 3.3× the noise floor.**

Epochs 15, 16 and 17 produced identical val metrics to four decimal places — cosine drove the LR to zero and predictions froze. The run was fully converged; more epochs would have bought nothing.

**The confusion triangle gave the pre-registered bad answer.** `leaf_blight` +0.069 but `tan_spot` −0.013 — one arm rose while the other did not. The model shifted the boundary between them rather than learning to separate them. Combined they gained 0.056, so it is not pure redistribution, but full fine-tuning did **not** fix tan spot.

---

## 8. Test Set Evaluation

Run **once**, on the checkpoint selected purely by val wheat-15. The test split was constructed in C4, verified contamination-free, and never touched by any decision in this project.

![Val vs test generalisation](fig10_test.png)

| Metric | val (selection) | **test (held out)** | cluster 95% CI | delta |
|---|---|---|---|---|
| **20-way macro-F1** | 0.9042 | **0.8671** | [0.8396, 0.8905] | −0.037 |
| **wheat-15 macro-F1** | 0.8793 | **0.8631** | [0.8403, 0.8853] | **−0.016** |
| rice-5 macro-F1 | 0.9828 | 0.8823 | [0.7915, 0.9523] | −0.100 |
| accuracy | 0.9143 | **0.8998** | — | −0.014 |

**The primary metric generalised.** wheat-15 fell 0.016 from val to test — well inside the ±0.023 interval. On a 15-class problem with 1,062 test images, that is a clean result and it means the val-driven model selection did not overfit the val split.

**The rice-5 drop of 0.100 is one class, not a trend.** Four of five rice classes score ≥ 0.97 on test (tungro 1.000, bacterial_blight 0.991, blast 0.990, brown_spot 0.970). The entire deficit is `rice__leaf_smut` at 0.4615 on **8 test images**.

**Per-class test results:**

| Class | test F1 | n_test | | Class | test F1 | n_test |
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

**Sixteen of twenty classes exceed 0.80. Eleven exceed 0.94.** Three sit below 0.65, and they are the same three the diagnostics predicted from the start.

![Test confusion matrix and per-class F1](fig11_test_confusion.png)

### 8.1 The error structure

The top-12 test confusions are not scattered — they concentrate almost entirely in one cluster of wheat foliar conditions:

| true → predicted | n | % of true class |
|---|---|---|
| `leaf_blight` → `mildew` | 7 | 12.3% |
| `leaf_blight` → `tan_spot` | 7 | 12.3% |
| `tan_spot` → `common_root_rot` | 7 | 10.8% |
| `tan_spot` → `leaf_blight` | 6 | 9.2% |
| `tan_spot` → `mite` | 6 | 9.2% |
| `common_root_rot` → `mite` | 4 | 7.0% |
| `aphid` → `mite` | 4 | 4.7% |
| **`leaf_smut` → `tan_spot`** | **3** | **37.5%** |

The C1 diagnostic on frozen features identified a `{tan_spot, leaf_blight, common_root_rot}` triangle. Test data shows it has grown into a five-node cluster: **`{tan_spot, leaf_blight, common_root_rot, mite, mildew}`**. All are wheat conditions producing necrotic or chlorotic foliar lesions, and they are genuinely hard to separate visually. This is a pathology problem, not a data artefact.

**One error deserves separate attention.** Three of eight `rice__leaf_smut` test images were predicted as `wheat__tan_spot` — a **cross-crop** error, in a model that separates rice from wheat at 0.98 F1 nearly everywhere else. Given that leaf_smut images are ~71% replicate padding, the most plausible explanation is that once brightness is neutralised, the remaining signal in a heavily padded rice image resembles a padded wheat lesion image more than it resembles other rice photographs. This is worth an explicit probe before any deployment.

---

## 9. Findings

**1. Shortcut destruction worked, and it was necessary.** Two non-pathological cues were identified by measurement in A5, and both were driven to near-chance by the B1c policy (brightness AUC 0.821 → 0.549; exposure-normalised sharpness 0.803 → 0.584). The `rice__leaf_smut` collapse from 0.923 to 0.320 the moment training used the shortcut-free pipeline is direct evidence that the original score was an artefact rather than skill.

**2. Evaluation-set construction mattered more than any modelling choice.** The published split leaked 15.3% of eval images into training scenes. The first repair (v2) removed the leakage but created a different bias — an eval set only 84% effective, with 30% of one class being a single repeated scene, which *raised* the score by +3.7 points. Only the corrected v3 split is simultaneously zero-contaminated and 100% effective. **Two different mechanisms would each have inflated the reported number, and both were invisible without explicit measurement.**

**3. Progressive unfreezing is where the gain lives — and it saturates.** Phase 1 → 2 delivered +0.098 wheat-15. Phase 2 → 3 delivered +0.013, inside the noise floor. On a 10K-image dataset, adapting the top three transformer blocks captures nearly all of the available improvement; full fine-tuning adds a fourfold parameter increase for no measurable return.

**4. Class separability, not data volume, is the binding constraint.** `rice__leaf_smut` reaches 0.92 on val with 25 training images; `wheat__tan_spot` reaches 0.61 on test with 516. Collecting more tan spot images is unlikely to help — the classes overlap in appearance at 224px.

**5. Regularisation must scale with unfrozen capacity.** Phase 2 ended at a 0.092 generalisation gap with 5.3M trainable parameters. Phase 3 held 0.073 with 21.7M, purely because drop-path was added alongside the unfreeze.

---

## 10. Limitations

Recorded honestly; several are inherent to the dataset and cannot be fixed by more training.

**1. `rice__leaf_smut` is statistically hollow.** 25 train / 6 val / 8 test. Its F1 swung 0.833 → 0.471 → 0.769 → 0.923 across consecutive phase-3 epochs, then landed at 0.462 on test. **Any single number for this class should be reported with n attached and treated as uninformative.** The val-test swing of 0.46 is the clearest possible demonstration of why.

**2. The necrotic-lesion cluster is unresolved.** `tan_spot` 0.613, `leaf_blight` 0.585 — below 0.65 after full fine-tuning. Three candidate remedies, none attempted: higher input resolution (384px, which changes the throughput picture entirely), a segmentation-first stage isolating lesions before classification (the STAR-Net approach), or accepting that these conditions are not separable at this resolution from this data.

**3. No field-domain validation.** This dataset is curated and pre-processed. Published results show CNNs dropping from ~99% on lab datasets to ~73% in the wild. Nothing here measures that gap, and **0.8671 on this test split should not be quoted as expected field accuracy.**

**4. The padding signature was never destroyed.** ~71% replicate padding on leaf_smut, and 21.8% dataset-wide. Phase 1 showed it is not sufficient to carry a class, but the cross-crop `leaf_smut → tan_spot` errors suggest it may still be distorting the representation.

**5. Shortcut invariance was measured on the augmentation, not the trained model.** The B-series audits proved the *pipeline* destroys the cues. No probe confirms the *final model* ignores them. Three tests were scoped and not run for time: exposure sweeps at inference, padding occlusion, and attention rollout on tan_spot errors.

**6. No calibration.** The model outputs uncontrolled softmax scores. For a system that routes disease predictions into treatment recommendations, a confidently wrong output is the exact failure mode the RAG design guards against elsewhere.

**7. Scene grouping is mildly circular** — defined by cosine similarity in the same frozen backbone later fine-tuned. Standard practice, errs conservative, but not an independent measurement.

---

## 11. Recommended Next Work

In priority order, by value per hour:

**1. Shortcut re-probe on the trained model (~20 min).** Closes the loop from Phase B. Feed test images at altered exposure and measure prediction stability; occlude the padding region and re-score; run attention rollout on tan_spot errors to check whether the model attends to lesions or background. A model that scores 0.867 *and* demonstrably ignores exposure is a substantially stronger claim than one that merely scores 0.867.

**2. Temperature scaling and an abstention threshold (~15 min).** Fit temperature on val, produce a coverage–risk curve. This gives the RAG router a principled trigger: below the confidence threshold, the system asks a clarifying question about leaf symptoms instead of asserting a diagnosis — connecting the CV module to the multi-turn dialogue component rather than leaving it a standalone classifier.

**3. A small field-image probe (~1 hour, mostly collection).** Even 50 hand-collected or scraped in-field images across a few classes would convert "0.867 on our test split" into a defensible statement about deployment behaviour.

**4. Inference artifact.** A `predict()` callable — transform plus `p3_full_best.pt` plus label map — returning top-k labels with calibrated confidence and an abstain flag.

**5. Only if tan spot must improve:** 384px input, or a segmentation-first stage. Both are substantially more expensive than everything above and neither is required for the milestone.

---

## Appendix — Reproduction

| Artefact | Path |
|---|---|
| Frozen data path | `pipeline.py` |
| Split (12,823 rows, physical `path` + logical `split`) | `clean_index_v3.csv` |
| Selected model | `runs/vits16_m1/ckpt/p3_full_best.pt` (epoch 16) |
| Phase checkpoints | `p1_head_best.pt`, `p2_blocks9_11_best.pt` |
| Training curves | `runs/vits16_m1/logs/{phase}_history.csv` |
| Per-class curves | `runs/vits16_m1/logs/{phase}_per_class_f1.csv` |
| Test results | `test_results.json`, `test_results.png` |
| Augmentation config + all audit tables | `aug_config.json` |
| Phase-A evidence | `exclusions.csv`, `geometry_hashes.csv`, `near_dup_pairs.csv`, `padding_robust.csv` |

**Note on the family metrics.** `wheat-15` and `rice-5` are computed by masking to rows whose **true** label is in that family, so a wheat image misclassified as a rice class is not counted as a false positive against rice. The convention is consistent across C3, C4, D1–D3 and E1, so all deltas are valid — but the family scores will not reconcile arithmetically with the per-class table, and readers should be told why.

**Total GPU time for all training and evaluation: ~42 minutes.** The overwhelming majority of the effort in this project went into establishing that the number at the end means what it says.
