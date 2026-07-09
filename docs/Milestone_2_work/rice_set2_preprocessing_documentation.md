# Rice Set 2 — Preprocessing Documentation

**Dataset:** `vbookshelf/rice-leaf-diseases` (Kaggle)
**Notebook:** C (rice Set 2), run on Kaggle
**Goal:** Standardize the small, clean rice Set 2 dataset to the shared 256×256 format and normalize its labels, so it can merge with wheat and rice Set 1 for a single crop-disease classifier. This dataset uniquely contributes the **`leaf_smut`** class.

---

## 1. Overview

Rice Set 2 is the smallest and cleanest of the three sources — 120 images, 3 disease classes at exactly 40 each. The EDA confirmed it is genuinely pristine: zero duplicates, zero corruption, all RGB, perfectly balanced. Its only real characteristic worth handling is an **extreme panoramic shape** (very wide, ~3.43:1 leaf strips), which must be resized carefully so lesions aren't distorted.

Its main value in the merged model is coverage: it is the **only source of the `leaf_smut` class**, and it adds a handful more `bacterial_blight` and `brown_spot` samples on top of rice Set 1.

**Headline result:** 120 raw images → **120 clean images** (no cleaning needed — the work was label normalization + aspect-preserving resize).

---

## 2. Before → After (at a glance)

| Property | Before preprocessing | After preprocessing |
|----------|---------------------|---------------------|
| Total images | 120 | **120** (unchanged — pristine) |
| Classes | 3 | 3 (unchanged) |
| Class names | `Bacterial leaf blight`, `Brown spot`, `Leaf smut` | `rice__bacterial_blight`, `rice__brown_spot`, `rice__leaf_smut` |
| Per-class counts | 40 / 40 / 40 | 40 / 40 / 40 |
| Balance ratio | 1.00 (perfect) | 1.00 (perfect) |
| Duplicates | 0 | 0 |
| Corrupt images | 0 | 0 |
| Color mode | all RGB | all RGB |
| Image size / shape | wide panoramas (~3.43:1), 89% aspect > 2.5 | all 256×256, letterboxed |
| Split | none (all one folder) | `unassigned` (split deferred to merge notebook) |
| On-disk size | part of raw dataset | **~2 MB** (estimate — confirm with `du -sh`) |

---

## 3. Starting Point (Raw Dataset)

```
Total images: 120
Raw labels: ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
Per-class: rice__bacterial_blight 40 | rice__brown_spot 40 | rice__leaf_smut 40
```

Perfectly balanced, three classes, 40 images each.

---

## 4. Step-by-Step Preprocessing

### Step 1 — Label normalization

The three folder names were mapped to shared canonical rice keys (the same map used in Notebook B), so Set 1 and Set 2 use identical labels when merged:

| Raw folder | Canonical label |
|------------|-----------------|
| `Bacterial leaf blight` | `rice__bacterial_blight` |
| `Brown spot` | `rice__brown_spot` |
| `Leaf smut` | `rice__leaf_smut` |

`bacterial_blight` and `brown_spot` overlap with rice Set 1 (so they merge into the same classes); `leaf_smut` is new and appears only here.

---

### Step 2 — Integrity + geometry check

We verified the dataset's cleanliness and measured the aspect-ratio spread.

**Output:**
```
Corrupt: 0 | exact dup files: 0
Color modes present: {'RGB': 120}

Aspect-ratio spread:
 mean 3.31 | 25% 3.43 | 50% 3.43 | 75% 3.43 | min 1.25 | max 5.30

How many are extreme panoramas (aspect > 2.5)? 107 of 120
```

**What this told us.** Nothing to clean — 0 corrupt, 0 duplicates, all RGB. The only finding is geometric: **89% of images (107 of 120) are extreme panoramas** clustered tightly at ~3.43:1, with 13 near-square outliers (min aspect 1.25). This directly dictates the resize strategy.

---

### Step 3 — Letterbox resize + materialize

Each image was converted to RGB and **letterboxed to 256×256** — resized aspect-preserving, with padding added to fill the square — then written to disk with a manifest.

**Output:**
```
Written: 120 | failed: 0
rice__bacterial_blight 40 | rice__brown_spot 40 | rice__leaf_smut 40
```

**Why letterbox and not a plain square resize.** Squashing a 3.43:1 strip into a 256×256 square would compress it horizontally by ~3.4×, distorting lesion shapes — round brown spots would become ovals, streaks would change proportion. Letterboxing preserves the true shape by padding instead of stretching. The trade-off: because the strips are so wide, the leaf ends up occupying only the middle band of the 256×256 frame, with padding above and below. This is accepted for consistency with the other two datasets. (An alternative — tiling each strip into 2–3 square patches — was considered but not used; it adds complexity and changes counts, and would only be worth revisiting if Set 2 underperforms during modeling.)

---

## 5. Final Prepared Rice Set 2

**120 images, 3 classes, all 256×256 RGB, letterboxed.**

| Class | Count |
|-------|------:|
| rice__bacterial_blight | 40 |
| rice__brown_spot | 40 |
| rice__leaf_smut | 40 |

**Manifest schema (shared across all 3 datasets):**
`src_path, filename, label, source_dataset, split, group_id`

`split` is `unassigned` — splitting is deferred to the merge notebook. Every image is its own `group_id` (no duplicates to group).

---

## 6. Issues Faced (Summary)

| Issue | Severity | Resolution |
|-------|----------|------------|
| Extreme panoramic aspect ratio (~3.43:1, 89% of images) | Medium | Letterboxed to 256² (aspect-preserving, no distortion) |
| Duplicates / corruption / wrong color mode | — | None found — dataset is pristine |
| Small size (120 images, 40/class) | Note | Not a defect here, but see remaining work — `leaf_smut` scarcity affects the merged split |

---

## 7. What's Still Remaining for Rice Set 2

Cleaning is complete (there was almost none to do); the following are deferred:

1. **`leaf_smut` scarcity — to be addressed in the merge notebook.** With only 40 `leaf_smut` images, a standard 80/10/10 split leaves ~4 images each for validation and test — thin but workable. The merged notebook will decide how to handle this (accept it, oversample `leaf_smut`, or adjust the split ratio for tiny classes), and `leaf_smut` will get heavier augmentation + class weighting at train time. This is a documented known limitation.

2. **Train / val / test split — deferred to Notebook D.** Done once, centrally, on the merged 20-class pool so it's stratified across all classes at once (`split = unassigned` for now).

3. **Augmentation & normalization — deferred to training.** Applied live in the training loader; only one clean copy per image is saved here.

4. **Publishing to Kaggle — next action.** Save `prepared/rice_s2/` as a Kaggle Dataset (e.g. `rice-s2-cleaned-256`).

---

## 8. Next Step

Publish `prepared/rice_s2/`, then move to **Notebook D — merge + split**: attach all three cleaned datasets, concatenate the manifests into one 20-class pool (~12,859 images), and perform the single group-aware, stratified train/val/test split — the step where `leaf_smut`'s scarcity and wheat's `stem_fly` (172) are finally handled so every class lands safely in all three splits with no leakage.
