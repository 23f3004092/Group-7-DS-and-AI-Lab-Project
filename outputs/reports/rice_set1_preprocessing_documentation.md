# Rice Set 1 — Preprocessing Documentation

**Dataset:** `nirmalsankalana/rice-leaf-disease-image` (Kaggle)
**Notebook:** B (rice Set 1), run on Kaggle
**Goal:** Clean and deduplicate the rice Set 1 dataset, normalize its labels to shared rice class names, and produce a uniformly-sized 256×256 image set ready to merge with wheat and rice Set 2 for a single crop-disease classifier.

---

## 1. Overview

Rice Set 1 looked large and clean on the surface — 5,932 images, 4 balanced disease classes, zero corruption. But the EDA had already exposed two hidden problems: the dataset was built from **burst-captured photos** (many near-identical frames of the same leaf), and one class (**Tungro**) was photographed in a systematically different way, creating a "shortcut" a model could cheat on.

**Headline result:** 5,932 raw images → **2,066 clean, unique images**. The dataset didn't so much shrink as reveal its *true* size — about 65% of the raw images were near-duplicate burst frames.

---

## 2. Before → After (at a glance)

| Property | Before preprocessing | After preprocessing |
|----------|---------------------|---------------------|
| Total images | 5,932 | **2,066** |
| Classes | 4 | 4 (unchanged) |
| Class names | `Bacterialblight`, `Blast`, `Brownspot`, `Tungro` | `rice__bacterial_blight`, `rice__blast`, `rice__brown_spot`, `rice__tungro` |
| Per-class counts | 1,584 / 1,440 / 1,600 / 1,308 | 514 / 477 / 606 / 469 |
| Balance ratio | 1.22 (partly fake — inflated by duplication) | 1.29 (honest) |
| Near-duplicate frames | ~65% of images | 0 (thinned to 1 per group) |
| Exact duplicate files | 2,234 | 0 |
| Image size / shape | 78% at 300×300; Tungro different (~331², 4:3) | all 256×256, letterboxed |
| Color mode | 5,776 RGB + 156 RGBA | all RGB |
| Split | none (all one folder) | `unassigned` (split deferred to merge notebook) |
| Dimension shortcut (Tungro) | present | removed |
| On-disk size | ~part of the raw dataset | **37 MB** |

---

## 3. Starting Point (Raw Dataset)

```
Total images: 5932
Raw labels:
 Brownspot          1600
 Bacterialblight    1584
 Blast              1440
 Tungro             1308

Top filename prefixes (burst-capture signal):
 TUNGRO2  277 | TUNGRO4  277 | TUNGRO3  277 | TUNGRO1  277
 BACTERIALBLIGHT / BACTERIALBLIGHT1 / BACTERIALBLIGHT2 ...  264 each
```

The suspiciously uniform prefix counts (Tungro = four batches of exactly 277; Bacterialblight = many batches of exactly 264) are the fingerprint of burst capture — the same subject photographed repeatedly. This is what drives the heavy duplication we remove below.

---

## 4. Step-by-Step Preprocessing

### Step 1 — Label normalization (4 classes)

The four folder names were normalized to shared canonical rice keys, using an explicit mapping so that Set 1 and Set 2 end up with **identical** class names when merged:

| Raw folder | Canonical label |
|------------|-----------------|
| `Bacterialblight` | `rice__bacterial_blight` |
| `Blast` | `rice__blast` |
| `Brownspot` | `rice__brown_spot` |
| `Tungro` | `rice__tungro` |

The `rice__` prefix keeps rice's "Blast" distinct from wheat's "Blast" in the merged model — they're different crop-specific diseases and must not be collapsed.

**Output:**
```
Canonical rice classes: ['bacterial_blight', 'blast', 'brown_spot', 'tungro']
rice__bacterial_blight  1584
rice__blast             1440
rice__brown_spot        1600
rice__tungro            1308
```

---

### Step 2 — Duplicate detection (exact + near-duplicate)

We used the same method as the wheat notebook: MD5 for exact byte-duplicates, and a **perceptual hash (pHash)** with a Hamming-distance threshold of ≤ 6 bits to catch near-duplicate burst frames. A *union-find* algorithm then clustered every chain of near-duplicates into a single group with a `group_id`.

**Output:**
```
Exact byte-duplicate files: 2234 across 4794 unique blobs
Unreadable: 0

Total images        : 5932
Unique groups       : 2066   <-- TRUE effective size
Multi-image groups  : 1816 | largest cluster: 13

Cluster-size distribution:
 1: 250 | 2: 732 | 3: 587 | 4: 333 | 5: 42 | 6: 56
 7: 8   | 8: 21  | 9: 27  | 10: 6  | 13: 4

TRUE unique images per class (groups counted once):
 bacterial_blight 514 | blast 477 | brown_spot 606 | tungro 469

Groups spanning >1 label (probable mislabels): 0
```

**What this told us.** The 5,932 images are really only **2,066 unique subjects**, mostly in bursts of 2–4 near-identical frames. Importantly, the true per-class counts are still well balanced (ratio 1.29) and there were **zero mislabels** — so the duplication is clean, uniform burst capture, not a data-quality mess.

---

### Step 3 — Thinning the bursts

**The decision.** Rather than keep all 5,932 images and merely group them, we thinned each near-duplicate group down to **one representative frame** — the *sharpest* one (highest Laplacian variance). Reasons:

- Burst frames differ by ≤ 6 bits — same leaf, same background, same framing, just camera jitter. They add almost no learnable signal and mostly waste training compute.
- Real-world variation is better supplied by **train-time augmentation** than by near-identical copies.
- It improves the merged model's balance: thinned rice classes (469–606) sit neatly inside wheat's per-class range (172–1,351), instead of a few heavily-photographed leaves dominating.
- Thinning to one image per group also makes **leakage impossible** — nothing can straddle the train/test split.

**Output:**
```
Before thinning: 5932 | after: 2066
rice__bacterial_blight 514 | rice__blast 477 | rice__brown_spot 606 | rice__tungro 469
Imbalance ratio: 1.29
```

---

### Step 4 — Letterbox resize + materialize

Each kept image was converted to RGB (flattening the 156 RGBA files), **letterboxed to 256×256** (aspect-preserving, padded — no squashing), and written to disk with a manifest.

**Output:**
```
Near-blank dropped: 0 | failed: 0 | written: 2066
rice__bacterial_blight 514 | rice__blast 477 | rice__brown_spot 606 | rice__tungro 469
```

The letterbox step matters here specifically because it **removes the Tungro dimension shortcut** — after resizing, Tungro's native ~4:3 shape and the others' 300×300 square all become identical 256×256, so the model can no longer use image size to cheat.

---

## 5. Final Prepared Rice Set 1

**2,066 images, 4 classes, all 256×256 RGB, letterboxed. On-disk size: 37 MB.**

| Class | Count |
|-------|------:|
| rice__bacterial_blight | 514 |
| rice__blast | 477 |
| rice__brown_spot | 606 |
| rice__tungro | 469 |

**Manifest schema (shared across all 3 datasets):**
`src_path, filename, label, source_dataset, split, group_id`

`split` is `unassigned` — splitting is deferred to the merge notebook.

---

## 6. Issues Faced (Summary)

| Issue | Severity | Resolution |
|-------|----------|------------|
| ~65% burst-frame near-duplication | High | Thinned to 1 sharp frame per group (5,932 → 2,066) |
| 2,234 exact duplicate files | Medium | Removed via grouping/thinning |
| Tungro **dimension** shortcut (0% at 300×300 vs 100% for others) | High | Removed by letterbox to 256² |
| Tungro **background/framing** shortcut (soil + whole-plant) | High | *Cannot be fixed by resizing* — deferred to train-time augmentation + robustness test |
| 156 RGBA files (named `.jpg`, PNG-encoded) | Low | Flattened via `.convert("RGB")` |
| Corrupt images | — | 0 found |
| Mislabels | — | 0 found |

---

## 7. What's Still Remaining for Rice Set 1

Cleaning is complete; the following are deliberately deferred:

1. **Tungro background/framing shortcut — deferred to training.** This is the most important open item. Resizing removed the *size* signal, but Tungro is still the only class shot on soil / as whole plants. The model can still latch onto that unless we apply **heavier augmentation to Tungro** (random crops, color jitter, ideally background randomization or segmentation) and run an explicit **robustness check** (e.g. a few Tungro images on green backgrounds) at modeling time. Flagged clearly as a known limitation.

2. **Train / val / test split — deferred to Notebook D.** Done once, centrally, on the merged 20-class pool so it's stratified and group-aware across all sources at once. (`split = unassigned` for now.)

3. **Augmentation & normalization — deferred to training.** Random flips/rotation/jitter/crop-to-224² and ImageNet normalization run live in the training loader; only one clean deterministic copy per image is saved here.

4. **Publishing to Kaggle — next action.** Save `prepared/rice_s1/` as a Kaggle Dataset (e.g. `rice-s1-cleaned-256`) so it can be attached to the merge notebook.

---

## 8. Next Step

Publish `prepared/rice_s1/`, then move to **Notebook C — Rice Set 2** (`vbookshelf/rice-leaf-diseases`): 120 pristine images, contributing the `leaf_smut` class plus a few extra `bacterial_blight` / `brown_spot` samples. Its only real challenge is the extreme 3.44:1 panoramic aspect ratio, handled by the same letterbox.
