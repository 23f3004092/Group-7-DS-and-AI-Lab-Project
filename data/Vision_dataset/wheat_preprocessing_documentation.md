# Wheat Dataset — Preprocessing Documentation

**Dataset:** `kushagra3204/wheat-plant-diseases` (Kaggle)
**Notebook:** A (wheat), run on Kaggle
**Goal:** Turn the raw wheat dataset into a clean, deduplicated, uniformly-sized image set ready to be merged with the rice datasets and used to train a single crop-disease classifier.

---

## 1. Overview

The wheat dataset is the largest of our three sources (~14k images, ~7 GB). The EDA had already flagged that it looked large and usable but hid three serious defects: **corrupted class labels**, **duplicate images**, and **data leakage** between the author's train/test split. This notebook fixes all three and produces a clean, letterboxed 256×256 image set with a manifest.

**Headline result:** 14,154 raw images → **10,673 clean images**, and 45 corrupted folder labels → **15 correct disease classes**.

---

## 2. Starting Point (Raw Dataset)

```
Total wheat images: 14154

Split counts:
 split
train    13104
test       750
val        300

Raw label count: 45 (expected — the corrupted naming)
```

The dataset ships with a train/val/test split already made by the author, and appears to have 45 classes — but that number is wrong, and fixing it is the first step.

---

## 3. Step-by-Step Preprocessing

### Step 1 — Label repair (45 → 15 classes)

**The problem.** The dataset only has 15 real wheat diseases, but the author named the folders inconsistently across splits:

| Split | Naming style | Example |
|-------|--------------|---------|
| train | Title Case, spaces | `Black Rust` |
| test  | lowercase + `_test` | `black_rust_test` |
| val   | lowercase + `_valid` | `black_rust_valid` |

This made each disease look like 3 separate classes (15 × 3 = 45). There was also one genuinely broken folder — `blast_test_valid` — where the suffix got doubled.

**The fix.** We stripped the `_test` / `_valid` / `_test_valid` suffixes, lowercased, and collapsed everything back to 15 canonical names. We also added a crop prefix (`wheat__`) so wheat's "Blast" never gets confused with rice's "Blast" later in the merged model.

**Output:**
```
Canonical classes: 15   (expected 15)
['aphid', 'black_rust', 'blast', 'brown_rust', 'common_root_rot',
 'fusarium_head_blight', 'healthy', 'leaf_blight', 'mildew', 'mite',
 'septoria', 'smut', 'stem_fly', 'tan_spot', 'yellow_rust']

Classes missing from a split: none

Did the broken 'blast_test_valid' fold into 'blast'?
blast    20   (correct — those 20 images were its val partition)
```

Every class was present in all three splits, and the broken folder repaired correctly.

---

### Step 2 — Duplicate detection (exact + near-duplicate)

**Why this matters.** Duplicates cause two problems. **Redundancy** wastes training compute. Far worse, **leakage** — a photo (or near-copy) appearing in both train and test — lets the model "see the answer key" during testing, so the reported accuracy looks great but is dishonest.

**How we detect them.**
- **Exact duplicates:** byte-for-byte identical files, caught with an MD5 checksum.
- **Near-duplicates:** resized / re-saved / lightly-edited copies. These have different bytes, so MD5 misses them. We use a **perceptual hash (pHash)** — a 64-bit fingerprint of each image's visual structure — and treat two images as near-duplicates if their fingerprints differ by ≤ 6 bits (Hamming distance).

**Output:**
```
Exact byte-duplicate files: 4856 across 11143 unique blobs
Unreadable images: 0

Near-dup groups (size>1): 2299 | largest cluster: 16
Images inside a multi-image group: 6795
Groups spanning >1 disease label (probable mislabels): 329
Groups leaking across the SHIPPED split: 671 | images involved: 2321

Dropped 3011 byte-exact duplicate files → 11143 kept.
```

**What we did with each finding:**
- **Exact duplicates (3,011):** deleted — kept one copy of each.
- **Near-duplicates:** *kept but grouped.* Mild variations of the same leaf actually help the model generalize, so we don't delete them. Instead each near-duplicate cluster gets a `group_id`, so the final split (in Notebook D) can keep an entire group on one side — making leakage structurally impossible.
- **671 leaking groups (2,321 images):** this is proof the author's original train/test split is untrustworthy, and is our written justification for throwing it away and re-splitting from scratch later.

---

### Step 3 — Mislabel investigation and removal

**The problem.** 329 near-duplicate groups contained images filed under *two different disease labels*. That's either a genuine mislabel (same leaf, two conflicting labels) or a harmless false-merge (two different leaves grouped because they share a plain background). These need opposite treatment, so we measured how visually identical each conflicting group actually was.

**Output:**
```
Max intra-group Hamming distance distribution:
 0    282
 2     31
 4      6
 6      8
 8      2

Most-conflicting disease pairings:
 black_rust, brown_rust       109
 leaf_blight, septoria         97
 leaf_blight, tan_spot         46
 ...
```

**Interpretation.** 313 of 329 groups (95%) sit at distance 0–2 — near-identical images with contradictory labels, i.e. **genuine mislabels**. The pairings show it's systematic: annotators genuinely confused visually similar diseases (`black_rust`↔`brown_rust`, `leaf_blight`↔`septoria`/`tan_spot`).

**The fix.** For a mislabeled group we can't trust *either* label, so we dropped the whole group (only the tight, dist ≤ 2 groups). The 10 loose groups (dist 6–8) were left alone.

**Output:**
```
Mislabel groups dropped : 313
Images removed          : 469
Wheat rows remaining    : 10674
Classes still present   : 15
```

---

### Step 4 — Quality filter + resize + materialize

**Three actions in one pass:**

1. **Near-blank filter** — dropped images with almost no detail (Laplacian variance < 8), which carry no disease signal. We deliberately kept moderately soft "phone-quality" images, because the model must handle those in the real world.
2. **Letterbox resize to 256×256** — resized aspect-preserving and padded the edges, so wheat's collages and odd-shaped images don't get squashed. This keeps lesion shapes true.
3. **Materialize** — wrote the cleaned JPEGs to `/kaggle/working/prepared/wheat/images/` plus a `manifest.csv`.

**Output:**
```
Near-blank dropped : 1
Failed to process  : 0
Images written     : 10673  ->  /kaggle/working/prepared/wheat/images
Manifest columns   : ['src_path','filename','label','source_dataset','split','group_id']
```

Only 1 near-blank image was found — confirming the dataset is genuinely photo-heavy.

---

## 4. Final Prepared Wheat Dataset

**10,673 images, 15 classes, all 256×256 RGB, letterboxed.**

| Class | Count | | Class | Count |
|-------|------:|-|-------|------:|
| wheat__aphid | 859 | | wheat__mildew | 1,125 |
| wheat__black_rust | 315 | | wheat__mite | 764 |
| wheat__blast | 584 | | wheat__septoria | 349 |
| wheat__brown_rust | 1,191 | | wheat__smut | 504 |
| wheat__common_root_rot | 568 | | wheat__stem_fly | **172** |
| wheat__fusarium_head_blight | 639 | | wheat__tan_spot | 649 |
| wheat__healthy | 1,036 | | wheat__yellow_rust | **1,351** |
| wheat__leaf_blight | 567 | | | |

**Manifest schema (shared across all 3 datasets):**
`src_path, filename, label, source_dataset, split, group_id`

Note: `split` is currently `unassigned` on purpose — see below.

---

## 5. Issues Faced (Summary)

| Issue | Severity | Resolution |
|-------|----------|------------|
| 45 corrupted labels (inconsistent naming across splits) | High | Repaired to 15 canonical classes |
| One broken folder (`blast_test_valid`) | Medium | Folded correctly into `blast` |
| 3,011 exact duplicate images | Medium | Removed (kept one copy) |
| 671 groups leaking across the author's split | High | Author split discarded; re-split deferred to Notebook D, group-aware |
| 313 mislabel groups (annotator confusion) | Medium | Dropped (469 images) |
| Near-blank / junk images | Low | 1 dropped |
| Non-square, mixed aspect ratios | Low | Letterboxed to 256² |

---

## new size of dataset after above preprocessing steps ~233 MB reduced from 7GB

## 6. What's Still Remaining for Wheat

Wheat's **cleaning** is complete, but a few things are deliberately **deferred**, because they must happen on the full merged dataset (wheat + both rice sets), not on wheat alone:

1. **Train / val / test split — deferred to Notebook D.** The split is done once, centrally, on the merged pool, so it can be **group-aware** (keeping each `group_id` on one side to prevent leakage) and stratified across all 20 classes at once. That's why `split` currently reads `unassigned`.

2. **Class imbalance handling — deferred to training.** After cleaning, imbalance widened slightly: floor `stem_fly` = 172, ceiling `yellow_rust` = 1,351, ratio ≈ 7.9. This is handled at *train time* with class-weighted loss / weighted sampling and evaluated with macro-F1 + per-class recall — **not** by deleting data.

3. **Data augmentation — deferred to training.** Random flips, rotation, color jitter, and the final crop to 224² run *live in the training loader* every epoch, so they must not be baked into the saved images. This prepared dataset holds one clean, deterministic copy per image.

4. **Normalization — deferred to training.** ImageNet mean/std normalization is applied in the loader, matched to the chosen backbone (default: EfficientNet-B0).

5. **Publishing to Kaggle — next action.** The `prepared/wheat/` output needs to be saved as a Kaggle Dataset (e.g. `wheat-cleaned-256`) so it can be attached to the merge notebook.

---

## 7. Next Step

Publish `prepared/wheat/` as a Kaggle Dataset, then move to **Notebook B — Rice Set 1** (`nirmalsankalana/rice-leaf-disease-image`), which has a different set of problems: heavy near-duplication and a "Tungro" background shortcut flagged in EDA.
