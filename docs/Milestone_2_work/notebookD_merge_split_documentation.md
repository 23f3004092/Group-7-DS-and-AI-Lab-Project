# Notebook D — Dataset Integration & Splitting Documentation

**Notebook:** D (merge + split), run on Kaggle
**Inputs:** the three cleaned datasets from Notebooks A/B/C (`wheat-cleaned-256`, `rice-s1-cleaned-256`, `rice-s2-cleaned-256`)
**Goal:** Merge the three cleaned sources into one unified pool, then produce a single leakage-safe, stratified train/val/test split — the final training-ready dataset for the combined wheat + rice disease classifier.

---

## 1. Overview

Notebooks A/B/C each cleaned one dataset and emitted images + a manifest in a **shared schema**. Notebook D is where they come together: it concatenates the three manifests into one 20-class pool, then splits that pool **once, centrally** — which is the only way to guarantee correct stratification of the tiny `leaf_smut` class and to keep near-duplicate groups from leaking across the train/test boundary.

**Headline result:** three cleaned datasets → **one unified dataset of 12,859 images across 20 classes**, split 80/10/10 with **zero leakage**, materialized to 273 MB.

---

## 2. Inputs → Output (at a glance)

| | Source | Images | Classes |
|--|--------|-------:|--------:|
| **Input 1** | wheat (Notebook A) | 10,673 | 15 (`wheat__*`) |
| **Input 2** | rice Set 1 (Notebook B) | 2,066 | 4 (`rice__*`) |
| **Input 3** | rice Set 2 (Notebook C) | 120 | 3 (`rice__*`) |
| **Output** | **merged + split** | **12,859** | **20** |

The class count is 20, not 22, because rice Set 1 and Set 2 **share** `bacterial_blight` and `brown_spot` (they merge into the same classes); Set 2 uniquely adds `leaf_smut`.

---

## 3. Step 1 — Merge (Integration)

**Method.** Each cleaned dataset was published with its `manifest.csv` and an `images/` folder. Notebook D discovers all three manifests, resolves each row to its current on-disk path, and concatenates them into one dataframe.

**Schema alignment.** All three manifests already share the identical column schema (`src_path, filename, label, source_dataset, split, group_id`) — this was enforced back in the cleaning notebooks precisely so the merge would need no reconciliation. Labels were also pre-normalized to canonical, crop-prefixed keys, so no conflicting attributes remained to resolve at merge time.

**Output:**
```
Total rows        : 12859 (expect 12859)
Found on disk     : 12859 | missing: 0
Unique group_ids  : 11530

Per source_dataset:
 wheat_kushagra3204         10673
 rice_s1_nirmalsankalana     2066
 rice_s2_vbookshelf           120

Classes: 20
```

**Two things this output confirms:**
- **0 missing on disk** — the published folder structures resolved correctly; every manifest row maps to a real image.
- **11,530 unique `group_id`s vs 12,859 rows** — the ~1,329 difference is wheat's kept near-duplicate clusters. This is exactly why the split must be group-aware.

**Merged rice classes (the merge working):** `bacterial_blight` = 554 (514 from S1 + 40 from S2), `brown_spot` = 646 (606 + 40), with `leaf_smut` standing alone at 40.

---

## 4. Step 2 — Group-Aware Stratified Split

**The two requirements this split must satisfy simultaneously:**
1. **No leakage** — near-duplicate images (same `group_id`) must never span two splits, or test accuracy is inflated.
2. **Stratification** — every class must appear in train/val/test at roughly the target ratio, including the tiny classes.

**Method.** Instead of splitting individual images, we split whole **groups**. For each class, groups are allocated (largest-first) to whichever split currently has the biggest deficit against its target — this keeps ratios tight even though whole groups move together. A **minimum-eval floor of 8** protects `leaf_smut`: any class is guaranteed ≥ 8 images in both val and test.

**`leaf_smut` handling (decision b+c).** With only 40 images, a plain 80/10/10 gives just 4 val / 4 test — too noisy to evaluate. The floor lifts it to **24 / 8 / 8** (that's why it reads 20%, not 10%). This is paired with train-time oversampling + heavy augmentation (Notebook E) so training isn't starved either.

**Split result (per class):**

```
label                        train  val  test  total  val%  test%
rice__bacterial_blight         444   55    55    554   9.9   9.9
rice__blast                    381   48    48    477  10.1  10.1
rice__brown_spot               516   65    65    646  10.1  10.1
rice__leaf_smut                 24    8     8     40  20.0  20.0
rice__tungro                   375   47    47    469  10.0  10.0
wheat__aphid                   687   86    86    859  10.0  10.0
wheat__black_rust              251   32    32    315  10.2  10.2
wheat__blast                   466   59    59    584  10.1  10.1
wheat__brown_rust              953  119   119   1191  10.0  10.0
wheat__common_root_rot         454   57    57    568  10.0  10.0
wheat__fusarium_head_blight    511   64    64    639  10.0  10.0
wheat__healthy                 828  104   104   1036  10.0  10.0
wheat__leaf_blight             455   56    56    567   9.9   9.9
wheat__mildew                  899  113   113   1125  10.0  10.0
wheat__mite                    612   76    76    764   9.9   9.9
wheat__septoria                279   35    35    349  10.0  10.0
wheat__smut                    404   50    50    504   9.9   9.9
wheat__stem_fly                138   17    17    172   9.9   9.9
wheat__tan_spot                517   66    66    649  10.2  10.2
wheat__yellow_rust            1081  135   135   1351  10.0  10.0

Overall: train 10275 | val 1292 | test 1292   (79.9 / 10.0 / 10.0)
```

**Leakage-prevention proof (the critical check):**
```
LEAKAGE — groups spanning >1 split (must be 0): 0
Classes below floor of 8 in val/test: none ✅
```

Both checks pass: **zero leakage** (the guarantee built since the wheat notebook), and every one of the 20 classes has ≥ 8 images in both evaluation splits. Stratification held tight — every class sits within a fraction of a percent of 10%, except `leaf_smut` where the floor deliberately applies.

---

## 5. Step 3 — Materialize the Final Dataset

**Method.** Images are copied into an **ImageFolder-ready** layout — `train/val/test / <label> / <image>` — alongside three artifacts.

**Output:**
```
Images written: 12859 | failed: 0
Master manifest rows: 12859
On-disk per split: {'train': 10275, 'val': 1292, 'test': 1292}
Classes in train : 20 (expect 20)
Wrote label_to_idx.json with 20 classes
```

Everything reconciles: 12,859 written, 0 failed, on-disk counts match the split plan exactly.

**`label_to_idx.json` — why it exists.** The class→integer mapping is frozen here and saved to a file. This prevents a subtle bug where a future loader could infer different integer IDs for the same class across train vs. test. Notebook E loads this file so the mapping is identical everywhere.

---

## 6. Final Prepared Dataset

**12,859 images · 20 classes · 256×256 RGB letterboxed · split 80/10/10 · 273 MB**

**Directory layout:**
```
final/
├── train/   (10,275 images, 20 class folders)
├── val/     ( 1,292 images, 20 class folders)
├── test/    ( 1,292 images, 20 class folders)
├── master_manifest.csv     (filename, label, source_dataset, split, group_id, rel_path)
└── label_to_idx.json       (frozen 20-class index map)
```

**Class composition:** 15 wheat classes + 5 rice classes (`bacterial_blight`, `blast`, `brown_spot`, `leaf_smut`, `tungro`).

---

## 7. Integration & Splitting — Design Decisions

This maps directly onto the Milestone 2 template's "Dataset Integration" and "Dataset Splitting" sections:

- **Datasets combined:** 3 (1 wheat, 2 rice), via concatenation of shared-schema manifests.
- **Integration methodology:** clean separately → emit identical schema + canonical labels → concatenate → split centrally.
- **Schema alignment:** enforced upstream (all manifests share 6 columns); no reconciliation needed at merge.
- **Handling conflicting attributes:** rice `bacterial_blight` / `brown_spot` intentionally *merged* across S1+S2; crop prefixes (`wheat__` / `rice__`) prevent same-named diseases in different crops (e.g. wheat vs rice "blast") from colliding.
- **Deduplication after merging:** exact + near-dup dedup was done *within* each source; the group-aware split then prevents any residual cross-source near-dups from leaking.
- **Split ratio:** 80/10/10, stratified, with an 8-image val/test floor.
- **Leakage prevention:** whole `group_id` clusters assigned to a single split; verified 0 groups spanning splits.
- **Split justification:** the datasets ship with no usable split (rice) or an untrustworthy leaking one (wheat, 671 leaking groups), so a fresh central split was required.

---

## 8. What's Remaining

Notebook D completes the **data preparation** for Milestone 2. Remaining:

1. **Publish `final/` as a Kaggle Dataset** (e.g. `crop-disease-prepared-256`) — the single read-only input to the training notebook. *(After this, the dataset is frozen and immutable.)*
2. **Notebook E (Milestone 3):** the training pipeline — augmentation, normalization, imbalance handling, and the Tungro robustness check — all applied live at train time, never modifying this dataset. Documented separately in the Notebook E design doc.

---

## 9. Summary

The three cleaned datasets were merged into one 20-class pool of 12,859 images and split 80/10/10 with a group-aware, stratified strategy that achieved **zero leakage** and correct representation of every class — including a protective floor for the 40-image `leaf_smut` class. The result is a 273 MB, ImageFolder-ready dataset with a frozen label map and master manifest: **ready for model training.**
