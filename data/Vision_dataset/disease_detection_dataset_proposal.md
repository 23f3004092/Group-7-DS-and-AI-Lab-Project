
---

### Strategy A: Dedicated Rice + Wheat (Clean & Focused)
- **Rice:** `vbookshelf/rice-leaf-diseases` — 4 classes (Brown Spot, Blast, Bacterial Blight, Healthy), ~38MB, 28K downloads, widely cited
- **Wheat:** `kushagra3204/wheat-plant-diseases` — 14K+ images, multiple rust types + Septoria
- **PlantVillage role:** Supplementary — used for transfer learning pretraining or dropped
- **PlantDoc role:** General field-noise benchmark for domain gap study
- ✅ Directly aligns with Milestone 1 scope ("exclusively Rice & Wheat")
- ⚠️ Rice dataset is small (~38MB); may need augmentation

### Strategy B: Bigger Rice Dataset + Wheat
- **Rice:** `anshulm257/rice-disease-dataset` — 6 classes (adds Sheath Blight, Leaf Scald), ~1GB, richer
- **Wheat:** `kushagra3204/wheat-plant-diseases` — same as above
- **PlantVillage:** Dropped entirely from vision pipeline
- ✅ More robust Rice coverage with 6 disease classes
- ⚠️ Larger download, some classes may be outside UP's common diseases

### Strategy C: Multi-Crop Broad + Rice/Wheat Focus
- **Primary:** PlantVillage (all 38 classes, 54K images) for general pretraining
- **Supplementary:** Add Rice + Wheat datasets for fine-tuning/evaluation
- **PlantDoc:** Field evaluation benchmark
- ✅ Largest training set, strongest transfer learning baseline
- ⚠️ Dilutes focus from Rice/Wheat; harder to justify in report

### Strategy D: Combined Rice Sources + Wheat
- **Rice:** Merge `vbookshelf/rice-leaf-diseases` + `nirmalsankalana/rice-leaf-disease-image` (~205MB, more diverse images)
- **Wheat:** `kushagra3204/wheat-plant-diseases`
- **PlantVillage:** Optional supplementary for pretraining
- ✅ Most diverse Rice coverage through dataset integration
- ⚠️ Integration adds complexity (dedup, label alignment) — good for Milestone 2 Section 7 though

---

Which strategy do you want to go with?