# FarmerVision — Licensing & References

*Milestone 6 · Section E*

## Code License

FarmerVision source code is released under the **MIT License** — see [`LICENSE`](../LICENSE) at the
repository root (© 2026 Group 7, DS & AI Lab, IIT Madras).

Third-party libraries retain their own licenses (PyTorch BSD-3, Hugging Face Transformers Apache-2.0,
FastAPI MIT, Qdrant Apache-2.0, LightGBM MIT, timm Apache-2.0, sentence-transformers Apache-2.0,
Expo/React Native MIT, etc.).

## Dataset Licenses & Sources

| Dataset | Source | License / Terms |
|---|---|---|
| KCC Query Dataset (Uttar Pradesh, 2020–2025) | [data.gov.in](https://data.gov.in) | Government Open Data License – India (GODL) |
| Crop production / yield records | Government agri-production data ([data.gov.in](https://data.gov.in)) | Government Open Data License – India (GODL) |
| ICAR & government advisory documents | [icar.org.in](https://icar.org.in) and government publications | Government publications (public use; cite source) |
| Rice + Wheat leaf-disease image corpus | Kaggle / curated (20 classes) | Per the respective Kaggle dataset licenses (research/education use) |
| Mandi (Agmarknet) live prices | [data.gov.in Agmarknet API](https://data.gov.in) | Government Open Data License – India (GODL); API key required |
| Weather (live) | [Open-Meteo](https://open-meteo.com) | CC-BY 4.0 (free, non-commercial & commercial) |
| Weather (IMD, staging) | [indianapi.in](https://indianapi.in) | Provider terms; API key required |

> Full corpora are external and not redistributed in this repo; only samples live under `data/`
> (see `data/README.md`). Users must obtain the source datasets under their respective terms.

## Model Sources & Citations

| Model | Source | License |
|---|---|---|
| `gemma-3-4b-it` (generator base) | Google, via Hugging Face | Gemma Terms of Use (gated; accept on the model page) |
| `BAAI/bge-m3` (embedder) | BAAI, via Hugging Face | MIT |
| `distilbert-base-multilingual-cased` (intent/entity/guardrail backbone) | Hugging Face | Apache-2.0 |
| `vit_small_patch16_224.augreg_in21k_ft_in1k` (vision backbone) | timm / Google AugReg | Apache-2.0 |
| Gemini (cloud model, backend orchestration) | Google | Google API Terms |
| LightGBM (yield) | Microsoft | MIT |

**Citations**
- Gemma Team, Google DeepMind. *Gemma 3 Technical Report.* 2025.
- Chen, J. et al. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings.* 2024.
- Sanh, V. et al. *DistilBERT, a distilled version of BERT.* 2019.
- Steiner, A. et al. *How to train your ViT? (AugReg).* 2021. Dosovitskiy, A. et al. *An Image is Worth 16×16 Words (ViT).* 2020.
- Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Qdrant — vector database. https://qdrant.tech
- Open-Meteo — free weather API. https://open-meteo.com

## Attribution & Acknowledgements

Kisan Call Centre (KCC) and Agmarknet data provided by the Government of India via data.gov.in.
Advisory content derived from ICAR and government agricultural publications. Built for the
DS & AI Lab course, Group 7, IIT Madras.

## Responsible-Use Note

Vision diagnoses and generated advice are **decision support, not a substitute for expert
judgement** — outputs are presented as suggestions and farmers are advised to confirm with their
local KVK / agriculture officer before acting (especially on dosages and restricted inputs).
