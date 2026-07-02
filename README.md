# AgriAssist: Farmer Query Assistant

**Course:** DS and AI Lab | **Group:** 7 | **Institution:** IIT Madras  
**GitHub:** [Group-7-DS-and-AI-Lab-Project](https://github.com/<org>/Group-7-DS-and-AI-Lab-Project) | **Status:** In Progress

---

## Project Overview

AgriAssist is an agentic AI system designed to help Indian farmers get instant, accurate answers to their agricultural queries — available 24/7. Farmers can type a question or upload a crop leaf image, and the system classifies their intent, retrieves information from the most relevant source, and generates a plain-language response.

This addresses a critical gap: the Kisan Call Centre (KCC) — India's primary farmer support system — is human-staffed, limited in hours, and cannot scale to meet demand.

---

## Problem Statement

Indian farmers frequently face urgent queries related to crop diseases, pest outbreaks, fertilizer usage, mandi prices, and government schemes. Existing automated systems (e.g. KisanQRS) use basic clustering over past KCC answers and explicitly filter out price queries — leaving a large portion of farmer needs unmet. No existing system combines multi-source retrieval, visual disease detection, and intent-based routing into a single unified assistant.

AgriAssist solves this by:
- Detecting crop disease from leaf images using a fine-tuned Vit
- Classifying query intent using LLM
- Generating a grounded, plain-language answer via LLM

---

## Features

- 🌿 **Crop Disease Detection** — Upload a leaf image, get disease name and confidence score
- 🔍 **Intent Classification** — Multi-label classification across 5 query categories
- 📚 **RAG-based Advisory** — Answers grounded in real KCC expert Q&A data
- 💰 **Live Mandi Prices** — Real-time price retrieval via Agmarknet API
- 🏛️ **Scheme Information** — Government scheme guidance via ICAR document RAG

---

## Architecture

![Architecture Diagram](docs/Architecture_Diagram.png)

> _Architecture diagram will be updated at Milestone 3._



---

## Dataset Details

| Dataset | Purpose | Source | Size |
|---|---|---|---|
| KCC Query Dataset | RAG knowledge base + classifier training | [data.gov.in](https://data.gov.in) | Lakhs of Q&A pairs |
| PlantVillage | ViT fine-tuning | [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) | ~54,000 images, 38 classes |
| ICAR Advisory Docs | Scheme/advisory RAG | [icar.org.in](https://icar.org.in) | PDF documents |
| Agmarknet API | Live mandi price retrieval | [agmarknet.gov.in](https://agmarknet.gov.in) | Real-time API |



---

## Installation



---

## Running


> _Running instructions will be finalized at Milestone 6._

---

## Results

> _To be updated from Milestone 4 onwards._

---

## Demo

> _Screenshots and demo link to be added at Milestone 6._

🔗 **Live Demo:** [HuggingFace Spaces](#) _(available after Milestone 6)_

---

## Project Structure

```
Group-7-DS-and-AI-Lab-Project/
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
├── docs/                  ← reports, architecture diagrams, presentations
├── data/                  ← sample data only; see data/README.md
├── notebooks/             ← EDA, preprocessing, experiments
├── src/                   ← reusable source code
│   ├── data/
│   ├── models/
│   ├── inference/
│   ├── retrieval/
│   ├── agents/
│   ├── evaluation/
│   └── utils/
├── app/                   ← Gradio deployment app
├── configs/               ← model and training YAML configs
├── models/                ← model cards; checkpoints on HuggingFace Hub
├── outputs/               ← predictions, figures, logs
└── tests/                 ← unit tests
```

---

## Milestones

| Milestone | Focus | Deadline | Status |
|---|---|---|---|
| M1 | Problem Definition & Literature Review | July 2 | ✅ Done |
| M2 | Dataset Preparation | July 9 | 🔄 In Progress |
| M3 | Model Architecture & End-to-End Setup | July 23 | ⏳ Upcoming |
| M4 | Model Training | July 30 | ⏳ Upcoming |
| M5 | Evaluation & Analysis | Aug 6 | ⏳ Upcoming |
| M6 | Deployment & Documentation | Aug 13 | ⏳ Upcoming |
| Final | Project Presentation | Aug 20+ | ⏳ Upcoming |

---

## Team

| Name | Role |
|---|---|
| [Harliv Singh] | CV model fine-tuning |
| [Aneeqa] | Dataset preparation + ChromaDB RAG setup |
| [Lokesh] | Query classification |
| [Mahesh Ishran] | Agent routing + Agmarknet API integration |
| [Tanmay Sahu] | Evaluation + Gradio app + HuggingFace deployment |

---

## References

1. **Rana, A., et al. (2023).** *KisanQRS: A Deep Learning-Based Automated Query-Response System for Kisan Call Centre.*
   
2. **ReadyTensor (2025).** *Agentic AI for Smart Agriculture: Multimodal RAG System.*
   

3. **Hughes, D., & Salathé, M. (2016).** *An Open Access Repository of Images on Plant Health to Enable the Development of Mobile Disease Diagnostics.*


4. **Singh, D., et al. (2020).** *PlantDoc: A Dataset for Visual Plant Disease Detection.*
  

5. **Mwebaze, E., et al.** *Cassava Leaf Disease Classification Dataset.*


6. **PlantWild (2024).**


7. **Balpande, A., et al. (2024).** *AI-Powered Agriculture Optimization Chatbot Using RAG and Generative AI.*
   

8. **Digital Green.** *FarmerChat.*


9. **International Food Policy Research Institute (IFPRI).** *GAIA Phase II (2025–2027): Generative AI for Agricultural Information Access.*
   

10. **World Economic Forum (2025).** *Shaping the Deep-Tech Revolution in Agriculture.*
    

11. **World Economic Forum (2025).** *How AI is Enabling Agricultural Intelligence and Revolutionizing Farming.*
    
12. **Government of India.** *Kisan e-Mitra.*
    

13. **Government of India.** *Union Budget 2026–27: Bharat-VISTAAR.*

14. **Wadhwani Institute for AI.** *AgriAI Collect.*


15. **Syngenta.** *Cropwise Digital Farm Management Platform.*

16. **Khurana, S., et al.** *MuRIL: Multilingual Representations for Indian Languages.*

17. **Kakwani, D., et al.** *IndicBERT: A Multilingual Model for Indian Languages.*



---

*Last updated: July 2025 | Group 7 | DS and AI Lab | IIT Madras*