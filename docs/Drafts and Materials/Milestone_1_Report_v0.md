# Milestone 1: Problem Definition, Literature Review & Gap Analysis

**Project Title:** Field-Robust, Confidence-Aware, Faithfulness-Grounded Crop Advisory System

**Team Members:** Aneeqa, Harliv, Lokesh, Mahesh, and Tanmay  

**Date:** July 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Scope and Boundaries](#2-scope-and-boundaries)
3. [Stakeholders](#3-stakeholders)
4. [Project Objectives](#4-project-objectives)
5. [Motivation and Context](#5-motivation-and-context)
6. [Proposed Solution Overview](#6-proposed-solution-overview)
7. [System Architecture](#7-system-architecture)
8. [Solution in Action — Interaction Scenarios](#8-solution-in-action--interaction-scenarios)
9. [Literature Review & Existing Solutions](#9-literature-review--existing-solutions)
10. [Comparative Analysis](#10-comparative-analysis)
11. [Datasets and Evaluation Framework](#11-datasets-and-evaluation-framework)
12. [Milestone 1 Scope — Inclusions and Exclusions](#12-milestone-1-scope--inclusions-and-exclusions)
13. [References](#13-references)

---

## 1. Problem Statement

India's 120+ million smallholder farming households depend on timely, accurate agronomic intelligence — identifying crop diseases, selecting appropriate treatments, understanding government schemes, and tracking market prices — to protect their yields and livelihoods. The primary institutional support channel, the Kisan Call Centre (KCC), handles millions of calls annually but is fundamentally bottlenecked by limited operating hours, high call volumes, language barriers, and the inability to process visual inputs such as photographs of diseased crops.

Meanwhile, existing AI-powered agricultural advisory tools suffer from two critical and largely unaddressed trust failures:

1. **Disease detection accuracy collapses in real field conditions.** Models trained on clean, lab-generated datasets like PlantVillage achieve 98–99% accuracy on controlled images but suffer catastrophic performance drops — with macro-F1 falling to 50–88% and sometimes as low as 41% — when tested on real-world field photographs with complex backgrounds, variable lighting, and overlapping foliage. A farmer photographing a leaf with their phone, in sunlight, against soil, is exactly the failure case these models cannot handle. Most student and research projects report 99% on PlantVillage and stop, without ever confronting this domain gap.

2. **LLM-based advisory systems hallucinate treatment advice.** General-purpose large language models generate plausible-sounding but factually incorrect agronomic advice — a dangerous failure when the output is a specific pesticide dosage. A 2025 KCC study found that while retrieval-only answering achieved near-perfect accuracy with zero hallucination in a closed domain, the generative RAG layer introduced hallucination. The tension between fluency and faithfulness is measurable but rarely measured in existing systems.

3. **Systems do not know when they are wrong.** Current solutions lack confidence calibration and graceful abstention mechanisms. When a disease detector is only 38% confident in its classification, telling a farmer "spray Mancozeb at 2.5 g/litre" is actively harmful. No widely deployed system admits uncertainty or declines to answer when its confidence is low.

This project addresses all three failures by building a **field-robust, confidence-aware, faithfulness-grounded crop advisory system** — one that detects crop diseases reliably under real field conditions, abstains gracefully when uncertain, and returns treatment advice provably grounded in authoritative sources with full citation traceability.

---

## 2. Scope and Boundaries

### In Scope (Milestone 1 and Core Build)

- Multimodal input ingestion: text queries, crop disease images, and voice input (via IndicWhisper ASR)
- LLM-based intent routing and query segmentation across agricultural query types (disease/pest, treatment, government schemes, market prices, general advice)
- CNN/ViT-based crop disease detection trained for field robustness with calibrated confidence scoring and abstention
- RAG-based advisory grounded in authoritative sources (ICAR advisories, KCC historical Q&A pairs, government scheme documents) with source citation
- Domain-adapted embedding model fine-tuned on KCC agricultural corpus for improved retrieval quality in Hindi/regional language queries
- Live API integration for real-time mandi prices (Agmarknet) and weather data
- Multi-turn dialogue with context elicitation (location, crop stage, farming method) for personalized responses
- Tiered response strategy: grounded RAG responses (Tier 1), transparent LLM knowledge with disclaimers (Tier 2), and graceful out-of-scope redirection (Tier 3)
- Voice-first output via Text-to-Speech for low-literacy users

### Out of Scope

- Physical farm actuation (automated irrigation, drone spraying)
- Real-time video feeds or live drone monitoring
- Crop planning and yield prediction (requires soil/weather/market data integration beyond current scope)
- Legal dispute handling between farmers and intermediaries
- Local/on-device LLM deployment (cloud-based inference for Milestone 1)
- Autonomous robotics or smart equipment integration
- Satellite-based crop monitoring

---

## 3. Stakeholders

| Stakeholder | Role | Interest |
|---|---|---|
| **Smallholder farmers** (primary users) | End users who query the system via text, voice, or image | Timely, accurate, localized agronomic advice in their language; disease identification; scheme eligibility; market prices |
| **Agricultural extension officers** | Field agents from KVKs and state agriculture departments | A tool that supplements their limited reach; reduces repetitive queries so they can focus on complex cases |
| **Kisan Call Centre (KCC) operators** | Staff handling farmer helpline queries | Automated first-line response to reduce call volume and wait times |
| **ICAR and State Agricultural Universities** | Content providers and domain experts | Their published advisories form the RAG knowledge base; system accuracy validates their outreach |
| **Government agencies** (Ministry of Agriculture, DAC&FW) | Policy makers and scheme administrators | Scalable delivery of scheme awareness (PMFBY, PM-KISAN) to last-mile beneficiaries |
| **Agritech platforms and NGOs** (Digital Green, Wadhwani AI) | Ecosystem partners and potential adopters | Integration or white-labeling of advisory capabilities |
| **Academic evaluators** | Capstone reviewers and examiners | Measurable contributions with quantitative evaluation across trainable ML components |

---

## 4. Project Objectives

Each objective is measurable, traceable to a milestone deliverable, and aligned with the three core trust failures identified in the problem statement.

| # | Objective | Success Metric | Milestone |
|---|---|---|---|
| O1 | Build a crop disease detector that maintains accuracy under real field conditions, not just lab datasets | Measure accuracy on PlantVillage (lab) vs. PlantDoc/Cassava (field); quantify the domain gap and demonstrate recovery through field-robust augmentation | M3–M5 |
| O2 | Implement confidence-aware abstention so the system declines to answer when unsure | Report abstention rate and abstention correctness (percentage of abstained cases that were genuinely ambiguous or misclassified) | M5 |
| O3 | Ensure all treatment advice is grounded in authoritative sources with verifiable citations | Measure faithfulness using RAGAS metrics (Context Precision, Context Recall, Faithfulness, Response Relevancy); no generated claim without a retrievable source | M5 |
| O4 | Fine-tune a domain-adapted embedding model that measurably improves retrieval quality for agricultural Hindi/regional queries | Compare Recall@5, Recall@10, MRR, and nDCG between generic embeddings and domain-adapted embeddings on a held-out farmer query test set | M4–M5 |
| O5 | Deliver a multimodal system accepting text, image inputs with intent-based routing to specialized backends | End-to-end functional demo processing all three input modalities with correct routing | M6 |
| O6 | Integrate live data sources (mandi prices, weather) and multi-turn context elicitation for personalized responses | Successful API calls returning real-time data; personalized responses varying by location, crop stage, and farming method | M6 |

---

## 5. Motivation and Context

### Why This Problem Matters Now

The global population is projected to reach 9.7 billion by 2050, requiring at least a 70% increase in food production — not through more farmland, but through smarter farming. The global AI in agriculture market reached $3.37 billion in 2026 (up from $2.71 billion in 2025) and is projected to hit $8.23 billion by 2030. The transition from experimentation to execution is underway, but critical gaps remain — particularly for smallholder farmers in the global south who are most vulnerable to information asymmetries.

### The Current State of Farmer Advisory in India

India's Kisan Call Centre handles millions of farmer queries annually across 11 languages, but remains constrained by human staffing limitations, fixed operating hours, and an inability to process visual inputs. The KCC dataset on data.gov.in — lakhs of real, labeled Q&A pairs tagged by crop, state, and category — represents a rich but underutilized resource for building AI advisory systems.

Government initiatives are moving in this direction. The Kisan e-Mitra voice-based chatbot now responds to over 20,000 queries daily in 11 regional languages. The Union Budget 2026-27 proposed Bharat-VISTAAR, a multilingual AI tool integrating AgriStack portals and ICAR agricultural practices with AI systems for customized advisory support. The National Pest Surveillance System uses AI to identify pests across 61 crops and 400+ species. Domain-specific models like BharatGen's AgriParam are being built as India-centric foundational models trained on local agrarian data.

Internationally, Digital Green's FarmerChat serves over 830,000 users across Kenya, Nigeria, Ethiopia, India, and Brazil, supporting 15 languages, with farmers submitting over 10 million queries. IFPRI's GAIA Phase II (2025-2027) is expanding content aggregation, integrating real-time data sources and multimodal models, and establishing evaluation protocols for LLM performance in agricultural extension. The Wadhwani Institute for AI developed AgriAI Collect, which uses automatic speech recognition for multilingual voice inputs and LLMs to extract structured responses with a human-in-the-loop validation system, onboarding 32,000 users.

Yet the World Economic Forum's 2025 report on deep-tech in agriculture identifies persistent barriers: data quality issues in smallholder contexts that decrease reliability of GenAI outputs; hallucinations and errors where GenAI produces information that sounds convincing but is factually incorrect; on-field variability where models trained under ideal conditions often fail in real agricultural settings; and limited model transferability across domains and regions that impairs farmer trust and adoption.

These are precisely the gaps this project targets.

---

## 6. Proposed Solution Overview

The system is a **multi-source agentic retrieval-augmented generation (RAG) framework** that acts as an autonomous multimodal routing architecture. It ingests unstructured inputs — vernacular voice, code-mixed text, and crop imagery — and routes them to specialized backends to deliver localized, high-fidelity agronomic intelligence.

### Core Components

The system has four core components working together, orchestrated by an LLM-based router:

**Component 1 — Disease Detector (Trainable Model 1: CNN/ViT).** A convolutional neural network or vision transformer trained for field-robust crop disease classification. Unlike standard PlantVillage-trained models, this detector is trained with aggressive field-realistic augmentation and evaluated against real-field datasets (PlantDoc, Cassava). It produces calibrated confidence scores and abstains when confidence falls below a defined threshold, preventing dangerous misclassification.

**Component 2 — Domain-Adapted Embedding Model (Trainable Model 2).** A sentence-transformer model fine-tuned on KCC agricultural Q&A pairs using contrastive learning. General-purpose embedding models fail on agricultural Hindi/regional terminology — terms like "makka" (maize), "tana chhedak" (stem borer), "phaphund" (fungus), and "geela sadan" (wet rot) are underrepresented in their training data. This fine-tuned model improves the semantic matching between farmer queries and authoritative advisory documents, directly improving RAG retrieval quality.

**Component 3 — RAG Advisory Engine.** A vector database (ChromaDB/FAISS) indexed with ICAR advisories, KCC historical Q&A pairs, state agricultural university guidelines, government scheme documents, and pest control medication details. Queries are embedded using the domain-adapted model (Component 2), and the top-K most semantically similar document chunks are retrieved. The LLM then generates responses grounded strictly in these retrieved documents, with source citations. A relevance score threshold determines whether the response is grounded (Tier 1), falls back to LLM knowledge with transparent disclaimers (Tier 2), or redirects as out-of-scope (Tier 3).

**Component 4 — LLM Router and Response Coordinator.** The LLM natively understands query intent and segments incoming queries across four paths: disease detection (image inputs), tool calls (mandi prices, weather APIs), RAG retrieval (advisory, scheme, pest management queries), and general LLM knowledge (for queries not covered by RAG). The Response Coordinator synthesizes outputs from all activated paths into a single grounded, cited final response.

### The Tiered Response Strategy

The system employs a transparency-first approach to managing the boundary between grounded knowledge and model inference:

**Tier 1 — Grounded Response (RAG Hit):** The query matches content in the knowledge base. The response is generated strictly from retrieved documents with source citations. The system presents this as verified information.

**Tier 2 — Model Knowledge with Transparency (RAG Miss):** The query is valid and agricultural, but the corpus does not cover it well. The system falls back to the LLM's general knowledge but explicitly tells the farmer that the recommendation is based on general knowledge and has not been verified against a specific advisory document, suggesting confirmation with their local KVK or relevant authority.

**Tier 3 — Out-of-Scope Redirection:** The query is completely non-agricultural or beyond system capabilities. The system redirects gracefully with pointers to appropriate resources.

---

## 7. System Architecture

### Architecture Diagram

![alt text](updated_architecture.png)

*Figure 1: Updated system architecture showing the multimodal routing pipeline. User queries (text, image, voice(stretch Goal)) combined with user context (location, crop, preferences) flow into the LLM Router for query segmentation. The router dispatches to four specialized paths: Disease Detection (Trainable Model 1 — CNN/ViT), Tool Calls (Mandi/Weather APIs), RAG Retrieval (with Domain Embeddings powered by Trainable Model 2, fine-tuned on KCC data, and a Vector DB containing ICAR, KCC, and scheme documents), and LLM Knowledge (for queries not covered by RAG, triggered when the relevance score is low). All outputs converge at the Response Coordinator, which synthesizes and cites sources to produce the final grounded response.*

### Data Flow

The end-to-end data flow operates as follows:

**Input Layer:** The farmer submits a query through text, image upload, or voice (transcribed via IndicWhisper ASR). User context — location, crop type, farming preferences — is collected through context elicitation prompts and optionally stored as a persistent farmer profile.

**Routing Layer:** The LLM Router analyzes the combined query and context, performing query segmentation to determine which backend paths to activate. A single query may activate multiple paths simultaneously (for example, an image of a diseased leaf triggers both the disease detector and the RAG engine for treatment retrieval).

**Processing Layer:** Each activated path processes the query independently. The disease detector returns a structured output: `{disease: "Early Blight", confidence: 0.82, crop: "Tomato"}`. This structured output flows into the RAG engine as an enriched query. The embedding model converts the query into a domain-optimized vector, searches the vector database, and returns top-K relevant document chunks. A relevance score check determines whether retrieved documents are sufficiently relevant (high score → Tier 1 grounded response) or not (low score → fall back to LLM knowledge with Tier 2 disclaimer).

**Synthesis Layer:** The Response Coordinator merges outputs from all activated paths — disease detection results, retrieved advisory documents, API data (prices, weather), and user context — into a single coherent response with inline source citations, confidence indicators, and appropriate disclaimers.

---

## 8. Solution in Action — Interaction Scenarios

The following scenarios illustrate the system's behavior across different query types, demonstrating each component and edge case.

### Scenario 1: Image-Based Disease Detection

**User:** A tomato farmer in Rajasthan notices yellow-brown spots on several leaves. He opens the app and takes a photo of one affected leaf — outdoors, in bright afternoon sun, with soil visible in the background.

**System behavior:** The router sees image input and routes to the disease detector. The model processes the photo — because it has been trained with field-realistic augmentation and tested against real-field datasets, it handles the complex background. It returns: "Early Blight (Alternaria solani) — confidence: 82%." Since 82% exceeds the abstention threshold (60%), the system proceeds. The detected disease becomes a structured query into the RAG engine, which retrieves treatment advisories for "Early Blight + Tomato" from ICAR documents.

**User sees:**

> 🍅 **Detected: Early Blight** (Alternaria solani) — Confidence: High (82%)
>
> **Recommended action** (Source: ICAR advisory, Rajasthan State Agricultural University):
> Remove and destroy affected leaves. Apply Mancozeb 75% WP at 2.5 g/litre of water. Repeat spray after 10–15 days if symptoms persist. Ensure proper spacing between plants for air circulation.
>
> ⚠️ *This advice is retrieved from ICAR published guidelines. Consult your local KVK for dosage confirmation specific to your soil conditions.*

Every fact in the response traces back to a retrieved source document. The LLM did not invent the dosage "2.5 g/litre" — it pulled it from an ICAR advisory.

---

### Scenario 2: Confidence-Aware Abstention

**User:** A farmer takes a photo of a wheat leaf in low light, partially folded, with heavy motion blur.

**System behavior:** The disease detector returns: "Rust — confidence: 38%." Since 38% is below the abstention threshold, the system admits uncertainty instead of guessing.

**User sees:**

> 📷 I couldn't identify the disease with enough certainty from this photo.
>
> **Tips for a clearer photo:** Hold the phone 15–20 cm from the leaf. Use natural daylight, avoid shadows. Place the leaf against a plain surface if possible.
>
> If the problem is urgent, contact your nearest Krishi Vigyan Kendra (KVK) or call the Kisan Call Centre at 1800-180-1551.

This is measurable — the percentage of inputs triggering abstention and whether those abstentions were correct can be evaluated against held-out labeled data.

---

### Scenario 3: Text-Based Pest Query with Ambiguity

**User:** "Meri kapas mein koi keeda lag gaya hai, kya karu?" (My cotton has some pest, what should I do?)

**System behavior:** The router classifies this as a pest management query. The query is vague — "koi keeda" (some pest) names no specific pest. The RAG engine retrieves the most common cotton pests from KCC historical data and differentiating symptoms.

**User sees:**

> You mentioned a pest issue in cotton. Here are the most common possibilities:
>
> **1. Bollworm** — Look for: holes in bolls, frass (insect droppings) near flowers. Treatment: Spray Emamectin Benzoate 5% SG at 4 g per 10 litres. *(Source: KCC advisory, Maharashtra)*
>
> **2. Whitefly** — Look for: tiny white insects on underside of leaves, sticky honeydew. Treatment: Spray Diafenthiuron 50% WP at 12 g per 10 litres. *(Source: ICAR CICR advisory)*
>
> **3. Jassid/Aphid** — Look for: leaf curling, yellowing from edges. Treatment: Spray Imidacloprid 17.8% SL at 3 ml per 10 litres. *(Source: KCC advisory, Gujarat)*
>
> Can you describe or photograph the pest for a more specific recommendation?

The system gives multiple grounded answers ranked by likelihood, each traced to a source, and asks for more information rather than hallucinating a single confident answer.

---

### Scenario 4: Live Market Price Query (Tool Call Path)

**User:** "What's the price of wheat in the mandi today?"

**System behavior:** The router classifies this as a market/price query and triggers the Agmarknet API tool call.

**User sees:**

> **Wheat prices today (nearest mandis to your region):**
> - Jaipur mandi: ₹2,450/quintal
> - Kota mandi: ₹2,420/quintal
> - MSP (2025-26): ₹2,425/quintal
>
> Prices suggest selling at Jaipur mandi would give you ₹25/quintal above MSP.

---

### Scenario 5: Government Scheme Query (RAG Path)

**User:** "I want to know about PM Fasal Bima Yojana"

**System behavior:** The router classifies this as a scheme/policy query. The RAG engine searches government scheme documents indexed in the vector store.

**User sees:**

> **Pradhan Mantri Fasal Bima Yojana (PMFBY)** *(Source: Department of Agriculture, GoI)*
>
> This is a crop insurance scheme covering losses from natural calamities, pests, and diseases. Premium: Kharif crops — 2% of sum insured; Rabi — 1.5%. Coverage: Drought, flood, hailstorm, cyclone, pest/disease. Enrollment through your bank or Common Service Centres (CSC). Deadline: Typically 2–3 weeks before crop season.
>
> For enrollment help, visit your nearest CSC or contact the Kisan Call Centre.

---

### Scenario 6: Out-of-Distribution Image Detection

**User:** Uploads a photo of their tractor instead of a crop leaf.

**System behavior:** The disease detector's OOD detection recognizes that this image does not resemble any plant/leaf in its training distribution. The VLM layer identifies it as agricultural equipment and responds using its general knowledge rather than forcing a disease classification.

**User sees:**

> This doesn't appear to be a photo of a crop or plant leaf. I can identify diseases from clear photos of affected plant parts. Please upload a close-up photo of the affected leaf, stem, or fruit.

Most systems confidently classify any input into some disease category — recognizing "this isn't even a leaf" is a critical safety differentiation.

---

### Scenario 7: Context-Aware Personalized Response

**User:** "What should I spray for early blight on tomato?" The system collects context: Rajasthan, Flowering stage, Organic preference.

**User sees:**

> For organic management of Early Blight at flowering stage in Rajasthan: Apply Trichoderma viride at 4 g/litre as a foliar spray. Alternatively, neem oil 5% at 5 ml/litre. Avoid spraying during peak flowering hours (morning 8–11 AM) to protect pollinators. *(Source: ICAR organic farming package)*

Had the farmer selected "Chemical," the response would recommend Mancozeb 75% WP at 2.5 g/litre with a different source citation. The same architecture personalizes by location, crop stage, and farming method.

---

### Scenario 8: Honest System Boundary

**User:** "What should I plant next season?"

**System behavior:** This crop planning query requires soil data, historical yields, market forecasts, and agro-climatic zone information — beyond current scope. The system redirects honestly.

**User sees:**

> Choosing the best crop for next season depends on your soil type, water availability, climate zone, and market prices — factors I don't have full data on right now.
>
> For personalized crop planning, I'd recommend contacting your local KVK for soil-testing and crop recommendations, checking the ICAR crop calendar for your agro-climatic zone, and reviewing market trends on the eNAM portal.
>
> I'm best at helping with crop disease identification, pest management advice, and government scheme information — ask me about those anytime!

---

## 9. Literature Review & Existing Solutions

### 9.1 AI-Powered Agricultural Advisory Systems

#### 9.1.1 KisanQRS (Rana et al., 2023) — Closest Prior Work

KisanQRS is a deep learning-based automated query-response system trained on a subset of 34 million KCC call logs. It uses a threshold-based clustering algorithm to group similar queries and an LSTM model for query mapping, achieving a top F1-score of 96.58% on query mapping across five major Indian states, and an NDCG score of 96.20% for answer retrieval.

**Strengths:** Large-scale validation on real KCC data; strong retrieval ranking performance; efficient clustering-based architecture.

**Weaknesses:** Returns clustered past answers rather than generating natural-language responses — no LLM-based answer generation. Time-sensitive queries (market prices, weather) are explicitly filtered out during preprocessing, meaning the system cannot handle price queries at all. No image-based disease detection component. No multi-source routing (cannot combine disease + price + scheme information for a single query). No multilingual or code-switched (Hinglish) query handling addressed explicitly.

#### 9.1.2 Agentic AI for Smart Agriculture: Multimodal RAG System (ReadyTensor, 2025)

This system combines YOLOv8 for image-based crop disease detection with a RAG pipeline using Qdrant vector search and Groq-powered LLMs for contextual reasoning and treatment recommendations. It operates across multiple languages and can refine responses based on user feedback.

**Strengths:** Combines computer vision and RAG effectively; multilingual support; autonomous refinement based on feedback.

**Weaknesses:** Does not use KCC data as a knowledge backbone (relies on generic agricultural knowledge base). No query intent classification. No live market price integration. Not specifically tailored to Indian farming context or Indian government data sources.

#### 9.1.3 Krishi Sathi and Multi-Turn Dialogue Systems

Krishi Sathi demonstrates excellent multi-turn dialogue capabilities for farmer advisory, with agent-initiated dialogue state tracking to proactively ask clarifying questions when a farmer's query is ambiguous. However, it lacks field-tested computer vision for disease detection.

#### 9.1.4 AI Chatbot for Farmers Using RAG and Leaf Image Analysis (2025)

A recent system combining RAG with leaf image analysis reported a 4.4/5 satisfaction rating in a user study involving 200 farmers. It includes multilingual and voice-enabled support aimed at users with varying literacy levels.

**Strengths:** Validated with real farmer user study; multilingual and voice support; positive user satisfaction.

**Weaknesses:** Does not use KCC data specifically. No explicit query intent classification. No live mandi price integration. Methodology for disease-to-treatment grounding not clearly tied to verified expert sources.

#### 9.1.5 RAG-Based Agricultural Advisory Frameworks (2024–2025)

Several recent works propose RAG pipelines over agricultural documents for farmer advisory (Balpande et al., 2024; RAG-based LLM advisory framework, 2025), evaluating retrieval and generation times, semantic similarity, and source attribution scores across crops and query types.

**Strengths:** Strong retrieval evaluation methodology; demonstrate feasibility of RAG for agricultural domains.

**Weaknesses:** General-purpose, not grounded in Indian-specific KCC data. Do not address multi-intent queries or live data integration (prices, schemes).

#### 9.1.6 FarmerChat (Digital Green / IFPRI GAIA Project)

FarmerChat is the most widely deployed GenAI agricultural advisory system, serving over 830,000 users across five countries with 15 language support and over 10 million queries processed. Farmers interact through voice, text, or pictures in their local languages.

**Strengths:** Massive scale validation; multilingual voice support; real farmer adoption metrics.

**Gaps relative to our work:** Published benchmarks on disease detection field-robustness are limited. No public evaluation of hallucination rates or faithfulness metrics. System architecture and evaluation methodology are not fully open for academic comparison.

#### 9.1.7 Government Initiatives: Kisan e-Mitra and Bharat-VISTAAR

India's Kisan e-Mitra voice-based chatbot responds to over 20,000 farmer queries daily in 11 regional languages. The proposed Bharat-VISTAAR (Union Budget 2026-27) aims to integrate AgriStack portals with AI systems for customized advisory. The National Pest Surveillance System uses AI for pest identification across 61 crops and 400+ species with 10,000+ extension agents.

**Relevance:** These represent the institutional direction India is moving — our project aligns with this trajectory while addressing the technical gaps (field-robustness, faithfulness) that government systems have not yet solved.

### 9.2 Computer Vision for Crop Disease Detection

#### 9.2.1 The Lab-to-Field Domain Gap — The Central Problem

Standard convolutional neural networks (like EfficientNet-B0) show 99% accuracy on clean, lab-generated datasets such as PlantVillage but suffer catastrophic failure on real-world field datasets. The original 2016 PlantVillage paper found accuracy fell to approximately 41% on real-world images. The causes are well-documented: complex backgrounds, variable lighting, tiny symptom regions, and overlapping foliage that lab images do not capture.

#### 9.2.2 Architectural Interventions for Field Robustness

The literature identifies several approaches to bridging this gap:

**Segmentation-first approaches:** Using BERT-based models or UNet to create segmentation masks before classification via ResNet. Architectures like STAR-Net handle complex fungal lesion morphologies by isolating the symptomatic region from the background before classification.

**Domain adaptation techniques:** Fourier Domain Adaptation and sim2real transfer bridge the synthetic-to-field gap by aligning feature distributions between clean and noisy image domains.

**Rapid edge detection:** Single-stage models like YOLOv8 provide rapid bounding-box detection suitable for mobile deployment, enabling real-time disease localization.

**Vision-Language Models (VLMs):** Frameworks like SCOLD use a dual-stream architecture (Swin-T and RoBERTa) with Context-Aware Soft Targets (CST) to turn classification into an image-text retrieval task with improved accuracy.

#### 9.2.3 Available Real-Field Datasets

| Dataset | Size | Crops / Classes | Characteristics |
|---|---|---|---|
| Cassava Leaf Disease (Makerere/Kaggle) | 21,367 images | 1 crop, 5 classes | Mobile phone photos taken by Ugandan farmers, expert-labeled — matches deployment scenario |
| PlantWild (2024) | 18,542 images | 33 species, 89 classes | Largest in-the-wild multi-crop set with text descriptions per class (multimodal) |
| FieldPlant (2023) | 5,170 images | 27 classes | Field images annotated by plant pathologists |
| PlantDoc (2020) | 2,598 images | 13 crops, 17 disease classes | Standard field benchmark capturing real-field conditions unlike PlantVillage |
| PlantSeg (2025) | Large | Multi-crop | In-the-wild with segmentation masks |

These public, expert-labeled datasets make field-robust evaluation feasible without collecting original data.

### 9.3 NLP for Indian Agricultural Context

#### 9.3.1 The Linguistic Challenge

Standard multilingual models like mBERT fail on transliterated and code-mixed Indian vernacular. Farmers write queries in Hinglish (Hindi-English code-mixing), Romanized Hindi, and regional dialects that general-purpose models are not equipped to handle. Agricultural terminology in Indian languages — terms like "tana chhedak" (stem borer), "phaphund" (fungus), "geela sadan" (wet rot), "davai" (pesticide) — is systematically underrepresented in the training data of standard multilingual models.

#### 9.3.2 Specialized Models for Indian Languages

Models pre-trained on Indian languages — MuRIL (Google), IndicBERT (AI4Bharat), and hybrid frameworks like BhaavNet — achieve significantly higher accuracy when fine-tuned on historical KCC logs and specialized corpora like AgriGov for intent classification and Named Entity Recognition (NER).

### 9.4 Context-Aware Retrieval and Sensor Integration

Generic RAG is insufficient and potentially dangerous for agriculture because advice must be hyper-personalized based on geolocation, weather, and soil type. Frameworks like GeoGraphRAG-Soil (GGRS) and AgriRegion intersect textual knowledge with geospatial rules. Vector databases with strict pre-retrieval metadata filtering (filtering by region, soil type, crop) ensure contextual safety.

### 9.5 Faithfulness and Hallucination in Agricultural AI

The RAGAS evaluation framework provides metrics for continuous safety monitoring: Context Precision and Recall (evaluating signal-to-noise ratio and completeness of retrieved documents), Faithfulness (the ultimate guardrail — ensuring every generated claim traces back to retrieved context), and Response Relevancy (ensuring outputs directly answer the user's prompt). Multi-agent validation systems using evaluator agents to verify outputs before they reach the user are proposed as neurosymbolic guardrails against hallucination.

### 9.6 Voice Interfaces and Low-Literacy Access

For low-literacy users, zero-shot interaction (expecting a perfect text prompt) fails. Multi-turn clarification with agent-initiated dialogue state tracking is essential. Voice-first design integrating robust Speech-to-Text (IndicWhisper) and Text-to-Speech (Kokoro-82M, Sarvam TTS) is mandatory, delivered via familiar channels like WhatsApp or IVR so farmers can listen repeatedly without needing to read complex instructions. The Wadhwani Institute's AgriAI Collect demonstrated this approach successfully, using automatic speech recognition for multilingual voice inputs combined with LLMs for structured extraction and human-in-the-loop validation, onboarding 32,000 users.

### 9.7 Industry Solutions

**Syngenta Cropwise:** A digital farm management platform deployed across over 70 million hectares in 30+ countries, representing the state of the art in precision agriculture at scale. However, it is designed for large-scale commercial operations, not smallholder contexts with low connectivity and literacy constraints.

**Plantix (PEAT):** Uses AI and image recognition for smartphone-based disease diagnosis for smallholder farmers in India and other developing countries. A 2025 study testing an AI diagnostic tool for cassava diseases in Burkina Faso found that while the tool increased survey coverage, its diagnostic accuracy against molecular analysis showed significant limitations — highlighting the lab-to-field gap even in deployed systems.

**Intello Labs (Fruitsort):** AI and computer vision for produce quality assessment, achieving 40x faster processing than manual sorting. Demonstrates commercial viability of agricultural computer vision but operates in post-harvest assessment rather than in-field advisory.

---

## 10. Comparative Analysis

### Feature Comparison Matrix

| Feature | KisanQRS | ReadyTensor Agentic AI | Krishi Sathi | AI Chatbot (RAG + Leaf) | FarmerChat | **Our System** |
|---|---|---|---|---|---|---|
| KCC data backbone | ✅ (34M logs) | ❌ | ❌ | ❌ | ❌ | ✅ |
| LLM-based response generation | ❌ (clustering only) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Image-based disease detection | ❌ | ✅ (YOLOv8) | ❌ | ✅ | ✅ | ✅ (field-robust) |
| Field-robustness evaluation | ❌ | ❌ | ❌ | ❌ | Not public | ✅ (PlantDoc, Cassava) |
| Confidence-aware abstention | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Faithfulness metrics (RAGAS) | ❌ | ❌ | ❌ | ❌ | Not public | ✅ |
| Live mandi price integration | ❌ (filtered out) | ❌ | ❌ | ❌ | ❌ | ✅ (Agmarknet API) |
| Multi-source routing | ❌ | ❌ | ❌ | ❌ | Partial | ✅ |
| Indian vernacular / Hinglish | ❌ | ❌ | ❌ | Partial | ✅ (15 languages) | ✅ (MuRIL/IndicBERT) |
| Multi-turn dialogue | ❌ | Partial (feedback) | ✅ | Partial | ✅ | ✅ |
| Voice I/O | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ (IndicWhisper + TTS) |
| Domain-adapted embeddings | ❌ | ❌ | ❌ | ❌ | Not public | ✅ (fine-tuned on KCC) |
| Context elicitation (location, crop, method) | ❌ | ❌ | ✅ | ❌ | Partial | ✅ |
| Tiered response (RAG + LLM fallback) | ❌ | ❌ | ❌ | ❌ | Not public | ✅ |
| Government scheme integration | ❌ | ❌ | ❌ | ❌ | Partial | ✅ |

### Key Differentiators of Our Approach

**1. Field-robustness as a measured claim, not an assumption.** We train on PlantVillage and evaluate on PlantDoc/Cassava/FieldPlant, explicitly quantifying the domain gap and demonstrating recovery through augmentation — a step no comparable system publicly reports.

**2. Confidence calibration and abstention.** When the disease detector is uncertain, the system says "I'm not sure" rather than guessing. This is measurable (abstention rate, abstention correctness) and a genuine safety contribution.

**3. Faithfulness as a quantitative metric.** We measure whether generated advice traces back to retrieved sources using RAGAS metrics, addressing the most dangerous failure mode in agricultural AI — hallucinated treatment advice.

**4. Domain-adapted retrieval.** Fine-tuning the embedding model on agricultural Hindi/regional terminology improves retrieval quality in a way that generic multilingual models cannot achieve. This is empirically verifiable (before/after retrieval recall).

**5. Synthesis of complementary strengths.** Our system combines KCC data grounding (KisanQRS's strength), effective CV+RAG integration (ReadyTensor's strength), multi-turn dialogue (Krishi Sathi's strength), and real farmer accessibility (FarmerChat's strength) — while adding field-robustness evaluation, faithfulness metrics, and confidence-aware abstention that none of them individually provide.

---

## 11. Datasets and Evaluation Framework

### Training and Evaluation Datasets

| Dataset | Purpose | Size | Source |
|---|---|---|---|
| KCC Query-Response Dataset | Intent classification training, RAG corpus, embedding model fine-tuning | Lakhs of labeled Q&A pairs (crop, state, category) | data.gov.in |
| PlantVillage | Disease detector training (lab-condition baseline) | 54,306 images, 38 classes | Public dataset |
| PlantDoc | Disease detector evaluation (field conditions) | 2,598 images, 13 crops, 17 disease classes | Public dataset |
| Cassava Leaf Disease | Disease detector evaluation (farmer-taken mobile photos) | 21,367 images, 5 classes | Kaggle / Makerere University |
| PlantWild (2024) | Multimodal field evaluation | 18,542 images, 89 classes | Public dataset |
| ICAR Advisories | RAG knowledge base | Hundreds of advisory documents | ICAR publications |
| Government Scheme Documents | RAG knowledge base (scheme queries) | PMFBY, PM-KISAN, state scheme documents | Government portals |

### Evaluation Metrics

**Disease Detector (Trainable Model 1):**
- Accuracy, macro-F1, per-class F1 on PlantVillage (lab) vs. PlantDoc/Cassava (field) — quantifying the domain gap
- Confusion matrix analysis — which diseases are most commonly confused
- Abstention rate and abstention correctness (percentage of low-confidence predictions that were genuinely ambiguous)
- Expected Calibration Error (ECE) — measuring reliability of confidence scores

**Domain-Adapted Embedding Model (Trainable Model 2):**
- Recall@5, Recall@10, Mean Reciprocal Rank (MRR), nDCG — comparing generic vs. fine-tuned embeddings
- Query-type breakdown — which agricultural categories improved most from domain adaptation
- Failure case analysis — which queries still retrieve incorrect documents and why

**RAG Advisory (End-to-End System):**
- RAGAS metrics: Context Precision, Context Recall, Faithfulness, Response Relevancy
- Tier distribution: percentage of queries resolved at Tier 1 (grounded), Tier 2 (LLM knowledge), Tier 3 (out-of-scope)
- Manual verification of Tier 2 responses for factual correctness (sample of 200+ responses)

**Faithfulness Evaluation (Optional M5 Extension):**
- Train a lightweight faithfulness classifier on 150–200 manually labeled (context + response) pairs
- Deploy to evaluate faithfulness at scale across 1,000+ system outputs
- Report quantitative faithfulness score: "94% of responses were grounded in retrieved sources"

---

## 12. Milestone 1 Scope — Inclusions and Exclusions

### Milestone 1 Deliverables

| Deliverable | Status |
|---|---|
| Problem statement with measurable objectives | ✅ Defined |
| Literature review of current solutions and academic work | ✅ Completed |
| Strengths/weaknesses analysis of existing approaches | ✅ Completed |
| Comparative analysis with proposed approach | ✅ Completed |
| Performance benchmarks and evaluation metrics identified | ✅ Defined |
| System architecture design | ✅ Designed |
| Dataset identification and availability confirmed | ✅ Confirmed |
| References from credible sources | ✅ Included |

### What Milestone 1 Includes

- Multimodal ingestion pipeline design (text, image, voice)
- Intent routing architecture with LLM-based query segmentation
- RAG pipeline design over ICAR/AgriGov/KCC data
- Live API integration specification (Agmarknet for mandi prices, weather APIs)
- Multi-turn voice dialogue design
- Two trainable model specifications (disease classifier, embedding model)
- Evaluation framework with quantitative metrics

### What Milestone 1 Excludes

- Physical farm actuation or autonomous field operations
- Live video feed processing
- Legal dispute handling
- Local/on-device LLM deployment
- Crop planning, yield prediction, or soil health analysis
- Satellite imagery integration

---

## 13. References

1. Rana, A., et al. (2023). "KisanQRS: A Deep Learning-Based Automated Query-Response System for Kisan Call Centre." — F1 score of 96.58% on query mapping across five Indian states using KCC call logs.

2. ReadyTensor (2025). "Agentic AI for Smart Agriculture: Multimodal RAG System." — YOLOv8 + Qdrant + Groq LLM for crop disease detection and advisory.

3. Hughes, D., & Salathé, M. (2016). "An open access repository of images on plant health to enable the development of mobile disease diagnostics." (PlantVillage) — Foundational dataset; accuracy dropped to ~41% on real-world images.

4. Singh, D., et al. (2020). "PlantDoc: A Dataset for Visual Plant Disease Detection." — 2,598 real-field images, 13 crops, 17 disease classes.

5. Mwebaze, E., et al. "Cassava Leaf Disease Classification Dataset." Kaggle / Makerere University. — 21,367 farmer-taken mobile phone images from Uganda.

6. PlantWild (2024). — 18,542 in-the-wild images across 33 species and 89 classes with multimodal text descriptions.

7. Balpande, A., et al. (2024). "AI-Powered Agriculture Optimization Chatbot Using RAG and GenAI."

8. Digital Green. "FarmerChat." — 830,000+ users, 15 languages, 10M+ queries across 5 countries.

9. IFPRI. "GAIA Phase II (2025–2027): Generative AI for Agricultural Information Access."

10. World Economic Forum (2025). "Shaping the Deep-Tech Revolution in Agriculture." — Barriers: data quality, hallucinations, on-field variability, model transferability.

11. World Economic Forum (2025). "How AI is Enabling Agricultural Intelligence and Revolutionizing Farming."

12. Government of India. "Kisan e-Mitra: Voice-based AI chatbot." — 20,000+ daily queries, 11 languages, 95 lakh+ questions answered.

13. Government of India, Union Budget 2026-27. "Bharat-VISTAAR: Multilingual AI tool for customized agricultural advisory."

14. Wadhwani Institute for AI. "AgriAI Collect." — ASR + LLM + human-in-the-loop for agricultural data collection; 32,000 users.

15. Syngenta. "Cropwise Digital Farm Management Platform." — 70M+ hectares, 30+ countries.

16. Khurana, S., et al. "MuRIL: Multilingual Representations for Indian Languages." Google Research.

17. Kakwani, D., et al. "IndicBERT: A Multilingual Model for Indic Languages." AI4Bharat.

18. Es, S., et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." — Context Precision, Recall, Faithfulness, Response Relevancy metrics.

19. Intello Labs. "Fruitsort: AI-based Produce Quality Assessment." — 40x faster than manual sorting using computer vision.

20. National Pest Surveillance System (India). — AI-based pest identification across 61 crops, 400+ species, 10,000+ extension agents.

---

*Document prepared for Milestone 1 submission. Architecture diagram reflects the updated multimodal routing pipeline as designed by the project team.*
