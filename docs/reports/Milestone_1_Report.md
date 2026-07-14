# Milestone 1 — Problem Definition & Literature Review

## A Decoupled, Agentic Multimodal Crop Advisory System for Uttar Pradesh

### Contents

 1. [Problem Statement](#1-problem-statement)
 2. [Scope & Boundaries](#2-scope--boundaries)
 3. [Stakeholders](#3-stakeholders)
 4. [Measurable Objectives](#4-measurable-objectives)
 5. [Proposed Solution](#5-proposed-solution)
 6. [System Architecture](#6-system-architecture)
 7. [Disease Detector — Architecture & Calibration](#7-disease-detector--architecture--calibration)
 8. [Literature Review & Existing Solutions](#8-literature-review--existing-solutions)
 9. [Comparative Analysis](#9-comparative-analysis)
10. [Datasets & Evaluation](#10-datasets--evaluation)
11. [Risk Assessment & Mitigation](#11-risk-assessment--mitigation)
12. [Ethical Considerations](#12-ethical-considerations)
13. [System Limitations](#13-system-limitations)
14. [Computational Requirements](#14-computational-requirements)
15. [References](#15-references)
16. [Appendix: System Scenarios](#16-appendix-system-scenarios)

### 1. Problem Statement

Smallholder farmers in Uttar Pradesh (UP), India, rely on timely agronomic intelligence to protect yields of staple crops like rice and wheat. While institutional lifelines such as the Kisan Call Centre (KCC) exist, they are constrained by human bandwidth, text-only interfaces, and the inability to visually diagnose crop diseases.

Recent advancements in Large Language Models (LLMs) and Computer Vision present opportunities to automate this advisory process, but existing monolithic AI solutions fail in three critical, measurable areas:

1. **The Vision Domain Gap:** Models trained on idealized laboratory datasets (e.g., PlantVillage) routinely achieve $> 98\%$ accuracy, but experience severe performance degradation (falling as low as $\sim 40 - 50\%$) when deployed on real-world field images characterized by complex backgrounds and poor lighting [1, 2].

2. **Generative Hallucination in Agronomy:** Standard LLMs lack localized domain grounding, often hallucinating chemical dosages or referencing outdated government schemes, posing a direct financial and chemical risk to farmers.

3. **Linguistic Incompatibility:** Standard embedding models fail to accurately map transliterated, code-mixed agricultural Hindi (e.g., “makka mein tana chhedak”), resulting in poor context retrieval for regional queries [3].

This project proposes a decoupled, agentic architecture that isolates vision classification from reasoning, utilizes Indian-specific embedding models (MuRIL) for accurate retrieval, and employs a quantized LLM acting as a reasoning agent to trigger external predictive tools and synthesize factually grounded advice.

### 2. Scope & Boundaries

To ensure rigorous evaluation and computational feasibility, the system's knowledge boundaries are strictly localized.

**Must-Have (Core Requirements):**

* **Geographic & Crop Scope:** Uttar Pradesh state data; exclusively targeting Rice and Wheat.

* **Decoupled Multimodal Input:** Independent processing of text (Hindi/English) and images.

* **Agentic Routing & Tool Calling:** An LLM agent capable of multi-turn context elicitation and executing external Python functions.

* **Localized RAG:** A vector database embedded via MuRIL, indexing strictly filtered UP Government scheme documents and localized KCC agronomic Q&A logs.

**Should-Have (Secondary Requirements):**

* **Yield Prediction Tool:** A classical machine learning model estimating district-level yield based on historical UP data, callable by the agent.

* **Integration of live APIs:** Real-time Mandi (market) prices via Agmarknet.

**Stretch Goals:**

* **Cross-Lingual Knowledge Transfer:** Expanding the vector indexing and metadata filtering to include one neighboring state (e.g., Bihar) to test architectural scalability.

* **Native Voice Input:** Processing raw local dialect audio directly through the LLM’s multimodal capabilities (e.g., Gemma 4 12B audio ingestion).

**Out of Scope (with justifications):**

* **Unified Vision-Language Model (VLM) fine-tuning:** Fine-tuning massive multimodal models exceeds the 16GB VRAM limit of available consumer-grade hardware and the 5-6 week project timeline. A decoupled architecture is utilized instead to ensure feasibility.

* **Physical farm actuation (irrigation control):** The project focuses strictly on software-based agronomic intelligence. Hardware actuation introduces physical safety liabilities and requires IoT integration outside the software scope.

* **Live satellite imagery processing:** Ingesting and processing real-time geospatial data requires high-bandwidth pipelines that would dilute the core focus on conversational RAG and localized edge-case vision.

* **Pan-India dataset coverage:** To ensure high retrieval precision, eliminate cross-state policy hallucination, and manage data-cleaning overhead, the knowledge base is strictly bounded to Uttar Pradesh for the pilot phase.

### 3. Stakeholders

* **Primary:** Smallholder farmers in Uttar Pradesh cultivating staple crops (Rice and Wheat).

* **Secondary:** Kisan Call Centre (KCC) operators and local Krishi Vigyan Kendra (KVK) extension officers utilizing the system as a supportive diagnostic and policy-retrieval tool.

* **Tertiary:** UP State Government and Agricultural Departments benefiting from modernized, scalable digital interventions.

### 4. Measurable Objectives

| Objective | Description | Target Metric | 
 | ----- | ----- | ----- | 
| **O1: Field-Robust Vision** | Narrow the lab-to-field performance gap for Rice/Wheat disease detection. | Achieve Macro F1-Score $\ge 0.85$ on in-the-wild datasets (e.g., PlantDoc) [2]. | 
| **O2: Calibrated Abstention** | Prevent confident misclassification of out-of-distribution or poor-quality images. | Maintain False Discovery Rate (FDR) $\le 10\%$ via threshold abstention. | 
| **O3: Retrieval Efficacy** | Accurately map code-mixed vernacular queries to appropriate documents. | Achieve Recall@5 $\ge 0.85$ using MuRIL embeddings. | 
| **O4: Grounded Generation** | Eliminate hallucination in treatment and policy advice. | Achieve RAGAS Faithfulness score $\ge 0.90$ [4]. | 
| **O5: Yield Estimation** | Provide accurate localized yield predictions via tool calling (if implemented). | Achieve RMSE within $15\%$ of actual historical UP district averages (R-squared $\ge 0.80$). | 

### 5. Proposed Solution

The system departs from standard monolithic conversational AI by utilizing a Decoupled Agentic Workflow.

* **Semantic Router (FastAPI):** Orchestrates incoming payloads. Images are routed exclusively to a fine-tuned Convolutional Neural Network (CNN). The CNN outputs a text label (e.g., `[Wheat Leaf Rust, Conf: 91%]`), which is appended to the user's text query.

* **Vector DB & Embedder (MuRIL):** UP government PDFs and filtered KCC logs (excluding outdated financial data) are chunked and vectorized using MuRIL, which specializes in projecting transliterated Indian languages into a shared semantic space [3].

* **Agentic LLM (Gemma-based):** A quantized, instruction-tuned LLM acts as the reasoning engine. Utilizing a ReAct (Reasoning and Acting) prompt structure, it evaluates the enriched text prompt, identifies missing variables (e.g., soil type), elicits context from the user, and triggers either RAG retrieval or the Yield ML tool (if available) before synthesizing a final answer.

### 6. System Architecture

![Architechture](<./../architecture/FarmerVisionServiceArch.png> "Proposed Solution")


**Data Flow Summary:** The User interface sends a text/image payload to the Backend Router. The image is processed by the Vision Service, returning a text label. The unified text payload enters the Agentic LLM. The LLM applies reasoning to determine if it must query the MuRIL-backed Vector DB or execute the Yield Prediction Tool (Should-Have). It gathers all context and generates a multilingual response for the user.

### 7. Disease Detector — Architecture & Calibration

To address the lab-to-field domain gap without exceeding local compute constraints, we utilize a dedicated, lightweight vision architecture rather than a heavy VLM.

**Candidate Backbones:**

* **EfficientNet-B0:** Shortlisted for its highly optimized parameter-to-accuracy ratio, ideal for constrained environments.

* **YOLOv8 (Classification Mode):** Shortlisted for its rapid inference latency and robust feature extraction in noisy image environments.

**Confidence Calibration & Abstention (**$\tau$**):**
Standard CNNs are often overconfident in their predictions. To enable graceful abstention (refusing to diagnose blurry or out-of-distribution photos), we will apply Temperature Scaling to the output logits to align probabilities with true accuracy.
An abstention threshold ($\tau$) will be calibrated using a holdout validation set. The optimal $\tau$ will be selected dynamically to maximize the F1-score while strictly constraining the False Positive Rate to $< 10\%$. If the maximum softmax probability $P_{max} < \tau$, the system outputs an "uncertain" label, prompting the LLM agent to ask the farmer for a clearer photo.

### 8. Literature Review & Existing Solutions

Recent academic focus on agricultural AI identifies three core challenges parallel to our proposed architecture: visual domain shifts, linguistic code-mixing, and generative hallucination.

**8.1 The 'Lab-to-Field' Domain Gap in Crop Disease Classification**
Deep learning architectures consistently achieve $> 99\%$ accuracy on idealized laboratory datasets like PlantVillage. However, in-situ field images (e.g., PlantDoc) introduce statistical distribution shifts—cluttered backgrounds, overlapping foliage, and harsh lighting—often causing accuracy to plummet below $40\%$ [2].
Recent cross-domain few-shot learning (CD-FSL) methods have emerged to bridge this gap. Zhao et al. (2026) demonstrated that Target-Domain Statistical Calibration during training, combined with lightweight extractors like EfficientNet-B0, can align source and target feature distributions without computationally expensive test-time optimization [6]. Furthermore, Quilondrino et al. (2025) mitigated field noise by proposing a Multi-Stage Hybrid Classification framework that combines deep neural features with traditional Gray-Level Co-occurrence Matrix (GLCM) texture features, forcing models to focus on structural anomalies rather than background context [7]. Similarly, Sharif et al. (2026) introduced Federated Multimodal Edge Learning (FMEL-FSDA) to achieve privacy-preserving, resilient disease detection with minimal labeled field samples [8]. Our proposed decoupled CNN architecture with temperature-scaled abstention directly builds upon these findings to prioritize field robustness over raw laboratory metrics.

**8.2 Indic Language Embeddings in Code-Mixed Contexts**
The digital "vernacular divide" severely limits the efficacy of standard Natural Language Understanding (NLU) systems in rural India. Farmers frequently employ code-mixing, blending Hindi, local dialects (e.g., Awadhi), and English in Romanized scripts (Hinglish).
Traditional embedding models like mBERT struggle with such inputs due to high "fertility ratios"—fragmenting words into meaningless sub-tokens. Research by Ingale and Margaj (2025) confirms that specialized Indic models like MuRIL significantly outperform standard transformers in code-mixed intent classification, achieving accuracies exceeding $87\%$ [9]. Recent applications, such as Krishi Mitra (Gajre et al., 2026) and FarmSaarthi (Gautam et al., 2026), have successfully leveraged MuRIL-based sentence transformers within Retrieval-Augmented Generation (RAG) pipelines to semantically map vernacular queries to official agricultural schemes and market data [10, 11]. By adopting MuRIL for our semantic router and vector indexing, the proposed system ensures high retrieval fidelity for code-mixed agronomic inquiries.

**8.3 Agentic Workflows & Hallucination Mitigation in Agronomy**
Deploying parametric-only Large Language Models (LLMs) in agriculture poses severe risks due to generative hallucinations (e.g., fabricating chemical dosages). The paradigm is rapidly shifting from passive Generation 1 RAG systems to autonomous Agentic AI frameworks utilizing the ReAct (Reasoning and Acting) methodology [5].
Mandiga et al. (2026) demonstrated with NutriCHAT that forcing an LLM to utilize expert-designed API tools (Tool Calling) to fetch exact parameters reduces hallucination scores by over $61\%$ [12]. Additionally, frameworks like AgriIR (Seal et al., 2026) highlight the efficacy of domain-aware retrieval over relying solely on model size, enabling highly accurate outputs even from efficient 1B-parameter models [13]. For multi-faceted agricultural support, systems like NeerVaani (Singh et al., 2025) successfully utilize multi-agent orchestration, routing tasks to specialized Worker agents (e.g., Vision, Weather) [14]. Our architecture incorporates these principles by treating the LLM not as a generalized knowledge base, but as a ReAct reasoning agent strictly bound by external tools (Yield ML, Localized RAG) and safety guardrails.

### 9. Comparative Analysis

| Capability | KisanQRS | Generic GenAI Agri-Bots | Our Proposed System | 
 | ----- | ----- | ----- | ----- | 
| **KCC Data Integration** | Yes | No | Yes (Filtered RAG) | 
| **Independent Field-Robust Vision** | No | No (Often relies on monolithic VLMs) | Yes (Decoupled CNN) | 
| **Indian-Language Optimized Retrieval** | No | No (Standard mBERT/Ada) | Yes (MuRIL) | 
| **Agentic Tool Calling (Yield ML)** | No | No | Yes (Should-Have) | 
| **Calibrated CV Abstention** | No | No | Yes | 

### 10. Datasets & Evaluation

**Proposed Datasets**

* **Vision Training:** PlantVillage [1] (Lab baseline) augmented with targeted field-noise.

* **Vision Evaluation:** PlantDoc [2] / Cassava Leaf Disease datasets (In-the-wild baseline).

* **NLP/RAG Corpus:** Uttar Pradesh filtered subset of the Government of India KCC dataset (strictly agronomic queries) + 50-100 regional UP policy PDFs.

* **Yield Prediction:** Historical district-level crop data (yield, rainfall, fertilizer usage) sourced from data.gov.in for Uttar Pradesh.

**RAG Tiered Relevance Thresholds**
To ensure faithfulness, the system will evaluate the cosine similarity of MuRIL embeddings during retrieval:

* **Tier 1 (Grounded):** Relevance score $\ge 0.85$. The LLM generates a confident response strictly citing the document.

* **Tier 2 (Transparent Fallback):** Relevance score $0.65 - 0.84$. The LLM generates advice based on broad agronomic knowledge but prepends a mandatory disclaimer advising local verification.

* **Tier 3 (Abstention/Redirect):** Relevance score $< 0.65$. The agent classifies the query as out-of-scope and refuses to generate agronomic advice.

### 11. Risk Assessment & Mitigation

| Risk Factor | Impact | Mitigation Strategy | 
 | ----- | ----- | ----- | 
| **KCC Dataset Noise/Outdated Info** | High | Apply strict temporal filtering; exclude financial/subsidy data from KCC logs, relying only on official PDFs for policy. | 
| **LLM Context Window Exhaustion** | Medium | Implement strict max-token limits on RAG chunk retrieval (Top-$K=3$) to fit within the quantized model's constraints. | 
| **Hallucination in Translation** | Medium | Utilize RAGAS Faithfulness metrics during testing to ensure the LLM's Hindi output logically matches the retrieved Hindi context. | 

### 12. Ethical Considerations

* **Transparency of AI Involvement:** The system interface will clearly state that advice is AI-generated and should not replace emergency consultation with a local KVK officer.

* **Chemical Safety:** The prompt engineering will strictly forbid the LLM from synthesizing novel chemical mixtures, forcing it to only output dosages exactly as written in the retrieved ICAR/KCC documentation.

### 13. System Limitations

* **Narrow Scope:** The system is explicitly constrained to Rice and Wheat in UP. Queries regarding cash crops (e.g., Sugarcane) will trigger the Tier 3 out-of-scope redirect.

* **No Hardware Actuation:** The system provides advice and predictive yield estimates but cannot interface with farm IoT hardware for automated irrigation or spraying.

* **Text/Image Dependency (Core):** While voice is a stretch goal, the core PoC requires a baseline level of literacy and internet connectivity to upload images and read text responses.

### 14. Computational Requirements

The architecture is explicitly designed to operate within consumer-grade and free-tier cloud constraints:

* **Vision Fine-Tuning:** Executable on a Kaggle free-tier NVIDIA T4 GPU (16GB VRAM) within 4-6 hours due to the lightweight nature of EfficientNet/YOLOv8.

* **LLM Deployment:** The reasoning agent (e.g., Gemma-4 4B-it / 12B-it variant) will be loaded using 4-bit quantization (bitsandbytes), reducing its memory footprint to 3.5 GB / 6.7 GB VRAM, leaving ample overhead for the FastAPI router and in-memory ChromaDB instance.

* **Target Inference Latency:** Vision classification $\le 1.5$ seconds; Agentic RAG text generation $\le 4.0$ seconds.

### 15. References

[1] **Hughes, D., Salathé, M.** (2016). *PlantVillage — An open access repository of images on plant health*.

[2] **Singh, D., et al.** (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection*. ACM CoDS-COMAD.

[3] **Khanuja, S., et al.** (2021). *MuRIL: Multilingual Representations for Indian Languages*. Google Research.

[4] **Es, S., et al.** (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*.

[5] **Yao, S., et al.** (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR.

[6] **Zhao, C., Xu, T., Zhang, Z., Geng, X.** (2026). *Lightweight Cross-Domain Few-Shot Plant Disease Recognition Through Target-Domain Statistical Calibration*. MDPI.

[7] **Quilondrino, I. C. T., Patan, G. V., Pitao, J. V. S.** (2025). *Mitigating Accuracy Loss in Plant Disease Detection: A Comparative Study of Multi-Stage Hybrid Classification Frameworks for Field Conditions*. Kyushu University.

[8] **Sharif, M. I., Zhong, Y., Sajid, M. Z., Marinello, F.** (2026). *Real-Field-Ready and Digitally Sustainable Plant Disease Recognition via Federated Multimodal Edge Learning and Few-Shot Domain Adaptation*. MDPI.

[9] **Ingale, O., Margaj, S.** (2025). *Comparative Analysis of Embedding Models for Hindi-English Code-Mixed University related queries*.

[10] **Gajre, J., et al.** (2026). *Krishi Mitra: A Multilingual AI-Powered Conversational Agent for Indian Farmers Integrating Government Schemes*. IJARCCE.

[11] **Gautam, L., et al.** (2026). *FarmSaarthi: A Vernacular-Enabled Digital Agricultural Ecosystem with Multimodal AI for Precision Farming*. JETIR.

[12] **Mandiga, A., et al.** (2026). *NutriCHAT: A Reasoning-Driven large language model agent with Expert-Designed tools for Knowledge-Grounded poultry nutrition Assistance*.

[13] **Seal, S. B., et al.** (2026). *AgriIR: A Scalable Framework for Domain-Specific Knowledge Retrieval*. arXiv.

[14] **Singh, R. K., et al.** (2025). *An agentic Multi-Agent Platform for Precision Agriculture*. GNIT.

### 16. Appendix: System Scenarios

**Scenario 1: The Vision Handoff (Field-Robustness in Action)**

* **Input:** Farmer uploads a poorly lit photo of a wheat leaf with spots and types, "What medicine for this?"

* **Vision Pass:** The CNN identifies Brown Spot (Conf: 88%) despite the background noise.

* **Agent Pass:** The LLM receives the text query + the CNN label. It queries the Vector DB for "Brown Spot Wheat Treatment UP", retrieves the exact Mancozeb dosage, and generates a fully grounded response.

**Scenario 2: Agentic Tool Calling (Yield Prediction)**

* **Input:** "I am planting wheat on 2 hectares in Varanasi. How much will I harvest?"

* **Reasoning:** The LLM identifies the intent as yield prediction (if tool is implemented). It checks its required tool parameters (crop, area, district). All are present.

* **Action:** It pauses text generation, executes the `predict_yield("wheat", 2.0, "Varanasi")` Python function, receives the numerical output (e.g., 65 quintals), and seamlessly integrates it into the final conversational response.

**Scenario 3: Context Elicitation (Missing Information)**

* **Input:** "Will my yield be good this year?"

* **Reasoning:** The LLM identifies the yield prediction intent but notes missing parameters (crop, area, district).

* **Action:** Instead of hallucinating, it responds: "I can help estimate your yield! Could you please tell me which district you are in, whether you are planting rice or wheat, and your farm size?"

---

## 17. Team Review & Sign-Off

All team members listed below have **reviewed and approved** this Milestone 1 document and confirm it accurately reflects the team's collective work and agreed direction.

| # | Team Member | Role | Reviewed & Approved | Date | Signature |
|:-:|-------------|------|:-------------------:|:----:|-----------|
| 1 | Mahesh | Architecture, retrieval experiments, final report | ☑ | 6 Jul 2026 | Mahesh |
| 2 | Harliv | Problem framing, slides, documentation review | ☑ | 6 Jul 2026 | Harliv |
| 3 | Lokesh | Report authoring, literature review | ☑ | 6 Jul 2026 | Lokesh |
| 4 | Aneeqa | Data/API inventory, threshold methodology | ☑ | 6 Jul 2026 | Aneeqa |
| 5 | Tanmay | Architecture & infrastructure research | ☑ | 6 Jul 2026 | Tanmay |

*By signing above, each member confirms they have read the complete document and approve it for submission.*

**Document version:** Milestone 1 — updated · **Prepared:** July 2026
