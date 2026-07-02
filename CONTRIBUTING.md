# Milestone 1 — Team Contribution Log

## Working Model

The team worked in a **fully collaborative, non-siloed manner** for Milestone 1. No member was locked to a fixed sub-task; instead, everyone was free to research any aspect, contribute to any component, and review each other's work. Research documentation was shared openly across the team so that findings compounded rather than duplicated. The entries below record each member's primary areas of contribution within that shared effort.

---

## Mahesh — Contributions
- Helped drive the team from a broad "farmer assistant" concept toward a **sharp, research-worthy problem statement** — reframing a generic agri-chatbot into a system defined by three *measurable* trust failures: field-robustness of disease detection, faithfulness/hallucination of advice, and confidence calibration/abstention.
- Contributed to defining the **project scope and boundaries**.

- **Designed and built the system architecture diagram on Excalidraw** (the updated multimodal routing pipeline), finalizing it by incorporating suggestions and inputs from the team.
- Shaped the core design decisions through detailed analysis and probing questions, including:
  - Why intent understanding can be LLM-native vs. a separately trained classifier, and where a fine-tuned model is genuinely justified.
  - The **VLM-aware routing** for image inputs (route leaves to the specialized detector; let the VLM handle non-leaf images) rather than forcing every image into a disease class.
  - The **context-elicitation / follow-up question** flow for personalizing responses (location, crop stage, farming method).
  - The **tiered response strategy** (grounded RAG → transparent LLM fallback → out-of-scope redirect) and how RAG retrieval actually works mechanically (embedding → similarity search → grounded generation).
  - Identifying **legitimate, non-forced trainable components** — settling on the field-robust disease classifier and the domain-adapted embedding model as the two core trainable models.

- Explored and compiled the **candidate datasets**.

- Reviewed current solutions, tools, and academic work relevant to the problem.
- Assembled key references and industry context (WEF deep-tech reports, FarmerChat/GAIA, government initiatives, Syngenta, Plantix, Intello Labs, Wadhwani AI) into the research documentation.

- Authored the primary research document (*Farm_assistant_research_doc_Mahesh*), capturing the problem framing, interaction scenarios, design doubts and their resolutions, dataset notes, and literature review — and **shared it with the team** as a common reference.

### 6. Final Report Preparation
- **Prepared the final Milestone-1 report**, consolidating inputs and research from all team members into a single structured deliverable covering the problem statement, scope, stakeholders, objectives, architecture, scenarios, literature review, comparative analysis, datasets, and evaluation framework.

---

## Harliv — Contributions
- Contributed to defining and refining the initial problem statement and project direction.

- Assisted in preparing and designing the milestone presentation slides.

- Reviewed the project documentation for technical accuracy, clarity, and completeness.

- Identified architectural inconsistencies and potential design issues within the proposed system.

- Suggested improvements and solutions to strengthen the system architecture and overall project design.

- Participated in discussions to refine the problem statement, objectives, and technical approach.
---

## Lokesh — Contributions

---

## Aneeqa — Contributions
- Participated in team discussions to narrow the broad "farmer assistant" concept into a focused problem statement.

- Contributed to defining project boundaries, identifying what is in scope versus explicitly out of scope.

- Researched existing solutions including DigiGreen, KisanSarathi.

- Explored government and industry initiatives including India's Digital Agriculture Mission, Kisan e-Mitra and Bharat-VISTAAR.

- Reviewed and contributed to research documentation and final Milestone-1 report, ensuring technical accuracy and completeness
---

## Tanmay — Contributions

---

## Shared / Collective Activities
- Joint discussion and refinement of the problem statement and scope.
- Cross-review of each other's research documents.
- Collective narrowing of the idea from a broad concept to a focused, measurable, milestone-aligned project.
- Shared identification of gaps and opportunities across current solutions.

---
