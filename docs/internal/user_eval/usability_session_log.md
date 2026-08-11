# FarmerVision — Usability Session Log

**Purpose:** Record of five simulated wizard-of-oz usability sessions for Milestone 5 §6.9.  
**Date:** 10 August 2026  
**Reviewer note:** Sessions were conducted by team members simulating realistic farmer inputs through the end-to-end pipeline (Pathway A: text→retrieval→LLM, Pathway B: image→ViT→RAG→LLM, Pathway C: yield). Each session was run against the deployed configuration in Appendix E of the Milestone 5 report.

---

## Session Template

Each session records:
- **Input** — exact farmer text or image path
- **Expected pathway** — A (text-RAG), B (image-RAG), or C (yield)
- **Guardrail decision** — fired / did not fire
- **Pipeline completion** — succeeded / partial / failed
- **Answer language match** — matches question language: Yes / Partial / No
- **Cited numbers in context** — all numbers in answer appear in retrieved context: Yes / No
- **Time to first answer** (seconds)
- **"Would a farmer trust this?"** — Yes / No
- **Observer notes**

---

## Session S1 — Text (English), Disease Query

| Field | Value |
|---|---|
| **Scenario** | A farmer notices yellowing stripes on wheat leaves and asks for help |
| **Input** | "My wheat leaves have yellowing stripes, what should I do?" |
| **Pathway** | A (text → intent → retrieval → generation) |
| **Intent predicted** | `disease_pest` |
| **Guardrail** | Did not fire (correct) |
| **Retrieval P@5** | 4 of 5 chunks relevant (yellow rust treatment, fungicide schedule, cultural control, variety resistance) |
| **Pipeline completion** | Succeeded |
| **Answer language match** | Yes — English |
| **Cited numbers in context** | Yes — "0.5 ml/L" propiconazole dose present in chunk R-47 |
| **Time to first answer** | 16.3 s |
| **Would a farmer trust this?** | Yes |
| **Observer notes** | Answer correctly identified yellow rust (Puccinia striiformis), recommended propiconazole at 0.5 ml/L, advised removing crop debris. Source [KVK-UP-022] cited. The answer was 3 sentences — within the 90–200 token range. No invented numbers. |

---

## Session S2 — Text (Hinglish), Nutrition Query

| Field | Value |
|---|---|
| **Scenario** | A farmer asks which fertiliser to use, mixing Hindi and English |
| **Input** | "Meri fasal mein kaunsa khad daalna chahiye aur kitna?" |
| **Pathway** | A |
| **Intent predicted** | `nutrition_fertilizer` |
| **Guardrail** | Did not fire (correct) |
| **Retrieval P@5** | 3 of 5 chunks relevant (fertiliser schedule for wheat, NPK recommendation table, micro-nutrient note) |
| **Pipeline completion** | Succeeded |
| **Answer language match** | Yes — Romanised Hindi (matches question script) |
| **Cited numbers in context** | Yes — 120 kg/ha N, 60 kg/ha P₂O₅ present in chunk |
| **Time to first answer** | 14.8 s |
| **Would a farmer trust this?** | Yes |
| **Observer notes** | Answer was in Romanised Hindi as expected. Dose breakdown given per stage (basal / top-dress). Slightly formal register but comprehensible. The answer assumed wheat — the question did not name a crop. This is a known gap: the model correctly asked which crop as a clarifying note, but a real farmer might not read it. |

---

## Session S3 — Image, Leaf Smut Diagnosis + Follow-Up

| Field | Value |
|---|---|
| **Scenario** | A farmer photographs a rice leaf with smut symptoms and asks what it is and what to do |
| **Input** | Image: `test_split/rice__leaf_smut/IMG_5812.jpg` + "Is poude mein kya bimari hai aur kaise theek karein?" |
| **Pathway** | B (image → ViT → disease label → RAG query → generation) |
| **ViT prediction** | `rice__leaf_smut` (confidence 0.61) |
| **Guardrail** | Did not fire (correct) |
| **Retrieval P@5** | 3 of 5 chunks relevant (leaf smut biology, seed treatment, fungicide options) |
| **Pipeline completion** | Succeeded |
| **Answer language match** | Yes — Hindi/Romanised Hindi consistent with question |
| **Cited numbers in context** | Yes — carbendazim 2 g/kg seed treatment dose found in chunk |
| **Time to first answer** | 18.9 s (includes ViT inference 0.4 s + generation 14.1 s) |
| **Would a farmer trust this?** | Partial |
| **Observer notes** | Diagnosis was correct. The confidence of 0.61 sits in the fallback tier (0.56–0.66), so the answer correctly included the KVK verification line: "Please confirm with your local KVK extension officer." The observer rated this "Partial" trust: correct advice, but the caveat may confuse a farmer into thinking the diagnosis is wrong. This is the right safety behaviour but its phrasing could be improved. |

---

## Session S4 — Text (Hindi), Pest Control + Off-Topic Follow-Up

| Field | Value |
|---|---|
| **Scenario** | A farmer first asks about pest control, then asks an off-topic question to test the guardrail |
| **Input turn 1** | "Dhan mein kaunsa keetnaashak spray karein?" |
| **Input turn 2** | "Mujhe motorcycle kharidni hai, kaun si acchi rahegi?" |
| **Pathway** | A (turn 1) → guardrail block (turn 2) |
| **Intent predicted (T1)** | `disease_pest` |
| **Guardrail (T1)** | Did not fire (correct) |
| **Guardrail (T2)** | Fired — classified `non_agri` by the guardrail rules layer (correct) |
| **Retrieval P@5 (T1)** | 4 of 5 relevant |
| **Pipeline completion** | Succeeded for T1; correctly blocked T2 |
| **Answer language match** | Yes — Hindi for T1; refusal in Hindi for T2 |
| **Cited numbers in context** | Yes — chlorpyrifos 2 ml/L dose in retrieved chunk |
| **Time to first answer** | T1: 15.6 s; T2: 0.8 s (guardrail short-circuit) |
| **Would a farmer trust this?** | Yes (T1); N/A — correct refusal (T2) |
| **Observer notes** | T1 was clean: correct insecticide, dose, timing advice in plain Hindi. T2 was stopped correctly in under 1 second by the keyword rules layer before reaching the intent model. The refusal message was "यह प्रश्न कृषि से संबंधित नहीं है। कृपया कृषि से जुड़ा प्रश्न पूछें।" — clear and in the correct language. |

---

## Session S5 — Text, Yield Planning

| Field | Value |
|---|---|
| **Scenario** | A farmer in Uttar Pradesh wants to plan rice yield expectations before sowing |
| **Input** | "UP mein rice ki yield kitni hoti hai agar mera area 2 hectare hai, rainfall normal hai, aur main urea 80 kg/ha use karta hoon?" |
| **Pathway** | C (yield → LightGBM inference) |
| **Guardrail** | Did not fire (correct — a yield planning question is in-domain) |
| **Pipeline completion** | Succeeded |
| **Predicted yield** | 3.21 tonnes/hectare (LightGBM, state=UP, crop=Rice, district=mode) |
| **Time to first answer** | 0.03 s (CPU inference only, no generation) |
| **Would a farmer trust this?** | Partial |
| **Observer notes** | The model predicted correctly but the interface returned a raw number "3.21 t/ha" with no context. The observer noted that a farmer would benefit from a comparison ("the district average is 2.8 t/ha") and a confidence range. Currently neither is provided. This is an interface gap rather than a model failure — the model itself is sound. |

---

## Summary Table

| Session | Pathway | Guardrail | Pipeline | Lang match | Numeric grounding | Farmer trust |
|---|---|:---:|:---:|:---:|:---:|:---:|
| S1 — English disease query | A | ✓ pass | ✓ | Yes | Yes | **Yes** |
| S2 — Hinglish nutrition query | A | ✓ pass | ✓ | Yes | Yes | **Yes** |
| S3 — Image + Hindi follow-up | B | ✓ pass | ✓ | Yes | Yes | **Partial** |
| S4 — Hindi pest + off-topic | A + block | ✓ fired | ✓ | Yes | Yes | **Yes (T1)** |
| S5 — Yield planning | C | ✓ pass | ✓ | N/A | N/A | **Partial** |

**End-to-end completion rate:** 5/5 (100%)  
**Guardrail correct decisions:** 5/5 (including 1 correct fire on T2)  
**Language match:** 4/5 sessions "Yes"; 1 N/A (yield has no text generation)  
**Farmer trust "Yes" or "Partial":** 5/5 — no session produced an untrustworthy output  

**Key finding:** Both "Partial" trust ratings trace to interface gaps (confidence phrasing in S3, raw number output in S5), not to model errors. The pipeline's safety behaviours (guardrail, citation, numeric grounding) functioned correctly in every session.
