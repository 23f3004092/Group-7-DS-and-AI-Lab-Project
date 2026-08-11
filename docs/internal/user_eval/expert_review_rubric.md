# FarmerVision — Expert Answer Review Rubric

**Purpose:** Structured scoring of generated answers for the Milestone 5 user-centred evaluation (§6.9).  
**Sample:** 20 answers drawn from the 77 real farmer questions, stratified by language (English / Hindi / Hinglish) and intent class.  
**Reviewers:** Two per answer — one acting as *domain expert* (agronomically trained; Mahesh / Tanmay), one as *naïve farmer* (no agri background; Aneeqa / Harliv / Lokesh).

---

## Scoring Dimensions

Each answer is scored independently by both reviewers on five dimensions. Use integers 1–5. Do **not** look at the context chunks or model internals — score only what the farmer would see.

### D1 — Factual Accuracy
*Is the agronomic advice correct? Would following it harm or help the crop?*

| Score | Anchor |
|:---:|---|
| 5 | Every dose, timing and product mentioned is correct and consistent with ICAR/KVK guidance |
| 4 | Advice is essentially correct; one minor inaccuracy that would not harm the crop |
| 3 | Mixed — the main recommendation is fine but one supporting detail is wrong or outdated |
| 2 | A specific claim (dose, product, timing) is wrong in a way that could damage the crop |
| 1 | The advice is outright wrong or dangerous for the stated crop and condition |

### D2 — Completeness
*Does the answer give the farmer enough to act? Is critical information missing?*

| Score | Anchor |
|:---:|---|
| 5 | Covers what to do, what to use, how much, and when — nothing important omitted |
| 4 | Covers the key action; one supporting detail (e.g. safety interval) is missing |
| 3 | Main recommendation present but missing at least one critical piece (dose or timing) |
| 2 | Partial — gives general advice but nothing specific enough to act on |
| 1 | Does not address the question at all, or is a pure refusal when retrieval was available |

### D3 — Clarity
*Would a farmer with low literacy in the answer's language understand this?*

| Score | Anchor |
|:---:|---|
| 5 | Plain language, short sentences, no unexplained jargon; a farmer can read it in one pass |
| 4 | Mostly clear; one term is jargon or one sentence is overlong |
| 3 | Understandable with effort; technical terms used without explanation |
| 2 | Hard to follow; answer is dense, long, or uses unexplained abbreviations throughout |
| 1 | Incomprehensible or internally contradictory |

### D4 — Actionability
*Can the farmer do something concrete with this answer today?*

| Score | Anchor |
|:---:|---|
| 5 | Specifies product + dose + timing + application method — fully actionable |
| 4 | Specifies product and dose or timing, one dimension missing but still actionable |
| 3 | Names a product or practice but dose/timing must be inferred or looked up |
| 2 | Gives only a category of intervention (e.g. "use a fungicide") without specifics |
| 1 | No concrete action; only describes the problem or says to consult someone |

### D5 — Language Appropriateness
*Is the answer in the right language and register for the question asked?*

| Score | Anchor |
|:---:|---|
| 5 | Matches the question's language exactly (script and register); tone is respectful and farmer-appropriate |
| 4 | Correct language, minor register mismatch (e.g. slightly formal for a casual Hinglish question) |
| 3 | Correct script but wrong register (formal where informal expected, or vice versa) |
| 2 | Partial language match (code-switches in a way that impedes reading) |
| 1 | Wrong language or wrong script entirely for the question |

---

## Binary Overall Rating

After scoring the five dimensions, record one binary judgement:

> **"Would you act on this answer if you were the farmer asking this question?"**
>
> - `YES` — the answer gives enough correct, clear, actionable information to act
> - `NO` — at least one critical flaw (wrong advice, wrong language, missing dose) makes it untrustworthy

---

## Reviewer Instructions

1. Read the question as stated (with any typos or code-switching left intact).
2. Read the answer. Do **not** view the retrieved context chunks.
3. Score D1–D5 independently.
4. Record your binary `act_on` judgement.
5. Write a one-sentence note under `comment` if any score is 1 or 2.
6. If you genuinely cannot judge D1 (you have no agronomic knowledge), write `N/A` and let the domain-expert reviewer's score stand alone for that dimension.

---

## Inter-Rater Agreement

After both reviewers complete scoring, compute Cohen's kappa on the `act_on` binary rating and Krippendorff's alpha on each of D1–D5 (ordinal). A kappa below 0.60 on `act_on` requires a third-reviewer tiebreak for those questions.

---

## Sample entry (filled)

| Field | Value |
|---|---|
| question_id | R041 |
| question | "Mango ki variety ki information chahiye?" |
| language | Hinglish |
| intent | cultivation_practice |
| model | distilled |
| D1_expert | 4 |
| D1_farmer | 4 |
| D2_expert | 3 |
| D2_farmer | 3 |
| D3_expert | 5 |
| D3_farmer | 4 |
| D4_expert | 3 |
| D4_farmer | 2 |
| D5_expert | 5 |
| D5_farmer | 5 |
| act_on_expert | YES |
| act_on_farmer | YES |
| comment_expert | Variety list given, no regional suitability guidance |
| comment_farmer | — |
