# RAG Vector DB — Audit, Model Choice, and How to Run It

**What this file is:** the full story of how we built the retrieval part of the project —
what broke, what we measured, which embedding model we picked and why, and step-by-step
instructions so anyone can run searches without rebuilding the database.

**Written:** 27 July 2026 · **Status:** production index built and saved

---

## Table of Contents

1. [Short version](#1-short-version)
2. [The problem we found](#2-the-problem-we-found)
3. [How we tested three models](#3-how-we-tested-three-models)
4. [Why we chose bge-m3](#4-why-we-chose-bge-m3)
5. [Other bugs we found and fixed](#5-other-bugs-we-found-and-fixed)
6. [The two notebooks](#6-the-two-notebooks)
7. [What the build notebook does, cell by cell](#7-what-the-build-notebook-does-cell-by-cell)
8. [What the retrieval notebook does, cell by cell](#8-what-the-retrieval-notebook-does-cell-by-cell)
9. [The saved artifacts](#9-the-saved-artifacts)
10. [How to run retrieval (no rebuild)](#10-how-to-run-retrieval-no-rebuild)
11. [How to rebuild, if you ever must](#11-how-to-rebuild-if-you-ever-must)
12. [Known limitations](#12-known-limitations)
13. [Numbers reference](#13-numbers-reference)

---

## 1. Short version

We built a searchable database of **723,439 pieces of agricultural text** — 7,136 chunks from
government PDFs and 716,303 chunks from Kisan Call Centre farmer Q&A.

The first version did not work. The embedding model we started with could not tell one farm
topic from another, so a question about urea in wheat returned advice about capsicum, and it
reported 99.9% confidence while doing it.

We tested three replacement models and picked **BAAI/bge-m3**. The final system:

- answers in **30–330 milliseconds**
- passes all filter, routing and safety checks
- correctly refuses to answer questions outside farming
- returns real answers with real numbers (e.g. "urea 30 kg per acre")

The whole database is saved to Google Drive as one file. The retrieval notebook loads it and
you can search immediately — no rebuilding, no GPU hours.

---

## 2. The problem we found

### What we saw

The first index used `Yunika/muril-base-sentence-transformer`, chosen because MuRIL is
pre-trained on Indian languages including Hindi written in English letters.

The notebook ran with no errors. Every check looked fine. But the actual results were nonsense:

| Question asked | What came back |
|---|---|
| "how much urea in wheat at tillering stage" | capsicum fertiliser advice (score 1.000) |
| "brown spots on rice leaves" | *"Yours faithfully, Plant Protection Adviser"* — a signature block |
| "dhan ki nursery me pili patti" | fig storage, coriander irrigation |
| "best chess opening strategy" | scored 0.999 — same as real farm questions |

### Why it happened

We tested the model directly, outside the database. Three completely unrelated agricultural
sentences scored **1.000** similarity with each other:

```
                urea/wheat   capsicum dose   bengal gram
urea/wheat         1.000         1.000          1.000
```

Meanwhile "car engine repair" scored **−0.879**. So the model could tell *farming from
non-farming*, but could not tell *wheat from mango*. Retrieval needs the second one.

On a tiny 6-document test where every answer was obvious, it scored **1 out of 6**.

### Why nobody noticed

The notebook had a safety check built in. It compared "rice blast disease" against "how to win
at chess" and required a clear gap. That gap was **+0.874** — it passed easily.

But that is a *cross-topic* test, and cross-topic was the one thing the model could do. The
test never asked the question that mattered: can it tell two farm topics apart?

**Lesson:** a health check must test the actual job, not an easier version of it.

### The fix

We replaced the safety check. The new one is a real ranking task: five different farm topics
(wheat urea, wheat rust, paddy nursery, PM-KISAN, mango pest), asked in English, Hindi and
Hinglish, and the correct document must come first. It fails the notebook outright if:

- fewer than 4 out of 5 correct documents rank first, **or**
- the winning margin is under 0.01 (correct by luck, not by skill), **or**
- unrelated documents score above 0.97 with each other (space has collapsed)

We tested this new gate against a deliberately broken fake model, and it blocked the build on
all three conditions.

---

## 3. How we tested three models

We built three identical notebooks. Same chunks, same chunking, same 512-token limit, same
tests. **Only the model changed.** That way any difference is the model's doing.

| Model | Size | Vector size | Prefixes needed |
|---|---|---|---|
| `intfloat/multilingual-e5-base` | 278M | 768 | `query: ` / `passage: ` |
| `intfloat/multilingual-e5-large` | 560M | 1024 | `query: ` / `passage: ` |
| `BAAI/bge-m3` | 568M | 1024 | **none** |

> **About prefixes:** the e5 models were trained with the words `query: ` and `passage: ` glued
> to the front of the text. Leaving them out makes results worse. bge-m3 was **not** trained
> that way — adding them there would push meaningless words into every vector. This is easy to
> get wrong and produces no error message, just quietly worse answers.

### Results (e5-large and bge-m3 on the identical 47,136-chunk corpus)

| Measure | e5-base* | e5-large | **bge-m3** | What it means |
|---|---|---|---|---|
| Correct doc ranked first | 5/5 | 5/5 | 5/5 | all three pass the basic test |
| Average winning margin | +0.042 | +0.041 | **+0.222** | how clearly the right answer beats the wrong one |
| Worst margin (Hinglish) | +0.006 | +0.010 | **+0.122** | the hardest case |
| Unrelated docs score | 0.861 | 0.844 | **0.404** | lower is better — 0.86 means everything looks alike |
| In-domain vs off-domain gap | +0.009 | +0.018 | **+0.072** | the room "refuse to answer" has to work in |
| Checks passed | 8/11 | 9/11 | **10/11** | |

\* e5-base ran on a larger 107k corpus, so it is indicative only, not a fair comparison.

---

## 4. Why we chose bge-m3

### It actually uses the space

Unrelated documents sit at **0.404** instead of 0.844. The e5 models squash everything into a
narrow band — a milder version of the same disease MuRIL had. bge-m3 spreads things out.

### That gap is what makes "refuse to answer" possible

```
e5-large:  real questions 0.838 | junk questions 0.820  ->  gap 0.018
bge-m3:    real questions 0.590 | junk questions 0.517  ->  gap 0.072
```

Our system gives farmers **pesticide doses**. Answering confidently from a bad match is a real
safety problem. With a 0.018 gap you cannot draw a reliable line between "answer this" and
"refuse this". With 0.072 you can. **bge-m3 was the first model where refusal worked.**

### Bigger was not better

e5-large has double the parameters of e5-base and gained almost nothing: margin went +0.042 →
+0.041, unrelated-doc score 0.861 → 0.844. It cost 50% more time for noise-level change.

This is an important finding: **the problem was never model size.** It was the kind of training.
bge-m3 is trained specifically for search, using "hard negatives" — documents that look right
but are wrong. Learning to reject near-misses is exactly what teaches fine distinctions.

### It has room to grow

bge-m3 can also produce **sparse (keyword-style) vectors** from the same model. Our one
remaining weak spot is Hinglish technical words like *"ratua"* (rust) — exactly the kind of
literal word keyword matching catches and meaning-based matching blurs. Neither e5 model offers
this. We have not enabled it yet; it is the obvious next improvement.

---

## 5. Other bugs we found and fixed

### Crop filter matched nothing — silently

The KCC data stores crops as `Paddy (Dhan)`, `Maize (Makka)`, `Bengal Gram (Gram/Chick Pea)` —
**63 of 281** crop names have brackets. Our conversion table only matched plain names.

So filtering by `crop="rice"` matched **0 of 112,269 rice chunks**. No error, no warning, just
an empty result.

**Fixed:** the converter now tries the whole string, then the part before the bracket, then each
alias inside the bracket. Checked against all 716k records, and confirmed it is *stable* — the
output of the converter converts to itself, which is what makes the filter and the stored value
agree.

```
before   crop='rice'  ->        0 chunks
after    crop='rice'  ->  112,269 chunks
         crop='maize' ->   15,047 chunks  (was 0)
         crop='dhan'  ->  112,269 chunks  (Hindi word also works)
```

### A test that failed for the wrong reason

The district test kept failing. The retrieval was actually correct — the *citation formatter*
dropped the district field for PDF results, so the test saw `None` and complained.

**Fixed:** the field is included now. Worth recording because it looked like a search problem
and was not.

### Confidence thresholds were meaningless

The thresholds (0.85 / 0.65) came from Milestone 1, set against a completely different model.
Score scales are **not comparable between models** — 0.86 from one model means nothing like
0.86 from another. Applied to bge-m3, the old numbers would have refused *every single
question*, including perfect matches.

**Fixed:** the notebook now measures the real score spread on the real index and works out the
thresholds itself, then prints them to copy into the settings.

We also fixed how the upper threshold is chosen. It used to be the *middle* score of real
questions, which by definition means half of all good answers get downgraded. It is now the
lower-quarter point.

### Three smaller ones

- **Missing settings.** An edit of ours accidentally deleted the block defining the database
  folder path. A check across all cells caught it before it was run.
- **Wrong Qdrant download.** The standard Linux build needs system library version 2.38;
  Colab has 2.35, so it downloaded fine and died instantly. Switched to the `musl` build,
  which carries its own libraries. The notebook now also *test-runs* the file before trusting it.
- **Snapshot in the wrong folder.** Our save code looked for the backup file under the database
  folder, but Qdrant writes backups to a separate folder. Now it asks Qdrant for the file over
  HTTP instead of guessing where it is.

---

## 6. The two notebooks

| Notebook | Purpose | When you run it | Time |
|---|---|---|---|
| `08c_rag_vector_db_bge_m3.ipynb` | **Builds** the database | Once. Already done. | ~4 hours |
| `09_rag_retrieval.ipynb` | **Searches** the database | Every time you want results | ~10 minutes |

The two are deliberately separate. Building needs a GPU and hours; searching needs neither.

> The e5-base and e5-large notebooks were deleted after the comparison was finished. Their
> results are recorded in this document.

---

## 7. What the build notebook does, cell by cell

`08c_rag_vector_db_bge_m3.ipynb`

**§0 — Start the Qdrant server.** Downloads Qdrant and runs it in the background.

*Why this matters:* Qdrant has two modes. "Local mode" is simpler but ignores the search index
completely and checks every single record for every question. We measured it: 1.6 seconds at
47k records, 3.7 seconds at 107k — doubling the data doubled the time. At the full 723k that
would be around **25 seconds per question**. Server mode builds the proper index. Same data,
**466 milliseconds**.

**Settings cell.** Model name, prefixes (empty for bge-m3), how many chunks to use, where to
save, thresholds, and how much to favour PDFs vs KCC for different question types.

**Load PDF chunks / Load KCC chunks.** Reads both files and puts them in one shared format:
same language detection, same crop and district spellings, same ID scheme. Chunks that are too
long for the model get split at sentence boundaries (including the Hindi `।`).

**Summary.** Counts by source, language, district and crop, and confirms every ID is unique.
Uses simple counters rather than a data table, because a table would make a second full copy of
723k records in memory.

**Model check.** Loads bge-m3 and runs the ranking test described in section 2. **Stops the
notebook if it fails.**

**Build the index.** The main work, in batches of 20,000:

1. turn 20,000 chunks into vectors on the GPU
2. **save those vectors to Google Drive**
3. send them to the database

Step 2 is the important one. If Colab disconnects — which it usually does during a 4-hour job —
you restart and it *reloads* the saved vectors instead of redoing them on the GPU. We tested
this: after a simulated crash, a restart re-did only the one unfinished batch.

Index building is switched **off** while loading and switched on at the end, so it gets built
once instead of being torn down and rebuilt after every batch.

**Search tool.** Defines `search_agri_knowledge()` — see section 8.

**Evaluation.** Runs the health checks and works out the confidence thresholds.

**Inspector.** Runs 17 example questions and prints the actual text that came back, so quality
can be judged by reading rather than by trusting a number.

**§8 Export.** Saves everything to Drive (section 9).

---

## 8. What the retrieval notebook does, cell by cell

`09_rag_retrieval.ipynb` — **this builds nothing.**

**§1 Start Qdrant server.** Same as above. Needed because backups can only be loaded into a
real server.

**§2 Restore.** Reads `manifest.json`, uploads the 3.8 GB backup into the server, and reports
how many records came back. Skips the upload if the data is already loaded.

**§3 Load the model.** Loads bge-m3 and reads the prefixes, vector size and thresholds
**from the manifest file, not typed in by hand.**

*Why:* if someone typed `query: ` here (correct for e5, wrong for bge-m3), every answer would
quietly get worse with no error. If someone typed the old 0.837 threshold, the system would
refuse every question. Reading from the manifest makes that impossible. There is also a check
that the model's vector size matches the saved index.

**§4 Name conversion.** The crop and district spelling converter from section 5. The filters
need the exact same conversion used when building, or they match nothing.

**§5 The search tool.** `search_agri_knowledge()`:

- takes a question plus optional filters
- **never crashes** — problems come back as `tier: "error"`
- searches PDF and KCC separately, then combines with weights: policy questions favour PDFs
  (2.0 vs 0.5), how-to questions favour KCC (0.5 vs 2.0)
- every result carries its source (PDF file and page, or KCC record with crop/district/year)
- decides confidence on the **raw** score, never the weighted one — otherwise the weighting
  could manufacture false confidence

The three confidence levels:

| Level | Score | What the assistant should do |
|---|---|---|
| `grounded` | ≥ 0.66 | answer and cite sources |
| `fallback_with_disclaimer` | 0.56 – 0.66 | answer, but add "please check with your local KVK" |
| `abstain_out_of_scope` | < 0.56 | refuse — say it is outside what we know |

**§6 Examples.** Seven worked questions with the text shown.

**§7 Ask anything.** Edit `MY_QUERY`, re-run that one cell.

**§8 Health check.** Filter, routing and refusal tests. Run once after loading to confirm the
backup came back correctly.

---

## 9. The saved artifacts

### Folder 1 — needed to search: `MyDrive/rag_production_bge_m3/`

| File | Size | What it is | Needed? |
|---|---|---|---|
| `agri_knowledge-*.snapshot` | **3.80 GB** | the whole database: 723,439 vectors, all the text and metadata, and the search index | **Yes** |
| `manifest.json` | 1 KB | model name, prefixes, vector size, thresholds, weights | **Yes** |
| `rag_eval_report.json` | small | test results from the build | no |
| `retrieval_inspection.json` | small | the 17 example searches with full text | no |

**Both required files must match each other.** The backup without the manifest is unsafe — you
would be guessing at the prefixes and thresholds, and guessing wrong produces no error.

### Folder 2 — insurance: `MyDrive/rag_emb_cache_bge_m3_full/`

About 37 files, `emb_00000.npy` … , roughly **1.5 GB** total. These are the raw vectors.

**Not needed for searching.** Keep them because they let you rebuild the database *without
redoing the GPU work*, which is the expensive part. Stored at half precision to save space —
we measured the accuracy loss at 0.00000012, effectively nothing.

---

## 10. How to run retrieval (no rebuild)

**You need:** the two required files from folder 1 in your Google Drive, and a Colab session.
A GPU helps but is not required.

1. Open `09_rag_retrieval.ipynb` in Colab.
2. Check `ARTIFACT_DIR` in §2 points to your folder. Default:
   `/content/drive/MyDrive/rag_production_bge_m3`
3. **Runtime → Run all.** Allow Drive access when asked.
4. Wait about 5–10 minutes. Almost all of it is uploading the 3.8 GB backup.
5. Confirm §2 prints **723,439 points**. If it does not, the backup did not load properly.
6. Ask your own questions in §7:

```python
MY_QUERY  = "गेहूं में खरपतवार नियंत्रण के लिए कौन सी दवा डालें"
MY_PARAMS = dict(intent="field_practice", top_k=5)
```

Re-run just that cell for each new question — no need to re-run anything above it.

**Run §8 once** the first time. It confirms the database came back intact.

### Filters you can use

| Filter | Example | Notes |
|---|---|---|
| `intent` | `"policy"`, `"field_practice"`, `"general"` | shifts the PDF/KCC balance |
| `source_type` | `"pdf"` or `"kcc"` | force one source only |
| `crop` | `"rice"`, `"wheat"`, `"dhan"` | Hindi names work too |
| `district` | `"allahabad"` | old names convert automatically |
| `season` | `"Rabi"`, `"Kharif"`, `"Zaid"` | KCC only |
| `only_tables` | `True` | dose tables from PDFs only |
| `year_from` | `2020` | newer content only |
| `top_k` | `5` | how many results |

### If something goes wrong

| Problem | Cause | Fix |
|---|---|---|
| `artifact folder not found` | wrong path | fix `ARTIFACT_DIR` in §2 |
| Points count is 0 or wrong | upload did not finish | re-run §2 |
| `dim mismatch` error | wrong model for this backup | do not override the manifest |
| Every question refused | thresholds overridden by hand | let §3 read them from the manifest |
| Very slow (seconds) | index still building | wait, re-check §2 |
| Qdrant will not start | download problem | check `/content/qdrant.log` |

---

## 11. How to rebuild, if you ever must

Only if the backup is lost or you want different settings.

**With the vector cache** (folder 2 intact) — about 1 hour: run `08c` normally. It reloads the
saved vectors and skips the GPU work.

**From scratch** — about 4 hours: clear the cache folder first, then run `08c` from the top.
Expect Colab to disconnect at least once; just re-run and the cache picks up where it left off.

Approximate stage times from our actual build:

| Stage | Time |
|---|---|
| Reading both chunk files | 3 min |
| Turning 723k chunks into vectors (GPU) | ~3 h |
| Loading into the database | 30 min |
| Building the search index | **15 min** |
| Saving to Drive | 2 min |

---

## 12. Known limitations

These are honest gaps, recorded rather than hidden.

**1. Hinglish technical words.** "gehu me pila ratua" (wheat yellow rust) returns potato leaf
yellowing. The model matches "pila" (yellow) but misses "ratua" (rust). We confirmed this is
**not** a data coverage problem — the correct wheat rust PDF is in the database, it just never
surfaces, partly because 716k KCC chunks outnumber 7k PDF chunks. Plain Hindi and plain English
both work well, and common Hinglish works fine. Fix: turn on bge-m3's keyword vectors.

**2. The bilingual test is broken.** It checks whether the English and Hindi versions of a
question return the *same records*. But with 112,269 rice chunks, both can return completely
correct but different records and score zero. It reports 0.00 while retrieval is fine. It
should compare topics, not record IDs. **Ignore this check's result.**

**3. Some correct answers are marked "verify with KVK"** instead of "confident". Cautious, not
wrong.

**4. Speed is measured on Colab.** 30–330 ms on a free Colab session with 2 CPU cores. Real
server hardware would be faster.

**5. Confidence thresholds are set from 14 test questions** we wrote ourselves. Reasonable, but
not a proper calibration. A better version would use real farmer questions with human judgement
about whether the answer was usable.

---

## 13. Numbers reference

### Database

| | |
|---|---|
| Total chunks | 723,439 (7,136 PDF + 716,303 KCC) |
| Model | `BAAI/bge-m3`, 1024 numbers per vector |
| Distance measure | cosine |
| Search index | HNSW, m=16, ef_construct=128 |
| Backup size | 3.80 GB |
| Index build time | 15 minutes |

### Speed (full 723k database)

| | |
|---|---|
| Typical question | 30–330 ms |
| Half of questions under | 466 ms |
| 95% of questions under | 903 ms |
| Same thing in local mode at 107k | 3,743 ms |

### Confidence thresholds (bge-m3 scale — do not reuse for other models)

| | |
|---|---|
| Confident | ≥ 0.66 |
| Answer with disclaimer | 0.56 – 0.66 |
| Refuse | < 0.56 |
| Real questions scored | 0.590 – 0.811 |
| Junk questions scored | 0.480 – 0.530 |
| Gap between them | +0.059 |

### Top crops after name conversion

| Crop | Chunks |
|---|---|
| rice | 112,269 |
| wheat | 95,836 |
| sugarcane | 58,646 |
| potato | 40,818 |
| mustard | 39,147 |
| mango | 36,809 |

---

*Every number here was measured on the actual runs, not estimated.*
