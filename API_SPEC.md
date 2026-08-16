# FarmerVision — Vision API (Leaf-Disease) Spec

Everything for the **image** side of the API: classify a leaf photo, or classify **and** get grounded
treatment advice. Two endpoints: `POST /vision` and `POST /diagnose`.

> Base URL and API key are provided separately. Below they are `<BASE_URL>` and `<API_KEY>`.

---

## 1. Connection

| | |
|---|---|
| **Base URL** | `<BASE_URL>` (e.g. `http://HOST:8000`) |
| **Auth** | header `X-API-Key: <API_KEY>` on every request |
| **Content type** | `multipart/form-data` (these two endpoints upload a file — **not** JSON) |
| **Image field name** | `file` |
| **Accepted formats** | JPG / PNG (anything PIL can open; RGB) |
| **Model** | ViT-S/16 (timm `vit_small_patch16_224`), 20 classes, runs on **CPU** (the GPU is reserved for the text LLM) |
| **Speed** | `/vision` ≈ **fast** (image only, no LLM). `/diagnose` ≈ **~10–13 s** (adds retrieval + LLM) |

The model is **lab-trained** — every response carries a "treat as a suggestion, confirm with a local
expert" note. Surface that in the UI. There is **no hard reject**: every image gets a label +
confidence, so use `confidence` (and `top_k`) to decide how much to trust it.

---

## 2. `POST /vision` — classify only

Returns the predicted crop + disease and confidence. No knowledge-base lookup, no LLM. Use this when
you only need the label (e.g. to show a diagnosis card, or to drive your own follow-up).

### Request
`multipart/form-data` with a single field `file` = the image.

```bash
curl -s -X POST <BASE_URL>/vision \
  -H "X-API-Key: <API_KEY>" \
  -F "file=@leaf.jpg"
```

### Response `200`
```json
{
  "label": "wheat__yellow_rust",
  "crop": "wheat",
  "disease": "yellow rust",
  "confidence": 0.94,
  "top_k": [
    { "label": "wheat__yellow_rust", "prob": 0.94 },
    { "label": "wheat__brown_rust",  "prob": 0.03 },
    { "label": "wheat__septoria",    "prob": 0.01 },
    { "label": "wheat__healthy",     "prob": 0.01 },
    { "label": "wheat__leaf_blight", "prob": 0.005 }
  ],
  "note": "Lab-trained model; treat as a suggestion, confirm with a local expert."
}
```

| field | type | meaning |
|---|---|---|
| `label` | string | full class, format `crop__disease` (see §4) |
| `crop` | string | crop part of the label (e.g. `wheat`, `rice`) |
| `disease` | string | disease part, spaces instead of underscores (e.g. `yellow rust`, `healthy`) |
| `confidence` | float 0–1 | softmax probability of the top class |
| `top_k` | array | top **5** predictions, each `{label, prob}`, highest first — use to show alternatives / detect uncertainty |
| `note` | string | fixed safety disclaimer — always show it |

**Reading confidence:** high (`≥ ~0.8`) → show the label plainly. Middle (`~0.5–0.8`) → show it but
also surface `top_k[1]` as "could also be…". Low (`< ~0.5`) or top two probs close together → tell the
user the photo is unclear and to retake (better lighting, single leaf, fill the frame).

---

## 3. `POST /diagnose` — classify **+** grounded treatment

Runs vision, then retrieves matching Kisan-Call-Centre / advisory records for that disease and has the
LLM write **cited treatment advice**. Use this for the one-tap "what's wrong and what do I do" flow.

### Request
`multipart/form-data`:

| field | required | meaning |
|---|---|---|
| `file` | ✅ | the leaf image |
| `question` | optional | a text question to steer the advice, e.g. `"is ke liye kya spray karun"`. If omitted, the server auto-builds a treatment query from the detected disease. |

```bash
curl -s -X POST <BASE_URL>/diagnose \
  -H "X-API-Key: <API_KEY>" \
  -F "file=@leaf.jpg" \
  -F "question=is ke liye kya spray karun aur kitni dose"
```

### Response `200` (treatment found)
```json
{
  "diagnosis": {
    "label": "wheat__yellow_rust",
    "crop": "wheat",
    "disease": "yellow rust",
    "confidence": 0.94,
    "top_k": [ { "label": "wheat__yellow_rust", "prob": 0.94 }, "..." ],
    "note": "Lab-trained model; treat as a suggestion, confirm with a local expert."
  },
  "tier": "grounded",
  "answer": "Likely wheat yellow rust. Spray Mancozeb 75 WP at 400 g/acre in 200 L water ... Sources: [1], [2]",
  "sources": [
    { "n": 1, "score": 0.68, "source_type": "pdf",
      "citation": { "corpus": "pdf", "file": "advisory.pdf", "pages": [12, 13],
                    "doc_category": "crop_advisory", "year": 2022 } },
    { "n": 2, "score": 0.66, "source_type": "kcc",
      "citation": { "corpus": "kcc", "record": "KCC Q&A", "crop": "wheat",
                    "district": "jhansi", "season": "Rabi", "query_type": "Plant Protection", "year": 2023 } }
  ],
  "gen_ms": 12000,
  "out_tokens": 88,
  "latency_ms": 12500
}
```

| field | meaning |
|---|---|
| `diagnosis` | the **full `/vision` result** (label, crop, disease, confidence, top_k, note) |
| `tier` | quality of the treatment answer: `grounded` (confident) or `fallback_with_disclaimer` (weaker match; a "verify with KVK" note is appended to `answer`) |
| `answer` | the cited treatment text to show. Ends with a `Sources: [n]` line. |
| `sources` | records the advice is grounded in — show as source chips. Each: `n`, `score` (0–1), `source_type` (`pdf`\|`kcc`), and a `citation` (shape differs per corpus — see §5). |
| `gen_ms` / `out_tokens` / `latency_ms` | generation time, tokens produced, total round-trip ms |

### Response `200` (no treatment found in the knowledge base)
The photo was still classified, but retrieval found nothing usable → `answer` is `null`, `message`
explains it:
```json
{
  "diagnosis": { "label": "rice__tungro", "crop": "rice", "disease": "tungro", "confidence": 0.81, "...": "..." },
  "tier": "abstain_out_of_scope",
  "answer": null,
  "message": "Diagnosis done, but no matching treatment found in the knowledge base.",
  "sources": [],
  "latency_ms": 900
}
```
**Client rule:** if `answer` is non-null → show it. If `answer` is null → show `diagnosis` + the
`message`, and optionally let the user ask a follow-up on `/query`.

---

## 4. Class labels (20)

Format is `crop__disease` (double underscore between crop and disease; single underscores inside a
multi-word disease). `disease` in the response has underscores replaced with spaces.

**Rice (5):** `rice__blast`, `rice__bacterial_blight`, `rice__brown_spot`, `rice__tungro`, `rice__leaf_smut`

**Wheat (15):** `wheat__healthy`, `wheat__yellow_rust`, `wheat__brown_rust`, `wheat__black_rust`,
`wheat__blast`, `wheat__septoria`, `wheat__mildew`, `wheat__aphid`, `wheat__mite`, `wheat__stem_fly`,
`wheat__smut`, `wheat__tan_spot`, `wheat__leaf_blight`, `wheat__common_root_rot`, `wheat__fusarium_head_blight`

> Note `wheat__healthy` is a valid label — a healthy wheat leaf returns that, not an error.

---

## 5. `citation` object (inside `sources`, `/diagnose` only)

Two shapes depending on the source corpus:
- **pdf** → `{ "corpus":"pdf", "file":..., "pages":[start,end], "section":..., "doc_category":..., "district":..., "year":... }`
- **kcc** → `{ "corpus":"kcc", "record":"KCC Q&A", "crop":..., "district":..., "season":..., "query_type":..., "year":... }`

---

## 6. Errors

| HTTP | body | cause |
|---|---|---|
| 400 | `{"detail":"empty image"}` | no file, or an empty/0-byte upload |
| 401 | `{"detail":"bad or missing X-API-Key"}` | wrong/absent key |
| 501 | `{"detail":"vision model not deployed"}` | this instance was started without the vision model (check `GET /health` → `modules.vision`) |
| 500 | `{"detail":"..."}` | server error (e.g. a corrupt/undecodable image) — retry with a valid JPG/PNG |

Check availability first if unsure: `GET /health` returns `"modules": { ... "vision": true }` when the
image endpoints are live (no key needed for `/health`).

---

## 7. Image guidance (pass this to users)

- One **leaf**, filling most of the frame, in focus, even daylight.
- Avoid heavy shadows, multiple overlapping leaves, or a tiny leaf in a big scene.
- JPG or PNG; a few hundred KB to a few MB is plenty (it's resized to 256→224 px internally).
- The model only knows **rice and wheat** leaf diseases (§4) — other crops will still return a label,
  but it won't be meaningful (use `confidence`/`top_k` to catch this).

---

## 8. Client code

**Python**
```python
import requests
BASE = "<BASE_URL>"; KEY = "<API_KEY>"

# classify only
with open("leaf.jpg", "rb") as f:
    v = requests.post(f"{BASE}/vision", headers={"X-API-Key": KEY},
                      files={"file": f}, timeout=60).json()
print(v["crop"], v["disease"], v["confidence"])

# classify + treatment
with open("leaf.jpg", "rb") as f:
    d = requests.post(f"{BASE}/diagnose", headers={"X-API-Key": KEY},
                      files={"file": f},
                      data={"question": "is ke liye kya spray karun"}, timeout=90).json()
print(d["diagnosis"]["disease"], "->", d["answer"] or d["message"])
```

**JavaScript / TypeScript (browser File)**
```ts
const BASE = "<BASE_URL>", KEY = "<API_KEY>";

async function vision(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/vision`, { method: "POST", headers: { "X-API-Key": KEY }, body: fd });
  return r.json();   // { label, crop, disease, confidence, top_k, note }
}

async function diagnose(file: File, question?: string) {
  const fd = new FormData();
  fd.append("file", file);
  if (question) fd.append("question", question);
  const r = await fetch(`${BASE}/diagnose`, { method: "POST", headers: { "X-API-Key": KEY }, body: fd });
  return r.json();   // { diagnosis, tier, answer, sources, ... }
}
```

**React Native / Expo** — build the FormData from the picker asset:
```ts
const fd = new FormData();
fd.append("file", { uri: asset.uri, name: "leaf.jpg", type: "image/jpeg" } as any);
// then fetch(`${BASE}/diagnose`, { method:"POST", headers:{ "X-API-Key": KEY }, body: fd })
```

---

## 9. `/vision` vs `/diagnose` — which to call

| Use case | Call |
|---|---|
| Just show "what disease is this?" | `/vision` (fast) |
| One-tap "diagnose + tell me the treatment" | `/diagnose` |
| Diagnose, then let the user ask free-form follow-ups | `/vision` for the label, then `/query` with the disease + their question (and multi-turn `history`) |

---

*Contract questions → this doc. Base URL + API key shared separately.*