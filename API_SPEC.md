# FarmerVision API — Integration Guide

A REST API for an agricultural assistant for farmers in Uttar Pradesh, India. It answers
crop/pest/fertiliser/scheme questions (English · Hindi · Hinglish) grounded in a knowledge
base with citations, classifies leaf-disease photos, holds multi-turn conversations, and can
weave in live data (mandi prices / weather / yield) that **your** app provides.

> You don't need to know anything about where this runs. You need exactly two things from
> whoever deployed it: the **Base URL** and the **API key**.

---

## 1. Connection

| | |
|---|---|
| **Base URL** | `http://<HOST>:8000`  ← get from the deployer |
| **API key** | a secret string ← get from the deployer |
| **Auth** | every `POST` needs header `X-API-Key: <API key>` (`GET /health` needs none) |
| **Body** | JSON (`Content-Type: application/json`) except `/vision` & `/diagnose` which use `multipart/form-data` |
| **CORS/TLS** | plain HTTP; call it server-side (don't embed the key in a browser app) |

Set these once for the examples below:
```bash
BASE=http://<HOST>:8000
KEY=<your-api-key>
```

---

## 2. Endpoints at a glance

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + what's loaded (no key) |
| POST | `/classify` | intent + guardrail + which external data a query needs |
| POST | `/query` | **main endpoint** — grounded, cited answer; supports multi-turn + live data |
| POST | `/vision` | leaf photo → disease label + confidence |
| POST | `/diagnose` | leaf photo (+ optional question) → disease + grounded treatment |

Typical flow: *(optional)* `/classify` to see if the question needs mandi/weather/yield → fetch
those yourself → `/query` with them in `live_data`. For plain Q&A, just call `/query`.

---

## 3. `GET /health`
No auth, no body.
```json
{"status":"ok","gpu":true,"gpu_name":"NVIDIA L4","collection":"agri_knowledge",
 "points":723439,"modules":{"retrieval":true,"generation":true,"ieg_model":false,"vision":true},
 "note":"mandi/weather/yield are provided by the caller via live_data","errors":[]}
```
`status:"ok"` means it's ready. `ieg_model:false` is normal (guardrail runs on rules + keywords).

---

## 4. `POST /classify`
Tells you the intent, whether it's blocked, and which external data the query is asking for —
so you know what (if anything) to fetch before `/query`.

**Request**
```json
{"query": "aaj mandi me gehu ka bhav kya hai"}
```

**Response**
```json
{"intent":"cultivation_practice","retrieval_intent":"general","blocked":false,
 "block_reason":null,"entities":{},"guardrail_backend":"rules-only",
 "suggested_external":["mandi_prices"]}
```
| field | meaning |
|---|---|
| `intent` | fine-grained intent (see §8) |
| `retrieval_intent` | `policy` / `field_practice` / `general` — pass this as `intent` to `/query` if you like |
| `blocked` | `true` if the guardrail would refuse this query |
| `suggested_external` | which live data to fetch: any of `mandi_prices`, `weather`, `yield` |

---

## 5. `POST /query` — the main endpoint

**Request body**

| field | type | required | meaning |
|---|---|---|---|
| `query` | string | ✅ | the farmer's question (English / Hindi / Hinglish) |
| `intent` | string | | `policy` \| `field_practice` \| `general` — biases retrieval (omit → auto) |
| `top_k` | int | | number of context chunks (default 5) |
| `filters` | object | | narrow the search — see §8 |
| `live_data` | object | | facts you fetched (mandi/weather/yield) to inject — see §7 |
| `skip_retrieval` | bool | | `true` → answer from `live_data` only, skip the knowledge base |
| `session_id` | string | | multi-turn: server remembers this conversation — see §6 |
| `history` | array | | multi-turn: you send prior turns — see §6 |

**Response (answered)**
```json
{
  "tier": "grounded",
  "blocked": false,
  "answer": "Spray Mancozeb 75 WP at 400 g/acre in 200 L water. [1] ...\nSources: [1], [2]",
  "sources": [
    {"n":1,"score":0.71,"source_type":"kcc",
     "citation":{"corpus":"kcc","crop":"wheat","district":"Varanasi","season":"Rabi","query_type":"Plant Protection","year":2023}},
    {"n":2,"score":0.66,"source_type":"pdf",
     "citation":{"corpus":"pdf","file":"advisory.pdf","pages":[12,13],"doc_category":"crop_advisory","year":2022}}
  ],
  "live_data_used": [],
  "top_score": 0.71,
  "intent": "field_practice",
  "lang": "hinglish",
  "guardrail_backend": "rules-only",
  "gen_ms": 11840, "out_tokens": 96, "latency_ms": 12010,
  "session_id": null,
  "history": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
  "suggested_external": []
}
```

**Response (guardrail blocked)** — off-topic or restricted queries:
```json
{"tier":"blocked","blocked":true,"block_reason":"non_agricultural",
 "answer":"I can only help with farming and agriculture questions.","sources":[], ...}
```

**Response (nothing relevant, no live data)** — the assistant abstains; `message` is localized to the query language:
```json
{"tier":"abstain_out_of_scope","blocked":false,"answer":null,
 "message":"This is outside the agricultural knowledge base ...","sources":[], ...}
```

**How to read a response**
- If `answer` is non-null → show it. It already ends with a `Sources: [n]` line.
- If `answer` is null → show `message` (it's the localized "can't help / rephrase" text).
- `tier`: `grounded` (confident) · `fallback_with_disclaimer` (answer + a KVK-verify note appended) · `abstain_out_of_scope` (no answer) · `blocked` (refused) · `skipped` (you set `skip_retrieval`).
- `lang`: the language the answer is written in (`en` / `hi` / `hinglish`) — matches the query.
- `sources`: what the answer is grounded in; `citation` differs for `pdf` vs `kcc` (see §9).

---

## 6. Multi-turn conversations

Pick **one** of two modes:

### Mode A — `session_id` (server remembers) — easiest
Send the same `session_id` string for every message in a conversation. The gateway keeps the
last few turns in memory and resolves follow-ups automatically.
```bash
# turn 1
curl -s -X POST $BASE/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"wheat me yellow rust ki dawa batao","session_id":"user42-chat","intent":"field_practice"}'
# turn 2 — "its dose?" resolves against turn 1
curl -s -X POST $BASE/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"iski dose kitni honi chahiye","session_id":"user42-chat","intent":"field_practice"}'
```
Use a unique `session_id` per user/conversation (e.g. `user42-chat`). Server memory is
in-process and resets if the service restarts — for durable history use Mode B.

### Mode B — `history` (you keep it) — stateless, durable
Send the prior turns yourself; the response returns the updated `history` to store and resend.
```json
{"query":"iski dose kitni honi chahiye","intent":"field_practice",
 "history":[
   {"role":"user","content":"wheat me yellow rust ki dawa batao"},
   {"role":"assistant","content":"Spray Mancozeb 75 WP ... Sources: [1]"}
 ]}
```
Take `response.history` from each call and send it back on the next. Keep it to the last ~4 turns.

Either way, follow-ups like *"iski dose?"*, *"aur kitna?"*, *"wahi"* are understood, and retrieval
is contextualized with the recent turns.

---

## 7. Injecting live data (mandi prices / weather / yield)

The API does **not** fetch these — your app does, then passes them in `live_data`. The model treats
them as authoritative and uses the exact values.
```json
{"query":"gehu abhi bech du ya rukun?",
 "live_data":{
   "mandi_prices":"Wheat @ Varanasi mandi: Rs 2480/quintal (2026-08-14)",
   "weather":{"rain_next_3d":"none","temp_c":[28,33]},
   "yield":"2.6 t/ha estimated"
 }}
```
- Keys are free-form; recognised ones get nice labels: `mandi_prices`, `market`, `weather`, `forecast`, `yield`, `yield_prediction`.
- Values may be strings or nested JSON.
- For a pure weather/price question that needs no knowledge-base lookup, add `"skip_retrieval": true`.
- `response.suggested_external` tells you what the query *looked like* it wanted, in case you forgot to send it.

---

## 8. `POST /vision` and `POST /diagnose`

Both take an image as `multipart/form-data`.

### `/vision` — classify only
```bash
curl -s -X POST $BASE/vision -H "X-API-Key: $KEY" -F "file=@leaf.jpg"
```
```json
{"label":"wheat__yellow_rust","crop":"wheat","disease":"yellow rust","confidence":0.94,
 "top_k":[{"label":"wheat__yellow_rust","prob":0.94},{"label":"wheat__brown_rust","prob":0.03}],
 "note":"Lab-trained model; treat as a suggestion, confirm with a local expert."}
```

### `/diagnose` — classify + grounded treatment
```bash
curl -s -X POST $BASE/diagnose -H "X-API-Key: $KEY" \
  -F "file=@leaf.jpg" -F "question=is ke liye kya spray karun"
```
```json
{"diagnosis":{"label":"wheat__yellow_rust","crop":"wheat","disease":"yellow rust","confidence":0.94,...},
 "tier":"grounded","answer":"Likely wheat yellow rust. Spray ... Sources: [1]",
 "sources":[{"n":1,"score":0.68,"source_type":"pdf","citation":{...}}],
 "gen_ms":12000,"out_tokens":88,"latency_ms":12500}
```
`question` is an optional form field. Accepted image types: JPG / PNG.

---

## 9. Value reference

**`intent`** (on `/query`):

| value | best for |
|---|---|
| `policy` | schemes, subsidies, eligibility, government guidelines |
| `field_practice` | how-to, dosage, disease/pest, cultivation |
| `general` | anything else / unsure (default) |

**`filters`** (object on `/query`, all optional):

| key | allowed values |
|---|---|
| `source_type` | `pdf` \| `kcc` |
| `doc_category` | `scheme_eligibility` \| `crop_advisory` \| `contingency_plan` \| `policy_guideline` (pdf) |
| `query_type` | KCC category, e.g. `Plant Protection` (kcc) |
| `crop` | e.g. `rice`, `wheat` |
| `district` | UP district, e.g. `Varanasi` |
| `season` | `Rabi` \| `Kharif` \| `Zaid` (kcc) |
| `language` | `en` \| `hi` \| `mixed` |
| `year_from` | integer |
| `only_tables` | `true` (dosage/scheme tables only, pdf) |

**`tier`** (response): `grounded` · `fallback_with_disclaimer` · `abstain_out_of_scope` · `blocked` · `skipped`.

**IEG `intent`** (from `/classify`): `cultivation_practice`, `disease_pest`, `nutrition_fertilizer`, `post_harvest_storage`, `specialty_other`, `general`, `non_agri`.

**`suggested_external`** / live-data keys: `mandi_prices`, `weather`, `yield`.

**Vision `label`** — 20 classes, formatted `crop__disease`:
`rice__blast`, `rice__bacterial_blight`, `rice__brown_spot`, `rice__tungro`, `rice__leaf_smut`,
`wheat__healthy`, `wheat__yellow_rust`, `wheat__brown_rust`, `wheat__black_rust`, `wheat__blast`,
`wheat__septoria`, `wheat__mildew`, `wheat__aphid`, `wheat__mite`, `wheat__stem_fly`,
`wheat__smut`, `wheat__tan_spot`, `wheat__leaf_blight`, `wheat__common_root_rot`, `wheat__fusarium_head_blight`.

**`citation`** object inside `sources` (varies by corpus):
- pdf → `{"corpus":"pdf","file":..., "pages":[start,end], "section":..., "doc_category":..., "district":..., "year":...}`
- kcc → `{"corpus":"kcc","record":"KCC Q&A","crop":..., "district":..., "season":..., "query_type":..., "year":...}`

---

## 10. Errors

| HTTP | body | cause |
|---|---|---|
| 401 | `{"detail":"bad or missing X-API-Key"}` | wrong/absent API key |
| 400 | `{"detail":"empty query"}` / `{"detail":"empty image"}` | missing input |
| 501 | `{"detail":"vision model not deployed"}` | vision not available on this instance |
| 500 | `{"detail":"..."}` | server error — retry; tell the deployer if it persists |

The first `/query` after the service (re)starts is slower (~10–15 s) while the model warms up;
subsequent calls are quick.

---

## 11. Client code snippets

**Python**
```python
import requests
BASE = "http://<HOST>:8000"
KEY  = "<your-api-key>"

def ask(query, session_id=None, live_data=None, intent="field_practice"):
    body = {"query": query, "intent": intent}
    if session_id: body["session_id"] = session_id
    if live_data:  body["live_data"]  = live_data
    r = requests.post(f"{BASE}/query", headers={"X-API-Key": KEY}, json=body, timeout=60)
    r.raise_for_status()
    d = r.json()
    return d.get("answer") or d.get("message"), d["tier"]

print(ask("wheat me yellow rust ki dawa", session_id="u1"))
print(ask("iski dose kitni",              session_id="u1"))   # follow-up, remembered

# classify a photo:
with open("leaf.jpg", "rb") as f:
    v = requests.post(f"{BASE}/vision", headers={"X-API-Key": KEY}, files={"file": f}).json()
print(v["crop"], v["disease"], v["confidence"])
```

**JavaScript (Node / server-side)**
```js
const BASE = "http://<HOST>:8000", KEY = "<your-api-key>";

async function ask(query, sessionId) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "X-API-Key": KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ query, intent: "field_practice", session_id: sessionId }),
  });
  const d = await res.json();
  return { answer: d.answer ?? d.message, tier: d.tier };
}

console.log(await ask("wheat me yellow rust ki dawa", "u1"));
console.log(await ask("iski dose kitni", "u1"));   // follow-up
```

---

*Questions about the API contract → this doc. To get the Base URL and API key, ask the deployer.*
