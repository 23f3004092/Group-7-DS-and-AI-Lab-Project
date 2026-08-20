# FarmerVision GCP API Specification

Version: 1.0  
Service: FarmerVision FastAPI Gateway  
Base URL: `http://34.93.236.19:8000`

This document describes the deployed model-inference API exposed by the FarmerVision GCP
instance. The gateway provides agricultural classification, retrieval-augmented answers,
leaf vision classification, and image-based diagnosis.

> Security note: the API key shown below is an example value supplied for this deployment.
> Do not commit real keys to a public repository or frontend application. Rotate the key if it
> has been exposed outside trusted server-side integrations.

## Quick Start

```bash
BASE_URL="http://34.93.236.19:8000"
API_KEY="1f4k0l9ayGEey9es5_-jNILS7aFYtSGEg-GIuu5FNJA"
```

All `POST` endpoints require this header:

```http
X-API-Key: 1f4k0l9ayGEey9es5_-jNILS7aFYtSGEg-GIuu5FNJA
```

`GET /health` does not require authentication.

## Endpoint Summary

| Method | Endpoint | Content type | Authentication | Purpose |
|---|---|---|---|---|
| GET | `/health` | none | No | Service and model status |
| POST | `/classify` | `application/json` | Yes | Intent, entities, guardrail, external-data hints |
| POST | `/query` | `application/json` | Yes | Main RAG and chat endpoint |
| POST | `/vision` | `multipart/form-data` | Yes | Leaf disease classification only |
| POST | `/diagnose` | `multipart/form-data` | Yes | Leaf classification plus grounded advice |

The gateway does not directly fetch mandi prices, weather, or yield predictions. The calling
application fetches those values and injects them into `/query` using `live_data`.

## 1. Health Check

### `GET /health`

Returns service readiness, GPU information, loaded modules, Qdrant collection information, and
startup errors.

```bash
curl -s "$BASE_URL/health"
```

Example response:

```json
{
  "status": "ok",
  "gpu": true,
  "gpu_name": "NVIDIA L4",
  "collection": "agri_knowledge",
  "points": 723439,
  "modules": {
    "retrieval": true,
    "generation": true,
    "ieg_model": true,
    "vision": true
  },
  "note": "mandi/weather/yield are provided by the caller via live_data",
  "errors": []
}
```

Possible `status` values are `starting` and `ok`.

## 2. Query Classification

### `POST /classify`

Classifies the farmer's message before the caller fetches optional external data.

#### Request

```json
{
  "query": "Aaj Varanasi mandi mein gehun ka bhav kya hai?"
}
```

#### Request fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `query` | string | Yes | Farmer's question. English, Hindi, or Hinglish is supported. |

#### Example request

```bash
curl -s -X POST "$BASE_URL/classify" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"Aaj Varanasi mandi mein gehun ka bhav kya hai?"}'
```

#### Example response

```json
{
  "intent": "general",
  "intents": ["general"],
  "retrieval_intent": "general",
  "blocked": false,
  "block_reason": null,
  "entities": {
    "crop": ["wheat"],
    "district": ["Varanasi"]
  },
  "guardrail_backend": "model+rules",
  "suggested_external": ["mandi_prices"]
}
```

Use `suggested_external` to decide whether the calling application should fetch `mandi_prices`,
`weather`, or `yield` data before calling `/query`.

## 3. Main Chat and RAG Endpoint

### `POST /query`

Runs guardrails, retrieves relevant agricultural knowledge, and generates an answer. It also
supports caller-injected mandi, weather, and yield data and multi-turn conversations.

#### Request fields

| Field | Type | Required | Default | Description |
|---|---|---:|---:|---|
| `query` | string | Yes | - | Current farmer message. |
| `intent` | string or null | No | automatic | `policy`, `field_practice`, or `general`. |
| `top_k` | integer or null | No | configured default | Maximum number of retrieved chunks. |
| `filters` | object or null | No | `{}` | Retrieval filters such as crop, district, source type, season, language, or year. |
| `live_data` | object or null | No | null | Mandi, weather, or yield facts fetched by the caller. |
| `skip_retrieval` | boolean | No | `false` | Skip Qdrant retrieval, useful for live-data-only questions. |
| `history` | array or null | No | null | Client-managed prior messages. |
| `include_content` | boolean | No | `false` | Include the actual retrieved chunk text in each returned source. |
| `session_id` | string or null | No | null | Conversation identifier for server-managed history. |

#### Basic request

```bash
curl -s -X POST "$BASE_URL/query" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Wheat mein yellow rust ke liye kya treatment hai?",
    "intent": "field_practice"
  }'
```

#### Example grounded response

```json
{
  "tier": "grounded",
  "blocked": false,
  "answer": "... Sources: [1], [2]",
  "grounded": true,
  "sources": [
    {
      "n": 1,
      "score": 0.71,
      "source_type": "kcc",
      "citation": {
        "corpus": "kcc",
        "record": "KCC Q&A",
        "crop": "wheat",
        "district": "Varanasi",
        "season": "Rabi",
        "query_type": "Plant Protection",
        "year": 2023
      }
    },
    {
      "n": 2,
      "score": 0.66,
      "source_type": "pdf",
      "citation": {
        "corpus": "pdf",
        "file": "advisory.pdf",
        "pages": [12, 13],
        "section": "Disease management",
        "doc_category": "crop_advisory",
        "district": null,
        "year": 2022
      }
    }
  ],
  "live_data_used": [],
  "top_score": 0.71,
  "intent": "field_practice",
  "lang": "hinglish",
  "guardrail_backend": "model+rules",
  "gen_ms": 11840,
  "out_tokens": 96,
  "session_id": null,
  "history": [],
  "latency_ms": 12010,
  "suggested_external": []
}
```

### Returning retrieved chunk content

By default, `sources` contains metadata and citations only. Set `include_content` to `true` to
return the actual text of every retrieved chunk used for the answer.

```bash
curl -s -X POST "$BASE_URL/query" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Wheat mein yellow rust ke liye kya treatment hai?",
    "intent": "field_practice",
    "include_content": true
  }'
```

A source then includes:

```json
{
  "n": 1,
  "score": 0.71,
  "source_type": "kcc",
  "citation": {"corpus": "kcc", "crop": "wheat"},
  "content": "For management of yellow rust in wheat, ..."
}
```

`content` is returned only for grounded or fallback responses that have retrieval context. It
is not returned for blocked, greeting, or empty-context responses.

### Live mandi, weather, and yield data

The gateway does not call external data providers. The caller should fetch the values and send
them in `live_data`.

```bash
curl -s -X POST "$BASE_URL/query" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Should I sell my wheat today in Varanasi?",
    "skip_retrieval": true,
    "live_data": {
      "mandi_prices": {
        "crop": "wheat",
        "market": "Varanasi",
        "modal_price_inr_per_quintal": 2480,
        "observed_at": "2026-08-20T09:30:00Z",
        "source": "caller-mandi-service"
      },
      "weather": {
        "location": "Varanasi",
        "forecast": "No rain expected for the next three days",
        "observed_at": "2026-08-20T09:00:00Z",
        "source": "caller-weather-service"
      }
    }
  }'
```

Recognised live-data keys include:

- `mandi_prices`, `mandi`, `market`, or `prices`
- `weather` or `forecast`
- `yield` or `yield_prediction`

Values may be strings, numbers, objects, or arrays. The calling application is responsible for
source validation, units, timestamps, freshness, and correct location/crop matching.

### Multi-turn conversation using `session_id`

Send the same `session_id` for each turn:

```bash
curl -s -X POST "$BASE_URL/query" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"My wheat has yellow rust","session_id":"farmer-42-chat-1"}'

curl -s -X POST "$BASE_URL/query" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"query":"What treatment should I use?","session_id":"farmer-42-chat-1"}'
```

The current gateway stores a limited conversation history in process memory. History is lost when
the service restarts, and callers should use a unique session ID per conversation.

### Multi-turn conversation using client-managed `history`

```json
{
  "query": "What treatment should I use?",
  "history": [
    {
      "role": "user",
      "content": "My wheat has yellow rust"
    },
    {
      "role": "assistant",
      "content": "Yellow rust is a fungal disease affecting wheat."
    }
  ]
}
```

The response includes an updated `history` array. Store that array and send it with the next
request when using client-managed history. Do not send both `session_id` and `history` unless you
intend the server-side session to take precedence.

### Response tiers

| Tier | Meaning |
|---|---|
| `grounded` | Retrieved context met the configured confidence threshold. |
| `fallback_with_disclaimer` | Context was weaker; the answer includes a KVK verification disclaimer. |
| `abstain_out_of_scope` | No usable context was found; the response is not grounded in retrieved sources. |
| `blocked` | Guardrail rejected the request. |
| `skipped` | Retrieval was explicitly skipped. |
| `error` | Retrieval failed; inspect the `error` field and service logs. |

## 4. Vision Classification

### `POST /vision`

Classifies a leaf image without generating treatment advice.

Request format is `multipart/form-data` with a required field named `file`.

```bash
curl -s -X POST "$BASE_URL/vision" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@leaf.jpg"
```

Example response:

```json
{
  "label": "wheat__yellow_rust",
  "crop": "wheat",
  "disease": "yellow rust",
  "confidence": 0.94,
  "top_k": [
    {"label": "wheat__yellow_rust", "prob": 0.94},
    {"label": "wheat__brown_rust", "prob": 0.03}
  ],
  "note": "Lab-trained model; treat as a suggestion, confirm with a local expert."
}
```

## 5. Image Diagnosis

### `POST /diagnose`

Classifies a leaf image, retrieves agricultural treatment information for the predicted crop and
disease, and generates grounded advice.

Form fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `file` | image upload | Yes | Leaf image. |
| `question` | string | No | Optional question about the uploaded image. |

```bash
curl -s -X POST "$BASE_URL/diagnose" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@leaf.jpg" \
  -F "question=Iske liye kya spray karun?"
```

Example response:

```json
{
  "diagnosis": {
    "label": "wheat__yellow_rust",
    "crop": "wheat",
    "disease": "yellow rust",
    "confidence": 0.94,
    "note": "Lab-trained model; treat as a suggestion, confirm with a local expert."
  },
  "tier": "grounded",
  "answer": "... Sources: [1]",
  "sources": [
    {
      "n": 1,
      "score": 0.68,
      "source_type": "pdf",
      "citation": {"corpus": "pdf", "file": "advisory.pdf", "pages": [12, 13]}
    }
  ],
  "gen_ms": 12000,
  "out_tokens": 88,
  "latency_ms": 12500
}
```

## 6. Retrieval Filters

The `filters` object on `/query` is passed to the knowledge-base search. Supported fields include:

| Field | Example | Description |
|---|---|---|
| `source_type` | `pdf` | Search only `pdf` or `kcc` records. |
| `doc_category` | `crop_advisory` | Filter PDF category. |
| `query_type` | `Plant Protection` | Filter KCC question category. |
| `crop` | `wheat` | Filter crop. |
| `district` | `Varanasi` | Filter district. |
| `season` | `Rabi` | Filter season. |
| `language` | `hi` | Filter language. |
| `year_from` | `2022` | Include records from this year onward. |
| `only_tables` | `true` | Restrict to table records where supported. |

Example:

```bash
curl -s -X POST "$BASE_URL/query" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{
    "query": "Wheat yellow rust treatment",
    "filters": {
      "crop": "wheat",
      "district": "Varanasi",
      "source_type": "kcc"
    },
    "include_content": true
  }'
```

## 7. Citation Objects

For PDF results, `citation` can contain:

```json
{
  "corpus": "pdf",
  "file": "advisory.pdf",
  "pages": [12, 13],
  "section": "Disease management",
  "doc_category": "crop_advisory",
  "district": null,
  "year": 2022
}
```

For KCC results, `citation` can contain:

```json
{
  "corpus": "kcc",
  "record": "KCC Q&A",
  "crop": "wheat",
  "district": "Varanasi",
  "season": "Rabi",
  "query_type": "Plant Protection",
  "year": 2023
}
```

## 8. Errors

| HTTP status | Response | Meaning |
|---:|---|---|
| `400` | `{"detail":"empty query"}` | Missing or blank query. |
| `400` | `{"detail":"empty image"}` | Uploaded file is empty. |
| `401` | `{"detail":"bad or missing X-API-Key"}` | Missing or incorrect API key when authentication is configured. |
| `501` | `{"detail":"vision model not deployed"}` | Vision model is unavailable on this deployment. |
| `500` | FastAPI error response | Unexpected server-side failure. |

Example unauthorized request:

```bash
curl -i -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is wheat rust?"}'
```

## 9. Python Client Example

```python
import requests

BASE_URL = "http://34.93.236.19:8000"
API_KEY = "1f4k0l9ayGEey9es5_-jNILS7aFYtSGEg-GIuu5FNJA"
HEADERS = {"X-API-Key": API_KEY}


def ask(query, session_id=None, include_content=False, live_data=None):
    body = {
        "query": query,
        "session_id": session_id,
        "include_content": include_content,
    }
    if live_data is not None:
        body["live_data"] = live_data

    response = requests.post(
        f"{BASE_URL}/query",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


first = ask("My wheat has yellow rust", session_id="farmer-42-chat-1")
second = ask("What treatment should I use?", session_id="farmer-42-chat-1")
print(second["answer"])

with open("leaf.jpg", "rb") as image:
    response = requests.post(
        f"{BASE_URL}/vision",
        headers=HEADERS,
        files={"file": ("leaf.jpg", image, "image/jpeg")},
        timeout=60,
    )
    response.raise_for_status()
    print(response.json())
```

## 10. JavaScript Client Example

Use this from a trusted server-side application. Do not expose the API key in browser code.

```javascript
const BASE_URL = "http://34.93.236.19:8000";
const API_KEY = "1f4k0l9ayGEey9es5_-jNILS7aFYtSGEg-GIuu5FNJA";

async function ask(query, sessionId) {
  const response = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query, session_id: sessionId })
  });

  if (!response.ok) {
    throw new Error(`FarmerVision request failed: ${response.status}`);
  }
  return response.json();
}

const first = await ask("My wheat has yellow rust", "farmer-42-chat-1");
const second = await ask("What treatment should I use?", "farmer-42-chat-1");
console.log(second.answer);
```

## 11. Operational Notes

- The first request after startup can be slower while models warm up.
- Keep the API key on a trusted backend, never in a public mobile or browser bundle.
- The caller owns mandi, weather, and yield retrieval and should include timestamps and source
  information in `live_data`.
- `include_content` can substantially increase response size; enable it only when the client needs
  the source text.
- Use a unique `session_id` for each user conversation.
- The current gateway keeps server-side session history in process memory; restarting the service
  clears it.
- For interactive API exploration, FastAPI normally exposes `/docs` and `/openapi.json`.
