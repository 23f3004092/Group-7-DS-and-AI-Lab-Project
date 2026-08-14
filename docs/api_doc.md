# FarmerVision — API Documentation

*Milestone 6 · Section D*

FarmerVision exposes **two REST API surfaces**:

1. **Product Backend API** (`/api/*`) — the app-facing API (mandi, weather, advisory, admin, MCP)
   consumed by the mobile app and admin dashboard. Lives in `backend/` (`tanmay` branch).
2. **Model-Inference API** — the deployed distilled models (RAG generation, vision, guardrail) on
   the GCP GPU VM. Full standalone spec: `docs/internal/do_not_open/API_SPEC.md`.

---

## Part 1 — Product Backend API

**Base URL:** `http://<host>:8000` **·** Interactive docs (Swagger): `GET /docs` **·** OpenAPI: `/openapi.json`
**Format:** JSON in / JSON out (image endpoints use `multipart/form-data`).
**Secrets:** external API keys live in the backend `.env` and are never exposed to clients.

### System

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service metadata (name, version, status, docs) |
| GET | `/health` | Component health: SQLite, vector DB, yield model, cloud AI, mandi, weather |

**`GET /health`**
```json
{"status":"ok","components":{"database":"ok","vector_db":"ok","yield_model":"ok",
 "cloud_ai":"ok","mandi":"ok","weather":"ok"}}
```

### Advisory — `/api/query`

| Method | Path | Body | Description |
|---|---|---|---|
| POST | `/text` | `{"query": "..."}` | RAG-grounded answer + sources |
| POST | `/image` | multipart `file` | leaf photo → crop/disease diagnosis |
| POST | `/multimodal` | multipart `file` + `query` | image + text → richer diagnosis |
| POST | `/yield` | `{"crop","district","area_ha"}` | yield estimate |
| POST | `/logs/{log_id}/feedback` | `{"score": -1..1}` | rate a response (clamped to ±1) |

**`POST /api/query/text`**
```bash
curl -s -X POST http://<host>:8000/api/query/text \
  -H "Content-Type: application/json" -d '{"query":"best medicine for yellow rust in wheat"}'
```
```json
{"answer":"Spray Mancozeb 75 WP at 400 g/acre ... ","sources":[{"title":"...","snippet":"..."}],
 "log_id":123}
```

**`POST /api/query/yield`**
```bash
curl -s -X POST http://<host>:8000/api/query/yield \
  -H "Content-Type: application/json" -d '{"crop":"Wheat","district":"Varanasi","area_ha":2.0}'
```
```json
{"crop":"Wheat","district":"Varanasi","area_ha":2.0,"yield_t_ha":2.6,"total_t":5.2}
```

### Mandi prices — `/api/mandi`

| Method | Path | Query | Description |
|---|---|---|---|
| GET | `/prices` | `crop`, `state`, `district`, `market` | latest prices (live + MSP fallback) |
| GET | `/districts` | `state` | district pick-list (75 UP districts, all states) |
| GET | `/states` | — | supported states/UTs |

```bash
curl -s "http://<host>:8000/api/mandi/prices?state=Uttar%20Pradesh&district=Lucknow&crop=Wheat"
```
```json
{"prices":[{"crop":"Wheat","market":"Lucknow","modal_price":2596.53,"change":5.85,"unit":"₹/quintal","source":"live"}]}
```

### Weather — `/api/weather`

| Method | Path | Query | Description |
|---|---|---|---|
| GET | `/current` | `lat`+`lon` **or** `city` | current + 3-day forecast |

```bash
curl -s "http://<host>:8000/api/weather/current?lat=26.85&lon=80.95"
```
```json
{"location":"Lucknow","temp_c":31,"condition":"Partly cloudy","rain_mm":0,"humidity":54,
 "forecast":[{"day":"Tue","min":26,"max":34},{"day":"Wed","min":27,"max":33}],"source":"open_meteo"}
```

### Admin — `/api/admin`

| Method | Path | Description |
|---|---|---|
| GET | `/stats` | aggregated usage statistics |
| GET | `/logs` | terminal-attached chat logs |
| GET | `/config` | read runtime config |
| POST | `/config` | update runtime config |
| GET | `/vectordb` | vector DB status / query (with local fallback) |

### Model Context Protocol — `/api/mcp`

| Method | Path | Description |
|---|---|---|
| GET | `/tools` | list exposed MCP tools |
| POST | `/tools/call` | invoke an MCP tool |
| POST | `/rpc` | raw JSON-RPC transport |

---

## Part 2 — Model-Inference API (deployed on GCP)

The distilled research models served on the GPU VM. **Base URL:** `http://<HOST>:8000` ·
**Auth:** header `X-API-Key: <key>` on every POST (`GET /health` needs none). Full contract with
schemas, multi-turn, and client snippets: `docs/internal/do_not_open/API_SPEC.md`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness + what's loaded |
| POST | `/classify` | intent + guardrail + which external data a query needs |
| POST | `/query` | grounded, cited RAG answer; multi-turn (`session_id`/`history`) + `live_data` |
| POST | `/vision` | leaf photo → disease label + confidence |
| POST | `/diagnose` | leaf photo (+ question) → disease + grounded treatment |

**`POST /query`**
```bash
curl -s -X POST http://<HOST>:8000/query \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"query":"wheat me yellow rust ki dawa","intent":"field_practice","session_id":"chat1"}'
```
```json
{"tier":"grounded","answer":"...\nSources: [1], [2]",
 "sources":[{"n":1,"score":0.71,"source_type":"kcc","citation":{...}}],
 "lang":"hinglish","top_score":0.71,"gen_ms":11840,"latency_ms":12010,
 "session_id":"chat1","suggested_external":[]}
```

### Request fields (`/query`)
| field | type | meaning |
|---|---|---|
| `query` | string (required) | the question (en/hi/hinglish) |
| `intent` | string | `policy` \| `field_practice` \| `general` |
| `top_k` | int | context chunks (default 5) |
| `filters` | object | narrow search (`crop`, `district`, `source_type`, `season`, ...) |
| `live_data` | object | inject mandi/weather/yield facts |
| `skip_retrieval` | bool | answer from `live_data` only |
| `session_id` / `history` | string / array | multi-turn |

### Common response fields
`tier` (`grounded`/`fallback_with_disclaimer`/`abstain_out_of_scope`/`blocked`/`skipped`),
`answer` (or `message` when abstaining), `sources[]` with `citation`, `lang` (en/hi/hinglish),
`top_score`, `latency_ms`.

### Errors (both surfaces)
| HTTP | body | cause |
|---|---|---|
| 401 | `{"detail":"bad or missing X-API-Key"}` | wrong/absent key (inference API) |
| 400 | `{"detail":"empty query"}` | missing input |
| 429 | (mandi) rate-limited upstream | falls back to MSP |
| 500 | `{"detail":"..."}` | server error — retry |

---

## Relationship between the two APIs

The **product backend** is the app's single entry point; it orchestrates retrieval + generation and
adds mandi/weather/MCP. The **model-inference API** hosts the distilled models on a GPU. The backend
can call the inference API (or a cloud model) for generation per configuration, and both share the
same Qdrant `agri_knowledge` knowledge base. Integrators building on the deployed models only need
the model-inference API and `API_SPEC.md`.
