# FarmerVision

FarmerVision is a multimodal agricultural advisory platform for Indian farmers. It delivers AI-powered crop advice, mandi (market) price intelligence, yield estimates, and live weather for the farmer's location — across a mobile app, a web admin dashboard, and an MCP (Model Context Protocol) server.

This document covers the architecture, the backend REST API, the mobile app, and a log of what has been built so far.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Repository Layout](#repository-layout)
3. [Backend API](#backend-api)
4. [External Data Integrations](#external-data-integrations)
5. [Mobile App](#mobile-app)
6. [Admin Dashboard](#admin-dashboard)
7. [Setup & Running](#setup--running)
8. [Work Log](#work-log)

---

## Architecture

```
┌──────────────┐      REST / JSON       ┌─────────────────────────────┐
│  Expo mobile │ ────────────────────▶ │  FastAPI backend (:8000)    │
│   app (RN)   │ ◀───────────────────── │  • RAG advisory             │
└──────────────┘        + /uploads      │  • Leaf-image diagnosis     │
                                        │  • Yield estimation         │
┌──────────────┐      REST / JSON       │  • Mandi prices (data.gov.in│
│ Admin web    │ ────────────────────▶ │  • Weather (Open-Meteo)     │
│ dashboard    │ ◀───────────────────── │  • Terminal-attached logs   │
└──────────────┘                        └──────────────┬──────────────┘
                                                       │
          ┌──────────────────────────┬─────────────────┴──────────────┐
          ▼                          ▼                                 ▼
   SQLite (logs/feedback)    Qdrant vector DB           External APIs
                             (agri_knowledge)        data.gov.in (mandi)
                                                     Open-Meteo (weather)
                                                     indianapi.in (IMD, staging)
                                                     Gemini / cloud models
```

- **Backend**: FastAPI + SQLModel/SQLAlchemy, serving a monorepo `backend/` package. Five router groups.
- **Vector retrieval**: Qdrant (`agri_knowledge` collection, bge-m3 1024-dim embeddings). If Qdrant is not running, a **local in-memory fallback** (8 points) keeps the RAG flow functional.
- **Admin viewers**: terminal-isolated chat viewers, plus a React/Vite admin UI in `admin/`.

---

## Repository Layout

```
backend/
  run.py              # uvicorn entry point (reload enabled)
  app/
    main.py           # FastAPI app, CORS, router wiring, /health
    config.py         # env-driven settings (API keys, TTLs)
    database.py       # SQLAlchemy engine + Base (SQLite)
    models.py         # ORM models
    schemas.py        # Pydantic response schemas
    routers/
      query.py        # /api/query    (RAG text, image, multimodal, yield, feedback)
      mandi.py        # /api/mandi    (market prices, states, districts)
      weather.py      # /api/weather  (live current + forecast)
      admin.py        # /api/admin    (stats, logs, config, vector DB)
      mcp.py          # /api/mcp      (Model Context Protocol tools/RPC)
    services/
      cloud_models.py # Gemini / cloud model orchestration
      qdrant_service.py
      yield_service.py
      mandi_service.py
      weather_service.py
mobile/               # Expo React Native app (App.js, i18n, expo-location)
admin/                # React/Vite admin dashboard
scripts/              # utility scripts
```

---

## Backend API

Base URL: `http://<host>:8000`. Interactive docs at `/docs` (Swagger).

### System

| Method | Path            | Description                                        |
| ------ | --------------- | -------------------------------------------------- |
| GET    | `/`             | Service metadata (name, version, status, docs)     |
| GET    | `/health`       | Component health: SQLite, vector DB, yield model, cloud AI, mandi, weather |

### Advisory Query Pipelines — `/api/query`

| Method | Path                      | Description                                                |
| ------ | ------------------------- | ---------------------------------------------------------- |
| POST   | `/text`                   | Ask a text question; returns RAG-grounded answer + sources |
| POST   | `/image`                  | Upload a leaf photo → crop/disease diagnosis               |
| POST   | `/multimodal`             | Combine image + text for a richer diagnosis               |
| POST   | `/yield`                  | Estimate yield; body `{crop, district, area_ha}`           |
| POST   | `/logs/{log_id}/feedback` | Rate a response (validators clamp to ±1 range)             |

### Mandi Market Prices — `/api/mandi`

| Method | Path        | Description                                                        |
| ------ | ----------- | ------------------------------------------------------------------ |
| GET    | `/prices`   | Latest prices; query `crop`, `state`, `district`, `market`         |
| GET    | `/districts`| District pick-list for a `state` (75 UP districts, all states)     |
| GET    | `/states`   | All supported states/UTs for the location selector                 |

### Weather — `/api/weather`

| Method | Path       | Description                                                          |
| ------ | ---------- | -------------------------------------------------------------------- |
| GET    | `/current` | Current + 3-day forecast; GPS via `lat`/`lon`, else `city` fuzzy match |

### Admin Dashboard — `/api/admin`

| Method | Path         | Description                                    |
| ------ | ------------ | ---------------------------------------------- |
| GET    | `/stats`     | Aggregated usage statistics                     |
| GET    | `/logs`      | Terminal-attached chat logs                     |
| GET    | `/config`    | Read runtime config                             |
| POST   | `/config`    | Update runtime config                           |
| GET    | `/vectordb`  | Vector DB status / query (with local fallback)  |

### Model Context Protocol — `/api/mcp`

| Method | Path          | Description                     |
| ------ | ------------- | ------------------------------- |
| GET    | `/tools`      | List exposed MCP tools          |
| POST   | `/tools/call` | Invoke an MCP tool              |
| POST   | `/rpc`        | Raw JSON-RPC transport          |

> **Security note:** API keys (`GEMINI_API_KEY`, `MANDI_API_KEY`, `WEATHER_API_KEY`) live in a gitignored `.env` and are **never** exposed to client code — all external calls are proxied through the backend.

---

## External Data Integrations

### Mandi prices — data.gov.in (Agmarknet)
- **Key**: `MANDI_API_KEY` (100 requests/day — rate-limited).
- **Service**: `mandi_service.py` — per-crop resilient fetch (`_fetch_composite`), case-insensitive canonicalization of state/district names, 6 h cache, and a static **MSP fallback** when the API is unavailable or a crop has no record.
- Headline crops on the home screen always surface **Wheat, Paddy, Maize, Mustard** (missing ones get an MSP-reference row).
- data.gov.in requires the `User-Agent: curl/8.0.1` header via httpx or it hangs.

### Weather — Open-Meteo (live) + indianapi.in (staging)
- **Provider selection**: `WEATHER_PROVIDER=open_meteo` (default, keyless, live) or `indian` (requires `WEATHER_API_KEY`).
- **Open-Meteo**: current conditions + 3-day forecast from GPS coordinates or a geocoded city name; WMO weather codes mapped to human-readable labels/emoji. 30 min cache + static fallback.
- **indianapi.in / IMD**: fully wired (endpoints `/global/current`, `/india/weather`, `/india/weather_by_id`), but its data endpoints were returning internal errors as of Aug 2026 — so Open-Meteo is served until it recovers. Flip `WEATHER_PROVIDER=indian` to switch.

### Vector retrieval — Qdrant (with local fallback)
- bge-m3 embeddings (1024-dim), collection `agri_knowledge`.
- When the Qdrant server is down, retrieval degrades to an in-memory local mode so advisory/chat still works.

---

## Mobile App

Expo React Native app in `mobile/` (`App.js`). Five tabs: **Home, Leaf Scanner, Advisor Chat, Yield, Settings**.

- **Theme system**: Light / Dark / High-Contrast modes, 6 accent colours, font-size scaling, all stored in `expo-sqlite/kv-store` (`farmervision.settings.v2`). Multi-language support via `react-i18next` (`locales/`, `i18n.js`).
- **Location-aware home**: a live **Mandi** card (one row per crop, `● Live` badge) and a live **Weather** card (temp, condition, location, rain mm, humidity, 3-day ↑/↓ forecast).
- **GPS / location flow** (`expo-location`): auto-detect on first launch, "Use My Location" in Settings, a manual **State → District** picker fed by `/api/mandi/states` + `/api/mandi/districts`. GPS stores `lat`/`lon`; manual selection clears them so weather falls back to city lookup.
- **Leaf Scanner**: upload/choose a photo, diagnose via `/api/query/image` (offline mock fallback).
- **Yield Estimator**: crop/district/area → `/api/query/yield` (local fallback computation).
- **Advisor Chat**: RAG conversation with clickable citation chips.

---

## Admin Dashboard

React/Vite web app in `admin/` consuming `/api/admin/*` — usage stats, terminal-attached viewer logs, runtime config management, and vector-DB viewer. Chat sessions are isolated per login for privacy.

---

## Setup & Running

```bash
# Backend
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env            # set GEMINI_API_KEY, MANDI_API_KEY, WEATHER_API_KEY
./venv/bin/python run.py        # serves on :8000 with auto-reload

# Optional: Qdrant vector DB (skipped when not running; local fallback used)
docker-compose up -d qdrant     # or docker run ... qdrant/qdrant

# Mobile (Expo)
cd mobile && npm install && npx expo start
```

> `.env` is gitignored. Place all secrets there, e.g.:
> ```
> GEMINI_API_KEY=
> MANDI_API_KEY=<data.gov.in Agmarknet key>
> WEATHER_API_KEY=<indianapi.in key, optional>
> WEATHER_PROVIDER=open_meteo   # or 'indian'
> ```

---

## Work Log

### System hardening & verification
- Stack verified end-to-end: all 14 endpoints tested (text, image, multimodal, yield, feedback, admin, MCP).
- Fixed `FeedbackSubmit` import + `Body(...)` in `query.py` (this broke `/openapi.json` with a 500).
- Added `file.filename=None` guards and a `field_validator` clamping feedback scores to ±1.

### Mandi (market prices) integration
- Rewrote around **client-driven location**: `state`/`district` now come from the device (GPS or manual picker), with server config only as a fallback default.
- Built `mandi_service.py`: composite per-crop fetch, caching (6 h live / 5 min fallback), name canonicalization, and MSP fallback.
- `GET /api/mandi/prices`, `/api/mandi/districts`, `/api/mandi/states`.
- Verified live: Meerut (83 rows), Lucknow (wheat ₹2,596.53, +₹5.85).
- Mobile: location-driven mandi card, Live badge, one-row-per-crop dedupe, GPS + district picker UI; `expo-location ~57.0.9` wired in `app.json`.

### Weather integration
- Added `weather_service.py` (**hybrid provider**): Open-Meteo live default, indianapi.in (IMD) fully wired behind `WEATHER_PROVIDER=indian`, cache + static fallback.
- `GET /api/weather/current` (GPS coords or city), registered in `main.py`, surfaced in `/health`.
- Mobile: GPS detection stores coordinates; weather card now shows live temp, condition, location, rain mm, humidity, and a 3-day forecast with a `● Live` badge; static card retained as offline fallback.

### Outstanding / shortcuts
- indianapi.in weather **data endpoints are broken upstream** (Aug 2026) — Open-Meteo covers live weather; verify `WEATHER_PROVIDER=indian` once the provider recovers.
- data.gov.in uses a shared/demo key, intermittently rate-limited (HTTP 429) — MSP fallback covers this.
- Qdrant server not running in this environment — retrieval runs in local fallback mode (8 points).