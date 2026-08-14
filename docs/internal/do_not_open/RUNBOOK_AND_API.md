# FarmerVision — Deployment Runbook & API Reference

The one document for running, stopping, changing, and using the deployed gateway.
Keep it; you shouldn't need to remember any commands.

**What's deployed**

| Thing | Value |
|---|---|
| GCP project | `project-7f232935-b8f7-4aad-881` |
| VM name | `farmervision` |
| Zone | `us-east4-c` (g2-standard-8 + 1× NVIDIA L4) |
| Bucket | `gs://farmervision-prod-artifacts` |
| Gateway | FastAPI in Docker, port `8000` |
| Vector DB | Qdrant in Docker, `agri_knowledge`, 723,439 vectors |
| Serves | RAG (`/query`), vision (`/vision`, `/diagnose`), intent (`/classify`), `/health` |
| Not served here | mandi prices / weather / yield → your app fetches these and injects via `live_data` |

---

## Table of Contents
1. [💰 Stop / Start / Costs (read first)](#1--stop--start--costs)
2. [Check it's healthy + view logs](#2-check-health--logs)
3. [Expose for external use](#3-expose-for-external-use)
4. [Where to change things (system prompt, tiers, etc.)](#4-where-to-change-things)
5. [Full setup from scratch (recreate)](#5-full-setup-from-scratch)
6. [API reference (endpoints, request/response)](#6-api-reference)
7. [Parameter values (all allowed options)](#7-parameter-values)
8. [Sample requests (every scenario)](#8-sample-requests)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. 💰 Stop / Start / Costs

> **You are billed for the L4 GPU only while the VM is RUNNING (~$0.85/hour).**
> Stop it whenever you finish. Data + setup are preserved.

### Stop (stops GPU billing)
From **Cloud Shell** (in `~/requiredforgcp`):
```bash
bash stop_vm.sh
```
or directly, from anywhere with gcloud:
```bash
gcloud compute instances stop farmervision --zone=us-east4-c
```

### Start again (~2–3 min; containers auto-restart)
```bash
bash start_vm.sh
```
or:
```bash
gcloud compute instances start farmervision --zone=us-east4-c
```
> After a restart the **external IP changes** (it's ephemeral). Re-fetch it (see §3) unless you reserved a static IP.

### Cost summary

| State | What you pay | Approx |
|---|---|---|
| Running | L4 GPU + VM + disk | **~$0.85/hr** |
| **Stopped** | 100 GB disk only (GPU billing OFF) | **~$10/month** |
| Deleted (teardown) | nothing (bucket only, if kept) | **~$0.30/month** for the 12 GB bucket |

While your $300 free credit lasts (90 days from signup), all of this comes out of the credit.

### Full teardown — ZERO compute charge (deletes the VM + disk)
Only do this when the project is fully done — recreating needs the full setup (§5).
```bash
gcloud compute instances delete farmervision --zone=us-east4-c -q     # VM + its disk
gcloud compute firewall-rules delete fv-gateway -q                    # only if you created it
# optional — also delete the uploaded models (you'd re-upload to redeploy):
# gcloud storage rm -r gs://farmervision-prod-artifacts
```

**Rule of thumb:** demoing again in the next few days → `stop_vm.sh` (keeps everything). Completely done → delete.

---

## 2. Check health + logs

SSH in:
```bash
gcloud compute ssh farmervision --zone=us-east4-c
```
On the VM:
```bash
cd ~/farmervision
API_KEY=$(grep -m1 '^API_KEY=' .env | cut -d= -f2-)      # load your key into the shell

curl -s localhost:8000/health | jq                        # should show gpu:true, points:723439
docker compose ps                                         # both containers "running"
docker compose logs -f gateway                            # live gateway logs (Ctrl+C to exit)
docker compose logs -f qdrant                             # vector DB logs
```
`/health` green looks like:
```json
{"status":"ok","gpu":true,"gpu_name":"NVIDIA L4","collection":"agri_knowledge",
 "points":723439,"modules":{"retrieval":true,"generation":true,"ieg_model":false,"vision":true}}
```
`ieg_model:false` is expected — the guardrail runs on rules + keyword intent (the DistilBERT class isn't wired). RAG, vision, and generation all work.

---

## 3. Expose for external use

> **This is already set up** — the gateway is bound to `0.0.0.0:8000` and the firewall rule
> `fv-gateway` exists. It stays that way across stop/start. This section is for re-checking,
> getting the current URL, and locking it down. **A teammate integrating only needs the URL +
> API key — hand them `API_SPEC.md`; they never touch GCP.**

### ⚠️ The one gotcha: firewall commands must run from **Cloud Shell**, not the VM
The VM's service account can't create/change firewall rules (`compute.firewalls.create` denied).
Always run `gcloud compute firewall-rules ...` from **Cloud Shell** (your own account).

### Get the current public URL (from Cloud Shell)
```bash
gcloud compute instances describe farmervision --zone=us-east4-c \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```
→ `http://<that-ip>:8000`. Your app / friend calls `/query`, `/classify`, `/vision`, `/diagnose`
with header `X-API-Key: <key>`.

### 📌 Pin the IP so the URL NEVER changes (do this for integration)
By default the IP changes on every start. Promote the **current** IP to static (run while the VM
is RUNNING, from Cloud Shell):
```bash
IP=$(gcloud compute instances describe farmervision --zone=us-east4-c --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
gcloud compute addresses create fv-ip --addresses=$IP --region=us-east4
```
Now `http://$IP:8000` is permanent across stop/start. (Costs ~$7/mo while the VM is stopped;
release with `gcloud compute addresses delete fv-ip --region=us-east4`.)

### How it was set up (only needed if you rebuild from scratch)
```bash
# on the VM: bind gateway to all interfaces
cd ~/farmervision && sed -i 's/127.0.0.1:8000:8000/8000:8000/' docker-compose.yml
docker compose --env-file runtime.env up -d gateway
# from CLOUD SHELL: open the firewall
gcloud compute firewall-rules create fv-gateway --allow=tcp:8000 --target-tags=farmervision --source-ranges=0.0.0.0/0
```

### Lock it down / turn it off
```bash
# restrict to one IP (from Cloud Shell):
gcloud compute firewall-rules update fv-gateway --source-ranges=YOUR_APP_IP/32
# close public access entirely:
gcloud compute firewall-rules delete fv-gateway
```

**Security:** with `0.0.0.0/0` the `X-API-Key` header is the ONLY protection — keep it secret.
It's HTTP (no TLS); fine for a prototype. For fully private access instead, skip the firewall and
tunnel: `gcloud compute ssh farmervision --zone=us-east4-c -- -N -L 8000:localhost:8000`, then call `http://localhost:8000`.

---

## 4. Where to change things

All files are on the VM in **`~/farmervision/`**. After any change, apply it as noted, then it's live.

| What you want to change | File / location | How to apply |
|---|---|---|
| **LLM system prompt** (behavior, tone, rules) | `app/generation.py` → `SYSTEM_RULES` (+ `OUTPUT_FORMAT`) | rebuild gateway* |
| **Reply-language behavior** (force/relax language matching) | `app/generation.py` → `detect_lang()`, `LANG_NAME`, the `### WRITE ... ###` line in `build_messages()` | rebuild gateway* |
| Localized abstain / KVK disclaimer text | `app/generation.py` → `ABSTAIN_MSG`, `KVK_DISCLAIMER` | rebuild gateway* |
| Multi-turn memory length | `app/pipeline.py` → `_MAX_MSGS` (default 8 messages) | rebuild gateway* |
| **Answer length** | `app/generation.py` → `generate(... max_new_tokens=256)` | rebuild gateway* |
| **Retrieval tiers** (grounded/fallback thresholds) | `/opt/farmervision/artifacts/qdrant/manifest.json` → `tiers` | restart gateway** |
| **Fusion weights** (pdf vs kcc per intent) | same `manifest.json` → `fusion_weights` | restart gateway** |
| Default top-k | same `manifest.json` → `top_k_default` | restart gateway** |
| **Guardrail rules** (blocked pesticides, non-agri words) | `app/ieg.py` → `_RESTRICTED`, `_NON_AGRI` | rebuild gateway* |
| External-data keyword hints (mandi/weather/yield) | `app/ieg.py` → `_HINT_PATTERNS` | rebuild gateway* |
| **API key** | `~/farmervision/runtime.env` → `API_KEY` | restart gateway** |
| Model files (merged LLM, IEG, vision) | `gs://.../` bucket → re-run `bash 04_vm_setup.sh` | pulls + restarts |

\* **rebuild gateway** (code changed):
```bash
cd ~/farmervision && docker compose --env-file runtime.env up -d --build gateway
```
\*\* **restart gateway** (config/data changed, no code):
```bash
cd ~/farmervision && docker compose --env-file runtime.env up -d --force-recreate gateway
```

Example — change the system prompt:
```bash
cd ~/farmervision
nano app/generation.py          # edit the SYSTEM_PROMPT string
docker compose --env-file runtime.env up -d --build gateway
```

---

## 5. Full setup from scratch

If you delete the VM and need to recreate it. Everything runs from **Cloud Shell** in `~/requiredforgcp` (models must already be in the bucket).

```bash
# 0. one-time: load your SSH key so it stops asking for the passphrase
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/google_compute_engine

# 1. create the GPU VM (sweeps zones for stock; lands wherever L4/T4 is free)
bash 03_create_vm.sh
#    note the zone it prints (">> CREATED in <zone>"), then:
Z=$(gcloud compute instances list --filter="name=farmervision" --format="value(zone)"); Z=${Z##*/}
sed -i "s/^ZONE=.*/ZONE=$Z/" .env

# 2. copy .env to the VM (scp skips dotfiles, so do it explicitly)
gcloud compute scp .env farmervision:~/farmervision/ --zone=$Z

# 3. SSH in and finish
gcloud compute ssh farmervision --zone=$Z
```
Then **on the VM** (`~/farmervision`):
```bash
# Docker is NOT preinstalled on this image — install it + NVIDIA runtime
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
sudo apt-get install -y python3-requests          # for the snapshot restore step

newgrp docker                                     # activate docker in this shell
cd ~/farmervision && bash 04_vm_setup.sh          # pulls models, restores DB, builds gateway (~10-15 min)
```
`04` ends with `>> Health:` and green JSON. The `generation.py` chat-template fix is already in your files.

---

## 6. API reference

Base URL: `http://<host>:8000` (localhost via tunnel, or the public IP).
**Auth:** every `POST` needs header `X-API-Key: <your API_KEY>`. `GET /health` needs no key.

### GET `/health`
Liveness + what's loaded. No body.
```json
{"status":"ok","gpu":true,"gpu_name":"NVIDIA L4","collection":"agri_knowledge",
 "points":723439,"modules":{"retrieval":true,"generation":true,"ieg_model":false,"vision":true},
 "note":"...","errors":[]}
```

### POST `/classify`
Intent + entities + guardrail + which external data the query needs. Use it to decide what to fetch before `/query`.
- **Request:** `{"query": "<text>"}`
- **Response:**
```json
{"intent":"disease_pest","retrieval_intent":"field_practice","blocked":false,
 "block_reason":null,"entities":{},"guardrail_backend":"rules-only",
 "suggested_external":["mandi_prices"]}
```

### POST `/query`  (the main RAG endpoint)
- **Request body:**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `query` | string | yes | the farmer's question (any language) |
| `intent` | string | no | fusion weighting: `policy` \| `field_practice` \| `general` (omit → auto) |
| `top_k` | int | no | how many chunks to retrieve (default 5) |
| `filters` | object | no | narrow the search (see §7) |
| `live_data` | object | no | facts your app fetched (mandi/weather/yield) to inject |
| `skip_retrieval` | bool | no | `true` = answer from `live_data` only, no vector search |
| `session_id` | string | no | **multi-turn:** gateway remembers this conversation |
| `history` | array | no | **multi-turn:** you send prior turns `[{"role":"user"/"assistant","content":"..."}]` |

- **Multi-turn:** pass **either** `session_id` (gateway keeps the history in memory) **or** `history` (you keep it). The response echoes `session_id` and the updated `history` so the client can persist and resend. Follow-ups ("its dose?", "wahi") are resolved from history + contextualized retrieval.
- **Response (answered):**
```json
{"tier":"grounded","blocked":false,"answer":"...cited answer...\nSources: [1], [2]",
 "sources":[{"n":1,"score":0.71,"source_type":"kcc","citation":{...}}],
 "live_data_used":["mandi_prices"],"top_score":0.71,"intent":"field_practice","lang":"hinglish",
 "guardrail_backend":"rules-only","gen_ms":11840,"out_tokens":96,
 "session_id":"demo1","history":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
 "latency_ms":12010,"suggested_external":[]}
```
- **Response (blocked by guardrail):** `{"tier":"blocked","blocked":true,"block_reason":"non_agricultural","answer":"<refusal>",...}`
- **Response (nothing relevant + no live_data):** `{"tier":"abstain_out_of_scope","answer":null,"message":"<localized abstain msg>",...}` (`message` is in the query's language)

> 📤 **Sharing the API?** `API_SPEC.md` (same folder) is a standalone, GCP-free version of this
> section — hand it to anyone integrating; they only need the URL + API key from you.

### POST `/vision`  (image → disease)
- **Request:** multipart form, field `file` = image (jpg/png).
- **Response:**
```json
{"label":"wheat__yellow_rust","crop":"wheat","disease":"yellow rust",
 "confidence":0.94,"top_k":[{"label":"wheat__yellow_rust","prob":0.94}, ...],
 "note":"Lab-trained model; treat as a suggestion, confirm with a local expert."}
```

### POST `/diagnose`  (image → disease + grounded treatment)
- **Request:** multipart form, `file` = image, optional `question` = text.
- **Response:**
```json
{"diagnosis":{"label":"wheat__yellow_rust","crop":"wheat","disease":"yellow rust","confidence":0.94,...},
 "tier":"grounded","answer":"...cited treatment...",
 "sources":[{"n":1,"score":0.68,"source_type":"pdf","citation":{...}}],
 "gen_ms":12000,"out_tokens":88,"latency_ms":12500}
```

---

## 7. Parameter values

**`intent`** (fusion weighting on `/query`):

| value | weights (pdf : kcc) | use for |
|---|---|---|
| `policy` | 2.0 : 0.5 | schemes, subsidies, eligibility, government guidelines |
| `field_practice` | 0.5 : 2.0 | how-to, dosage, disease/pest, cultivation |
| `general` | 1.0 : 1.0 | anything else / unsure |

**`filters`** object (all optional):

| key | allowed values |
|---|---|
| `source_type` | `pdf` \| `kcc` |
| `doc_category` | `scheme_eligibility` \| `crop_advisory` \| `contingency_plan` \| `policy_guideline` (pdf only) |
| `query_type` | KCC category string, e.g. `Plant Protection` (kcc only) |
| `crop` | canonical crop, e.g. `rice`, `wheat` |
| `district` | canonical UP district, e.g. `Varanasi` |
| `season` | `Rabi` \| `Kharif` \| `Zaid` (kcc only) |
| `language` | `en` \| `hi` \| `mixed` |
| `year_from` | integer (only content from this year onward) |
| `only_tables` | `true` (pdf dosage/scheme tables only) |

**`live_data`** object — free-form keys; these get friendly labels: `mandi_prices`, `market`, `weather`, `forecast`, `yield`, `yield_prediction`. Values may be strings or nested JSON. Example:
`{"mandi_prices":"Wheat @ Varanasi Rs 2450/qtl","weather":{"rain_3d":"none","temp_c":[28,33]},"yield":"2.6 t/ha"}`

**`tier`** values in responses: `grounded` (score ≥ 0.638) · `fallback_with_disclaimer` (≥ 0.553) · `abstain_out_of_scope` (< 0.553) · `blocked` (guardrail) · `skipped` (skip_retrieval).

**`suggested_external`** / IEG hint values: `mandi_prices`, `weather`, `yield`.

**IEG `intent`** classes (from `/classify`): `cultivation_practice`, `disease_pest`, `nutrition_fertilizer`, `post_harvest_storage`, `specialty_other`, `general`, `non_agri`.

**Vision `label`** — 20 classes (`crop__disease`):
`rice__blast`, `rice__bacterial_blight`, `rice__brown_spot`, `rice__tungro`, `rice__leaf_smut`,
`wheat__healthy`, `wheat__yellow_rust`, `wheat__brown_rust`, `wheat__black_rust`, `wheat__blast`,
`wheat__septoria`, `wheat__mildew`, `wheat__aphid`, `wheat__mite`, `wheat__stem_fly`,
`wheat__smut`, `wheat__tan_spot`, `wheat__leaf_blight`, `wheat__common_root_rot`, `wheat__fusarium_head_blight`.

---

## 8. Sample requests

Set these first (on the VM, or use your public IP/tunnel):
```bash
URL=http://localhost:8000
KEY=$(grep -m1 '^API_KEY=' ~/farmervision/.env | cut -d= -f2-)
```

```bash
# 1. health
curl -s $URL/health | jq

# 2. classify (decide what to fetch) — returns intent + suggested_external
curl -s -X POST $URL/classify -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"aaj mandi me gehu ka bhav kya hai"}' | jq

# 3. RAG — English, field practice
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"how do I control yellow rust in wheat","intent":"field_practice"}' | jq

# 4. RAG — Hindi
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"गेहूं में पीला रतुआ के लिए कौन सी दवा डालें","intent":"field_practice"}' | jq

# 5. RAG — Hinglish
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"paddy me brown spot ke liye kya spray karun","intent":"field_practice"}' | jq

# 6. RAG — policy / scheme question
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"PM Kisan scheme eligibility kya hai","intent":"policy"}' | jq

# 7. RAG — with filters (wheat, Varanasi, KCC only)
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"urea dose","intent":"field_practice","filters":{"crop":"wheat","source_type":"kcc"}}' | jq

# 8. RAG — with injected live data (mandi + weather your app fetched)
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"gehu abhi bech du ya rukun","live_data":{"mandi_prices":"Wheat @ Varanasi Rs 2480/qtl (2026-08-14)","weather":"dry, no rain 3 days"}}' | jq

# 9. Pure live-data question (skip vector search)
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"kal barish hogi kya","skip_retrieval":true,"live_data":{"weather":"70% chance of rain tomorrow, 24-29C"}}' | jq

# 10. Guardrail — should be blocked (non-agri / restricted)
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"where can I buy monocrotophos for my wheat"}' | jq

# 11. Abstain — nothing relevant, no live data
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"how to fix my motorcycle engine"}' | jq

# 12. Vision — image only
curl -s -X POST $URL/vision -H "X-API-Key: $KEY" -F "file=@leaf.jpg" | jq

# 13. Diagnose — image + question → disease + grounded advice
curl -s -X POST $URL/diagnose -H "X-API-Key: $KEY" -F "file=@leaf.jpg" -F "question=is ke liye kya spray karun" | jq

# 14. MULTI-TURN — turn 1, then a follow-up (gateway remembers via session_id)
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"wheat me yellow rust ki dawa batao","session_id":"chat1","intent":"field_practice"}' | jq -r '.answer'
curl -s -X POST $URL/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"iski dose kitni honi chahiye","session_id":"chat1","intent":"field_practice"}' | jq -r '.answer'
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: command not found` after recreate | This image ships without Docker — run the install block in §5. |
| snapshot restore fails / `No module named requests` | `sudo apt-get install -y python3-requests`, then re-run `04`. |
| `/query` 500, `KeyError: 'shape'` | Chat-template returns a dict — the fix is already in `app/generation.py`; rebuild gateway. |
| `.env not found` when running `04` | `scp` skips dotfiles — copy it: `gcloud compute scp .env farmervision:~/farmervision/ --zone=us-east4-c`. |
| VM create: `STOCKOUT` / `ZONE_RESOURCE_POOL_EXHAUSTED` | No GPU stock in that zone — `03_create_vm.sh` sweeps zones; or wait 15–30 min. |
| VM create: `QUOTA_EXCEEDED` | Request GPU quota (IAM & Admin → Quotas → "GPUs (all regions)" / "NVIDIA L4 GPUs"). |
| gateway not answering | `docker compose logs -f gateway`; ensure `/health` shows `gpu:true`. |
| forgot the API key | `grep API_KEY ~/farmervision/runtime.env` on the VM. |
| repeated SSH passphrase prompts | `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/google_compute_engine` once per Cloud Shell session. |

**Always end a session with `bash stop_vm.sh`.**
