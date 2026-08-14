# FarmerVision — Deploy Everything on ONE GCP VM (Simple Step-by-Step)

**Goal:** Put the whole inference stack — Qdrant vector DB + distilled Gemma generator + intent/entity/guardrail (and optional vision + yield) — on a **single GPU VM**, behind **one API** your main app calls. Fast inference, minimum cost, written so anyone on the team can follow it.

**Read this first, then run the scripts in `requiredforgcp/` in order.**

> 💻 **Doing it all from GCP with no laptop?** Follow **`RUN_FROM_CLOUD_SHELL.md`**
> instead — same deployment, every command in the browser (Google Cloud Shell), and
> it uploads your models from Colab instead of a local `./artifacts` folder. This
> file is still the reference for costs, the API, and troubleshooting.

---

## 0. What you are building (the whole picture)

```
                YOUR MAIN APP (web / mobile / whatever)
                              |
                              |  HTTPS, one API key
                              v
        ┌─────────────────────────────────────────────┐
        │   ONE GCP VM  (n1-standard-8 + 1x T4 GPU)     │
        │                                               │
        │   Gateway API  (FastAPI, port 8000) ──────┐   │
        │     • guardrail + intent  (CPU)           │   │
        │     • query embedding BGE-M3 (CPU,        │   │
        │        FastEmbed ONNX — fast, no GPU)      │   │
        │     • Gemma-3-4B generator  (GPU, 4-bit)  │   │
        │     • vision (optional, CPU)              │   │
        │                    |                       │   │
        │                    v                       │   │
        │   Qdrant server  (Docker, port 6333) ◄─────┘   │
        │     • agri_knowledge, 723k vectors, HNSW      │
        └─────────────────────────────────────────────┘
```

**Scope of this gateway:** it serves **vision + intent/entity/guardrail + RAG (vector DB + generation)** only. **Mandi prices, weather, and yield prediction are done by YOUR main app** and injected into `/query` as `live_data`, which the LLM weaves into the grounded answer.

**The one idea that makes this cheap and fast:** only the **LLM needs the GPU**. Everything else (Qdrant, embedding, guardrail, vision) runs on CPU. This is the split your Milestone-5 §13 eval used, now turned into a real service.

**What the app sends:** a question (text) — plus, optionally, `live_data` (mandi prices / weather / a yield number your app computed) — and gets back a grounded, cited answer + a tier (`grounded` / `fallback` / `abstain`).

---

## 1. Cost — read this or you will burn your free credits

Google gives new users **$300 free credit for 90 days**. A T4 GPU VM costs roughly **$0.35–0.55/hour** (verify current price for your region). So:

| How you use it | Rough cost |
|---|---|
| Left ON 24/7 for a month | **~$300 (your whole credit in ~3–4 weeks)** ❌ |
| Turned ON only when demoing (~2 hrs/day) | **~$25–35/month** ✅ |
| Stopped (disk only, ~150 GB) | **~$6/month** ✅ |

### 🟡 THE GOLDEN RULE: **STOP THE VM WHEN YOU ARE NOT USING IT.**
You pay for the GPU **only while the VM is running**. Stopping it keeps all your data and models on disk for a few dollars a month. Use `stop_vm.sh` every single time you finish. Starting it again takes ~2 minutes.

We also set a **budget alert** (Step 2.4) so Google emails you at 50% / 90% / 100% of a limit you choose.

---

## 2. One-time setup on YOUR computer (do this once)

You need: a Google account with billing enabled, and the artifacts listed in Step 3.

### 2.1 Install the Google Cloud CLI
Download from <https://cloud.google.com/sdk/docs/install>. Then log in:

```bash
gcloud auth login
```

### 2.2 Create / pick a project and set it
```bash
gcloud projects create farmervision-prod --name="FarmerVision"   # or reuse an existing project id
gcloud config set project farmervision-prod
```

### 2.3 Turn on the services we use
```bash
gcloud services enable compute.googleapis.com storage.googleapis.com
```

### 2.4 Set a budget alert (do NOT skip)
Console → **Billing → Budgets & alerts → Create budget** → set amount (e.g. **$50**) → alerts at 50/90/100%. This is the safety net.

### 2.5 ⚠️ Request GPU quota — DO THIS NOW, it takes hours to approve
New projects have **0 GPU quota**, so VM creation will fail until this is granted.
Console → **IAM & Admin → Quotas** → filter for **"GPUs (all regions)"** → select it → **Edit Quotas** → request **1** → submit. Wait for the approval email (usually a few hours, sometimes a day).

> While you wait for quota, you can still do Step 3 (upload artifacts).

---

## 3. Collect your model artifacts (the things the VM needs)

Put all of these into one local folder, e.g. `./artifacts/`. Get them from your Kaggle/Colab/Drive outputs (see Milestone-5 Appendix E for exact paths).

```
artifacts/
├── qdrant/
│   ├── agri_knowledge-XXXX.snapshot     # the 3.8 GB vector DB snapshot
│   └── manifest.json                    # query-side contract (tiers, dims, weights)
├── ieg/
│   ├── intent_entity_guardrail_model.pt # DistilBERT multilingual, 3 heads (514 MB)
│   └── label_maps.json                  # intent + entity + guardrail label maps
├── generator/
│   └── merged/                          # base Gemma with the LoRA adapter MERGED in (fast inference)
└── vision/                              # for /vision + /diagnose
    ├── p3_full_best.pt                  # ViT-S/16 checkpoint
    └── label_to_idx.json                # class -> index map
```

> **Note:** there is no `yield/` folder — mandi prices, weather, and yield are handled by your main app and injected via `live_data` (see §8).

**How the bucket gets filled** (via `upload_artifacts_from_colab.py`, run on a GPU Colab):
- **IEG + Vision** — downloaded with **kagglehub**, the same datasets `scripts/run_e2e_eval.py` uses (`aneeqasiddiqui377/v4-output`, `iitm21f1003346/vits16-crop-disease`).
- **RAG snapshot + `manifest.json`** — from your **Google Drive**, as-is.
- **Generator (`generator/merged/`)** — the distilled **LoRA adapter merged into base Gemma** so there's no adapter overhead at inference. If a merged model is absent, the gateway falls back to base + `best_adapter/`.

**One config file that must be present:** `vision/label_to_idx.json` (the class→index map).

**Note on Gemma (the base model):** `gemma-3-4b-it` is **gated** on Hugging Face. You do NOT upload it — the VM downloads it at startup using your HF token. So:
1. Accept the licence at <https://huggingface.co/google/gemma-3-4b-it> (once, with your HF account).
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Put it in your `.env` file as `HF_TOKEN=hf_xxx` (Step 4).

---

## 4. Configure the scripts

In `requiredforgcp/`, copy `.env.example` to `.env` and fill it in:

```bash
cp requiredforgcp/.env.example requiredforgcp/.env
```

Edit `.env`:
```
PROJECT_ID=farmervision-prod
ZONE=us-central1-a
VM_NAME=farmervision
BUCKET=gs://farmervision-prod-artifacts     # must be globally unique
HF_TOKEN=hf_your_token_here
API_KEY=pick-a-long-random-string           # your app will send this to the gateway
```

`config.sh` reads this `.env` so every script uses the same names.

---

## 5. Upload your models to a bucket (run on your computer)

```bash
bash requiredforgcp/02_upload_artifacts.sh
```
**What it does:** creates the storage bucket and uploads your `./artifacts/` folder to it. The VM will pull from here (fast, inside Google's network, and you never re-upload).

---

## 6. Create the GPU VM (run on your computer)

```bash
bash requiredforgcp/03_create_vm.sh
```
**What it does:** creates the VM using Google's **Deep Learning image** (NVIDIA driver + Docker already installed — saves you an hour of setup), attaches 1× T4 GPU, opens nothing to the public internet, and tags it so we can reach it safely over SSH.

> 💰 **Cheaper option — Spot VM:** open `03_create_vm.sh` and uncomment the two `SPOT` lines. Spot VMs cost ~60–70% less but Google can shut them down if it needs the capacity. Fine for dev/demos; not for a live exam demo you can't afford to have interrupted.

---

## 7. Set up and start everything (run ON the VM)

SSH into the VM:
```bash
gcloud compute ssh $VM_NAME --zone=$ZONE
```

The first login copies over the deploy files. Then, on the VM, run:
```bash
cd ~/farmervision && bash 04_vm_setup.sh
```
**What it does, in plain steps:**
1. Pulls your artifacts from the bucket to `/opt/farmervision/artifacts`.
2. Starts the **Qdrant** container.
3. **Restores** your 3.8 GB snapshot into Qdrant (uploads it through Qdrant's API — same method your notebook used).
4. Builds and starts the **Gateway** container (loads Gemma on the GPU, everything else on CPU).
5. Waits until `/health` is green.

First run takes ~10–15 min (downloading Gemma + building the image). Later starts are ~2 min.

---

## 8. Test it

Still on the VM (or from your laptop after Step 9):
```bash
curl -s localhost:8000/health | jq
```
Then try each endpoint your app can use:

```bash
# 1) Text question (RAG)
curl -s -X POST localhost:8000/query \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"query":"wheat me yellow rust ke liye kaunsi dawa use karein","intent":"field_practice"}' | jq

# 2) Text question WITH injected live data (mandi price + yield your app fetched)
curl -s -X POST localhost:8000/query \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"query":"gehu ka abhi bech du ya rukun? meri paidawar kitni hogi",
       "live_data":{"mandi_prices":"Wheat @ Varanasi mandi: Rs 2450/quintal (2026-08-13)",
                    "yield":"estimated 2.6 t/ha for your plot"}}' | jq

# 3) Classify only (your app uses suggested_external to decide what to fetch)
curl -s -X POST localhost:8000/classify \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"query":"aaj mandi me pyaaz ka bhav kya hai"}' | jq   # -> suggested_external: ["mandi_prices"]

# 4) Photo -> disease + grounded treatment advice
curl -s -X POST localhost:8000/diagnose \
  -H "X-API-Key: $API_KEY" -F "file=@leaf.jpg" -F "question=is ke liye kya spray karun" | jq

# 5) Photo -> disease label only (no LLM)
curl -s -X POST localhost:8000/vision -H "X-API-Key: $API_KEY" -F "file=@leaf.jpg" | jq
```

`/query` and `/diagnose` return an `answer`, a `tier`, the retrieved `sources` with citations, and timing. If `tier` is `abstain_out_of_scope` **and** no `live_data` was sent, retrieval found nothing relevant. When you inject `live_data`, the answer always uses those exact values.

### The endpoints your main app has

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health` | — | GPU, point count, which modules loaded |
| POST | `/classify` | JSON `{query}` | intent + entities + guardrail + `suggested_external` |
| POST | `/query` | JSON `{query, intent?, filters?, live_data?, skip_retrieval?}` | grounded cited answer + tier |
| POST | `/diagnose` | image file + optional `question` | disease + grounded treatment advice |
| POST | `/vision` | image file | disease label + confidence + top-5 |

**Injecting mandi/weather/yield:** your app fetches them (its own APIs + your yield model) and passes them in `live_data` as a JSON object — keys like `mandi_prices`, `weather`, `yield`, values as strings or nested JSON. The LLM treats them as authoritative and current. Use `/classify`'s `suggested_external` to know which to fetch, and `skip_retrieval:true` for pure weather/mandi questions that need no advisory lookup.

---

## 9. Let your main app reach the API (safely)

**Do NOT open port 8000 to the whole internet.** Two safe choices:

**Option A (simplest & safest) — SSH tunnel.** On the machine running your app:
```bash
gcloud compute ssh $VM_NAME --zone=$ZONE -- -N -L 8000:localhost:8000
```
Now your app calls `http://localhost:8000/query` and it is securely forwarded to the VM. Nothing is public.

**Option B — firewall to your IP only.** If your app runs elsewhere and needs direct access, run `expose_to_my_ip.sh` (opens 8000 **only** to your current public IP). Still protected by the `X-API-Key` header.

Your app just does an HTTP POST to `/query` with the `X-API-Key` header — that's the single API you asked for.

---

## 10. When you finish: STOP THE VM

```bash
bash requiredforgcp/stop_vm.sh      # stops billing for the GPU; keeps everything on disk
bash requiredforgcp/start_vm.sh     # bring it back in ~2 min next time
```
To delete everything permanently (VM + disk) when the project is over:
```bash
gcloud compute instances delete $VM_NAME --zone=$ZONE
gcloud storage rm -r $BUCKET        # optional: also delete uploaded models
```

---

## 11. Why this is fast (what we did for speed)

- **LLM on GPU in 4-bit with the adapter MERGED in** — folding the LoRA into the base weights removes the per-step adapter overhead (the exact latency issue flagged in Milestone-5 §12/§15). ~11 s/answer instead of minutes on CPU.
- **Query embedding uses FastEmbed** (BGE-M3 as an optimized ONNX model on CPU). Your Milestone-5 §13.5 showed embedding — not the vector search — was the 7.5 s bottleneck. FastEmbed cuts that a lot **without a GPU**.
- **All models load once at startup**, never per request.
- **Qdrant runs in server mode with HNSW** — vector search is tens of milliseconds (never the slow `path=` local mode).
- **Everything is on one machine**, so there are no network hops between components.

**Optional bigger speed-ups later** (not needed to start): serve the LLM with **vLLM** for higher throughput; enable **int8 quantization** in Qdrant; keep Qdrant vectors **in RAM** (the VM has enough) instead of on-disk.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| VM create fails: "quota exceeded" | GPU quota not granted yet (Step 2.5). Wait for the email. |
| VM create fails: "no capacity" in zone | Try another zone (`us-central1-b`, `us-west1-b`) in `.env`, or wait. |
| Gemma download 401 | HF token missing/invalid, or licence not accepted. Redo Step 3 note. |
| `/health` shows `gpu: false` | Driver still installing on first boot — wait 3–4 min and retry, or `nvidia-smi` to check. |
| Qdrant restore says 0 points | Wrong snapshot filename in `manifest.json`, or snapshot didn't upload. Re-check the bucket. |
| Everything slow (~25 s search) | Qdrant not in server mode / snapshot restored into local mode. Confirm the gateway talks to `http://qdrant:6333`. |

---

## 13. What each file in `requiredforgcp/` is

| File | Runs where | What it does |
|---|---|---|
| `.env.example` / `config.sh` | your PC | your project names + secrets, shared by all scripts |
| `01_gcloud_setup.sh` | your PC | enable APIs, print the GPU-quota reminder |
| `02_upload_artifacts.sh` | your PC | create bucket, upload `./artifacts/` |
| `03_create_vm.sh` | your PC | create the GPU VM (+ optional Spot) |
| `04_vm_setup.sh` | the VM | pull artifacts, start Qdrant, restore snapshot, start gateway |
| `restore_qdrant.py` | the VM | upload the snapshot into Qdrant + verify point count |
| `docker-compose.yml` | the VM | defines the Qdrant + Gateway containers |
| `Dockerfile` | the VM | builds the Gateway image |
| `requirements.txt` | the VM | Python dependencies for the gateway |
| `app/` | the VM | the FastAPI gateway (retrieval, generation, guardrail, pipeline) |
| `start_vm.sh` / `stop_vm.sh` | your PC | start/stop billing |
| `expose_to_my_ip.sh` | your PC | (optional) open port 8000 to your IP only |

Config values baked in from your `manifest.json`: collection `agri_knowledge`, BGE-M3 @ 1024-dim, COSINE, tiers **fallback 0.553 / grounded 0.638**, top-k 5, fusion weights (policy pdf 2.0/kcc 0.5, field_practice pdf 0.5/kcc 2.0, general 1/1). The gateway reads these from the manifest at runtime, so they stay correct automatically.
