# Deploy Entirely From GCP — Using Google Cloud Shell (No Laptop)

This is the same deployment as `GCP_DEPLOYMENT_PLAN.md`, but every command runs
in **Google Cloud Shell** (the free browser terminal in the GCP console). You
install nothing on your own machine.

The deployment folder includes `.env.example`, but never share your completed
`.env`. The VM creation script copies your private `.env` to the VM separately.
Shell scripts should be uploaded with Unix (LF) line endings; Cloud Shell can
then run them directly.

---

## The mental model

Three things, and Cloud Shell is only the first one:

| Piece | What it is | Cost |
|---|---|---|
| **Cloud Shell** | your control terminal (replaces the laptop). `gcloud` is preinstalled and already logged in. | free |
| **GCS bucket** | holds your model files (snapshot, adapter, etc.) | pennies |
| **GPU VM** | does the actual work: Qdrant + models + the API | the only real cost — **stop it when idle** |

Cloud Shell just *drives* GCP. The heavy lifting is on the VM.

### Cloud Shell limits to know
- **5 GB** home dir — do **not** try to store your 3.8 GB snapshot here. (We don't.)
- **Times out after ~20 min idle** and the session VM is wiped (your home files survive). So don't start something long and walk away — see the `tmux` tip in Step 7.

---

## ✅ Things YOU add manually (do these yourself — nobody can script them)

1. **Enable billing** on your GCP project (Console → Billing).
2. **Request GPU quota** — Console → *IAM & Admin → Quotas* → filter **"GPUs (all regions)"** → request **1**. **Blocks everything until granted (hours).** Do it first.
3. **Set a budget alert** — Console → *Billing → Budgets & alerts* → e.g. **$50**.
4. **Hugging Face** — accept the Gemma licence at <https://huggingface.co/google/gemma-3-4b-it> and make a token at <https://huggingface.co/settings/tokens>. Revoke any token exposed in an older copy of this package.
5. **Colab GPU access** — your Colab runtime must have GPU enabled to **merge the LoRA adapter into Gemma** (Step 5, cell 4). This is a one-time cost; subsequent inference runs on the VM's GPU without re-merging.
6. **Fill `.env`** (Step 3 below) with `PROJECT_ID`, `ZONE`, `VM_NAME`, `BUCKET`, `HF_TOKEN`, `API_KEY`.
7. **RAG snapshot + manifest** from notebook 08b must be in Google Drive (cell 3 will pull it).
8. **Generator LoRA adapter** from your distillation training must be in Google Drive (cell 4 will merge it).
9. **Vision + IEG models** — the Kaggle datasets will be auto-downloaded in Step 5, cell 5.
10. **One config file**: `vision/label_to_idx.json` must be saved with your training output (Step 5, cell 5 handles this).

---

## Step 1 — Open Cloud Shell
Go to <https://console.cloud.google.com>, pick your project (top bar), and click the
**terminal icon** (top-right, "Activate Cloud Shell"). Wait for the prompt.

## Step 2 — Get the deploy files into Cloud Shell
The scripts live in your repo under `docs/internal/do_not_open/requiredforgcp`.

```bash
git clone <YOUR-REPO-URL> fv && cd fv/docs/internal/do_not_open/requiredforgcp
```

> No GitHub remote? Then click the **⋮ (three dots) → Upload** in the Cloud Shell
> toolbar and upload the small `requiredforgcp` folder (it's only a few KB), then
> `cd` into it.

## Step 3 — Fill in your settings
```bash
cp .env.example .env
cloudshell edit .env          # opens the built-in editor; fill every value, save
```
Set `PROJECT_ID`, `ZONE`, `VM_NAME`, `BUCKET` (globally unique), `HF_TOKEN`, `API_KEY`.

## Step 4 — Project setup (APIs)
```bash
bash 01_gcloud_setup.sh
```
Then make sure you've done the **GPU quota** and **budget** from the manual checklist above.

## Step 5 — Upload models to the bucket (from Colab)
Your files are in Drive/Kaggle, not in Cloud Shell. **Do this from Colab** (files are already there, uploads at Google's speed).

### ⚠️ **Switch to GPU runtime for this step** — cell 4 needs a GPU to merge the adapter.

Open a Colab notebook and paste cells `1a`, `1b`, `2`, `3`, `4`, `5`, and `6` from `upload_artifacts_from_colab.py` **in order**:
1. **Cells 1a/1b** — Sign into GCP and Kaggle.
2. **Cell 2** — Set `PROJECT`, `BUCKET`, `HF_TOKEN`, and both shareable Drive links (matching `.env`).
3. **Cell 3** — Upload RAG snapshot from Drive to `$BUCKET/qdrant/`; it requires `manifest.json` at the folder's top level.
4. **Cell 4 (requires GPU)** — Merge the LoRA adapter into Gemma-3-4B and upload it to `$BUCKET/generator/merged/`. The VM now loads this local merged model when `config.json` is present.
   - Cell 4 automatically upgrades dependencies (peft, torchao) to compatible versions.
   - If you see import errors about `torchao` or `peft`, restart the kernel and re-run cell 4.
5. **Cell 5** — Download + upload IEG + Vision models from Kaggle to `$BUCKET/ieg/` and `$BUCKET/vision/`.
6. **Cell 6** — Verify the final bucket structure.

Edit the Drive paths in cell 3 to point to where notebook 08b saved your RAG snapshot.

Verify the full structure uploaded from Cloud Shell:
```bash
source config.sh
gcloud storage ls -r "$BUCKET"
```
Expected structure:
```
$BUCKET/
  qdrant/
    manifest.json
    snapshot_state.json    (or snapshot_*)
  generator/
    merged/
      config.json
      model.safetensors    (merged adapter already baked in)
      tokenizer.json
      tokenizer_config.json
  ieg/
    intent_entity_guardrail_model.pt
    label_maps.json
  vision/
    p3_full_best.pt
    label_to_idx.json
```

**Troubleshooting**:
- **Cell 4 import errors** (torchao, peft): Cell 4 auto-upgrades these. If you see an error, restart the kernel (Runtime → Restart runtime) and re-run cell 4.
- If `generator/merged/` is missing: the gateway loads base Gemma only, so the trained adapter is not applied. Rerun cell 4 before relying on the distilled model.
- If cell 4 runs out of memory: reduce `torch_dtype` or batch size, or use a larger GPU.
- Cell 5 may take a few min to download the Kaggle datasets — this is normal.

## Step 6 — Create the GPU VM
```bash
bash 03_create_vm.sh
```
Runs from Cloud Shell exactly like from a laptop: it creates the VM and copies the
deploy folder onto it. (Keep the Cloud Shell tab active while it waits for SSH.)

## Step 7 — Set up + start everything ON the VM
SSH in from Cloud Shell:
```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE"
```
On the VM, run inside **tmux** so a Cloud Shell disconnect can't kill the build:
```bash
tmux new -s deploy
cd ~/farmervision && bash 04_vm_setup.sh
```
(If you get disconnected: SSH back in, then `tmux attach -t deploy` to reattach.)

This does the following (all automatic):
1. Pulls all models from `$BUCKET` into `/opt/farmervision/artifacts/`.
2. Starts Qdrant (vector DB) and restores your snapshot.
3. Builds + starts the FastAPI gateway container.
4. **Loads the merged Gemma model** from `generator/merged/`. If it is absent, it loads base Gemma only.

**First run: ~10–15 min** (Gemma download + build). Subsequent restarts: ~2 min (cached).

## Step 8 — Test (from Cloud Shell)
Check if the gateway is ready:
```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="curl -s localhost:8000/health" | jq
```
Look for:
```json
{
  "status": "ok",
  "gpu": true,
  "modules": { "generation": true, ... }
}
```

For a real query, open an SSH tunnel and test:
```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- -N -L 8000:localhost:8000 &
curl -s -X POST localhost:8000/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"wheat yellow rust dawa","intent":"field_practice"}' | jq
```

**Check the logs if /health fails**:
```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="docker compose logs -f gateway"
```

## Step 9 — Let your main app reach the API
Same as the main plan: your app opens the SSH tunnel (Option A) or you open port 8000
to your app's IP only (`bash expose_to_my_ip.sh`). The `X-API-Key` header is always
required. The endpoints are `/classify`, `/query` (with optional `live_data`), `/diagnose`, `/vision`, `/health`.

## Step 10 — STOP THE VM (every time you're done)
Always stop when not in use to avoid GPU charges:
```bash
bash stop_vm.sh      # stops GPU billing; all models + state stay on disk
bash start_vm.sh     # ~2 min to warm up; containers auto-restart
```
**Cost**: ~$0.25/hour GPU running, $0/hour stopped.

---

## Why merged adapters? (The performance story)

The generator uses **Gemma-3-4B (4-bit quantized)** with a **LoRA adapter** trained for your domain. There are two ways to serve this:

| Approach | Step | Runtime cost | Inference speed |
|----------|------|--------------|-----------------|
| **Merged** (recommended) | Cell 4 merges adapter into weights → upload merged model | $0 extra | Fast (~baseline) |
| **Merged model missing** | Skip cell 4; gateway loads base Gemma only | $0 extra | Trained adapter is not applied |

**Cell 4 does the merge once in Colab**, so the VM just loads the merged weights and runs fast. No performance penalty at serving time.

---

## Cloud Shell tips
- **Keep the tab active** during VM creation and the build, or use `tmux` **on the VM**
  (Step 7) so long jobs survive a Cloud Shell timeout.
- Your `.env` and cloned repo persist in Cloud Shell's home dir across sessions (5 GB).
- If Cloud Shell says "session ended", just reopen it and `cd fv/.../requiredforgcp` again —
  the VM and bucket are unaffected.
- **Logs**: `docker compose logs -f gateway` (on VM) shows real-time inference traces and any load errors.
- Everything else (troubleshooting, costs, file details, architecture) is in
  `GCP_DEPLOYMENT_PLAN.md`.

---

## Quick reference — the whole thing
```bash
# === IN CLOUD SHELL ===
git clone <repo> fv && cd fv/docs/internal/do_not_open/requiredforgcp
cp .env.example .env && cloudshell edit .env      # fill it in
bash 01_gcloud_setup.sh                            # enable APIs

# === IN COLAB (GPU runtime!) ===
# Paste cells 1a, 1b, 2, 3, 4, 5, and 6 from upload_artifacts_from_colab.py
# Cell 4 merges the adapter (critical for fast inference)

# === BACK IN CLOUD SHELL ===
bash 03_create_vm.sh
gcloud compute ssh "$VM_NAME" --zone="$ZONE"

# === ON THE VM ===
tmux new -s deploy
cd ~/farmervision && bash 04_vm_setup.sh
# (wait ~10–15 min for first build; check logs if it stalls)

# === BACK IN CLOUD SHELL ===
curl -s localhost:8000/health | jq   # confirm gateway is ready
bash stop_vm.sh                        # stop when done
```
