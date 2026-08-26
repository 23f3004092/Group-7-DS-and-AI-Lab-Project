#!/usr/bin/env bash
# STEP 4 (run ON THE VM, inside ~/farmervision): pull models, start Qdrant,
# restore the snapshot, build + start the gateway. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

ART=/opt/farmervision/artifacts
echo ">> Preparing $ART"
sudo mkdir -p "$ART"
sudo chown -R "$USER" "$ART"

echo ">> Pulling artifacts from $BUCKET (VM's service account has access)"
gcloud storage cp --recursive "$BUCKET"/* "$ART"/

test -f "$ART/qdrant/manifest.json" || { echo "ERROR: qdrant/manifest.json missing from $BUCKET"; exit 1; }

# Runtime env used by docker-compose for BOTH variable interpolation and the
# gateway container's environment (secrets + paths).
cat > runtime.env <<EOF
HF_TOKEN=${HF_TOKEN}
API_KEY=${API_KEY}
ARTIFACTS_DIR=${ART}
GEN_MODEL_ID=google/gemma-3-4b-it
LOG_LEVEL=${LOG_LEVEL:-INFO}

# --- retrieval A/B knobs (migrated from run_e2e_eval.py; defaults shown).
# Uncomment + change to tune without rebuilding the image, then restart:
#   docker compose --env-file runtime.env up -d gateway
#CONF_GATE_LOOSEN=0.60
#MIN_CHUNK_SCORE=0.50
#MIN_CHUNK_CHARS=50
#MAX_PER_SOURCE=3
#TOP_K_BASE=10
#RERANK_TOP_N=5
EOF
echo ">> wrote runtime.env"

echo ">> Starting Qdrant"
docker compose --env-file runtime.env up -d qdrant

echo ">> Waiting for Qdrant to be ready"
until curl -sf localhost:6333/healthz >/dev/null 2>&1; do sleep 2; done

echo ">> Restoring the vector DB snapshot (skips if already restored)"
sudo apt-get update && sudo apt-get install -y python3-requests
python3 restore_qdrant.py --artifacts "$ART/qdrant" --qdrant-url http://localhost:6333

echo ">> Building + starting the gateway (first build downloads Gemma, ~10-15 min)"
docker compose --env-file runtime.env up -d --build gateway

echo ">> Waiting for the gateway /health to go green"
for i in $(seq 1 120); do
  if curl -sf localhost:8000/health >/dev/null 2>&1; then break; fi
  sleep 5
done
echo ">> Health:"
curl -s localhost:8000/health || echo "  (not ready yet — check: docker compose logs -f gateway)"

cat <<'MSG'

============================================================
 DONE. Test it:
   curl -s localhost:8000/health | jq
   curl -s -X POST localhost:8000/query \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $API_KEY" \
     -d '{"query":"wheat yellow rust dawa","intent":"field_practice"}' | jq

 Logs:   docker compose logs -f gateway
 STOP billing when done (from your PC):  bash stop_vm.sh
============================================================
MSG
