#!/usr/bin/env bash
# OPTIONAL: open port 8000 ONLY to your current public IP (not the whole internet).
# Use this only if you can't use the SSH tunnel (Option A in the plan).
# The API_KEY header is still required on every request.
source "$(dirname "$0")/config.sh"

MY_IP="$(curl -s https://api.ipify.org)"
echo ">> Your public IP is $MY_IP"

gcloud compute firewall-rules create fv-gateway-myip \
  --allow=tcp:8000 \
  --source-ranges="${MY_IP}/32" \
  --target-tags=farmervision \
  --description="FarmerVision gateway, restricted to my IP" \
  2>/dev/null \
  || gcloud compute firewall-rules update fv-gateway-myip --source-ranges="${MY_IP}/32"

# The container currently binds 8000 to localhost only. To use this rule, change
# the gateway ports in docker-compose.yml from "127.0.0.1:8000:8000" to "8000:8000"
# and re-run: docker compose --env-file runtime.env up -d gateway
EXTERNAL_IP="$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
echo ">> Reach the API at:  http://${EXTERNAL_IP}:8000/query   (header X-API-Key)"
echo ">> To close it later:  gcloud compute firewall-rules delete fv-gateway-myip"
