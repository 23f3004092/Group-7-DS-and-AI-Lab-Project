#!/usr/bin/env bash
# Start the VM again. Containers come back automatically (restart: always).
# Give it ~2-3 min, then health-check.
source "$(dirname "$0")/config.sh"
echo ">> Starting $VM_NAME ..."
gcloud compute instances start "$VM_NAME" --zone="$ZONE"

echo ">> Waiting for SSH ..."
until gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo ok" >/dev/null 2>&1; do
  sleep 8
done

echo ">> Checking the gateway (containers auto-restart)"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="curl -s localhost:8000/health || echo 'gateway still warming up — try again in a minute'"

cat <<MSG

To use the API from your app, open a secure tunnel (no public port):
  gcloud compute ssh $VM_NAME --zone=$ZONE -- -N -L 8000:localhost:8000
Then call http://localhost:8000/query with header X-API-Key.
MSG
