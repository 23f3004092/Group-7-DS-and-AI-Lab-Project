#!/usr/bin/env bash
# Stop the VM = stop paying for the GPU. Your data & models stay on disk.
# RUN THIS EVERY TIME YOU FINISH.
source "$(dirname "$0")/config.sh"
echo ">> Stopping $VM_NAME (GPU billing stops; ~\$6/mo disk remains)"
gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
echo ">> Stopped."
