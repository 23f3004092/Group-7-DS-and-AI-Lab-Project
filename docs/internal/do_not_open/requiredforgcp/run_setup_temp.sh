#!/usr/bin/env bash
cd /mnt/d/Group-7-DS-and-AI-Lab-Project/docs/internal/do_not_open/requiredforgcp
source config.sh
echo ">> Creating ~/farmervision on VM"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="mkdir -p ~/farmervision"

echo ">> Copying files to VM"
gcloud compute scp --recurse ./* "$VM_NAME":~/farmervision/ --zone="$ZONE" --quiet
gcloud compute scp .env "$VM_NAME":~/farmervision/.env --zone="$ZONE" --quiet

echo ">> Running 04_vm_setup.sh on VM"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="cd ~/farmervision && bash 04_vm_setup.sh"
