#!/usr/bin/env bash
source /mnt/d/Group-7-DS-and-AI-Lab-Project/docs/internal/do_not_open/requiredforgcp/config.sh
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="sudo apt-get install -y cloud-guest-utils && sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1"
