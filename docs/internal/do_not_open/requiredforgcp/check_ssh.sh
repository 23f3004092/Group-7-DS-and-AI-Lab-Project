#!/usr/bin/env bash
source /mnt/d/Group-7-DS-and-AI-Lab-Project/docs/internal/do_not_open/requiredforgcp/config.sh
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="echo 'SSH WORKS'"
