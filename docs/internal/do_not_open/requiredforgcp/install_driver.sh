#!/usr/bin/env bash
source /mnt/d/Group-7-DS-and-AI-Lab-Project/docs/internal/do_not_open/requiredforgcp/config.sh
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="sudo apt-get update && sudo apt-get install -y linux-headers-\$(uname -r) && curl -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py && sudo python3 install_gpu_driver.py"
