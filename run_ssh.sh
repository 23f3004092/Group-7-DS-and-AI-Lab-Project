#!/bin/bash
/usr/bin/ssh -i /home/maruti/.ssh/google_compute_engine -o StrictHostKeyChecking=no maruti@136.117.222.118 'sudo apt-get update; sudo apt-get install -y linux-headers-$(uname -r); curl -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py; sudo python3 install_gpu_driver.py'
