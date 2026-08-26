#!/bin/bash
/usr/bin/ssh -i /home/maruti/.ssh/google_compute_engine -o StrictHostKeyChecking=no maruti@136.117.222.118 'sudo grep -A 2 -B 2 -i "error" /var/log/nvidia-installer.log'
