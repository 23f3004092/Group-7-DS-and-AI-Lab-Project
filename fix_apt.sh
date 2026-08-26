#!/bin/bash
/usr/bin/ssh -i /home/maruti/.ssh/google_compute_engine -o StrictHostKeyChecking=no maruti@136.117.222.118 << 'EOF'
if [ -f /etc/apt/sources.list.d/debian.sources ]; then
  sudo sed -i 's/Components: main$/Components: main contrib non-free non-free-firmware/g' /etc/apt/sources.list.d/debian.sources
fi
if [ -f /etc/apt/sources.list ]; then
  sudo sed -i 's/ main$/ main contrib non-free non-free-firmware/g' /etc/apt/sources.list
fi
sudo apt-get update
sudo apt-get install -y nvidia-driver firmware-misc-nonfree
EOF
