#!/bin/bash
/usr/bin/ssh -i /home/maruti/.ssh/google_compute_engine -o StrictHostKeyChecking=no maruti@136.117.222.118 << 'EOF'
cd ~/farmervision
sed -i '/^torch/d' requirements.txt
sed -i '/^torchvision/d' requirements.txt
sed -i 's/RUN pip install --no-cache-dir -r requirements.txt/RUN pip install --no-cache-dir torch torchvision --index-url https:\/\/download.pytorch.org\/whl\/cu124\nRUN pip install --no-cache-dir -r requirements.txt/' Dockerfile
sudo docker compose --env-file runtime.env build --no-cache gateway
sudo docker compose --env-file runtime.env up -d --force-recreate gateway
EOF
