#!/bin/bash
/usr/bin/ssh -i /home/maruti/.ssh/google_compute_engine -o StrictHostKeyChecking=no maruti@136.117.222.118 'cd ~/farmervision; sudo docker compose --env-file runtime.env up -d gateway; sudo docker compose --env-file runtime.env ps'
