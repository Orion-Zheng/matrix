#!/bin/bash
# ssh -L 8001:localhost:8001 -L 8081:localhost:8081 root@171.240.141.228 -p 45927
# CUDA_VISIBLE_DEVICES=1 ray start --include-dashboard=True --head
cleanup() {
  echo "🧹 Cleaning up background processes..."
  kill $BACK_PID_1 $BACK_PID_2 $BACK_PID_3
  exit
}

trap cleanup SIGINT

python ray_web_server.py --port 8001 &
BACK_PID_1=$!
python -m http.server 8081 &
BACK_PID_2=$!

python decode_daemon.py &  # this will use the GPU designated by CUDA_VISIBLE_DEVICES to `ray start`
BACK_PID_3=$!

bash start_dit.sh