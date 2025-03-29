#!/bin/bash
# To enable Practical-RIFE, you need to unzip `RIFE_trained_model_lite.zip` under the Practical_RIFE folder to get the `train_log` directory.
# ssh -p 11775 root@140.228.20.3  -L 8001:localhost:8001 -L 8081:localhost:8081
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ray start --include-dashboard=True --head
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cleanup() {
  echo "🧹 Cleaning up background processes..."
  kill $BACK_PID_0 $BACK_PID_1 $BACK_PID_2 $BACK_PID_3
  exit
}

trap cleanup SIGINT
python create_ray_pipe.py &
BACK_PID_0=$!
python ray_web_server.py --port 8001 &
BACK_PID_1=$!
python -m http.server 8081 &
BACK_PID_2=$!

python start_daemon.py &  # this will use the GPU designated by CUDA_VISIBLE_DEVICES to `ray start`
BACK_PID_3=$!

bash start_dit.sh
