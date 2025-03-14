# CUDA_VISIBLE_DEVICES=7 ray start --include-dashboard=True --head
CUDA_VISIBLE_DEVICES=1 python /workspace/matrix/start_vae_decoding_worker.py

CUDA_VISIBLE_DEVICES=0 python /workspace/matrix/decoupled_generation_example.py