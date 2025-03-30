#!/bin/bash
# This file is modified from https://github.com/xdit-project/xDiT/blob/0.4.1/examples/run_cogvideo.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -x

export PYTHONPATH=$PWD:$PYTHONPATH
echo $PYTHONPATH
# export HF_HOME="/mnt/world_model/longxiang/.cache/huggingface"

export NCCL_BUFFSIZE=1048576  # for 24 GB memory, the default 32MB NCCL buffer per channel would be too large and cause OOM (These memory wouldn't be released by `torch.cuda.empty_cache()`)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes 1 --nproc-per-node 2 --master-port 29501 /workspace/matrix/stage3/inference_ulysses.py \
--model_path /matrix_ckpts/stage2 \
--transformer_path /matrix_ckpts/stage3/transformer \
--output_path /workspace/matrix/samples/ulysses/output_debug/ulysses_debug.mp4 \
--prompt "The video shows a white car driving on a country road on a sunny day. The car comes from the back of the scene, moving forward along the road, with open fields and distant hills surrounding it. As the car moves, the vegetation on both sides of the road and distant buildings can be seen. The entire video records the car's journey through the natural environment using a follow-shot technique." \
--image_or_video_path /workspace/matrix/samples/base_video.mp4 \
--control_signal "" \
--guidance_scale 1 \
--seed 43 \
--split_text_embed_in_sp true \
--num_sample_groups 500 \
--control_repeat_length 5 \
--ulysses_degree 2 \
--ring_degree 1 \
--height 480 \
--width 720 \
--warmup_steps 0 \
--init_video_clip_frame 33 \
--ray_mode 