MODEL_PATH="/mnt/d/model_ckpts_stage4/model_ckpts_stage4/stage3"
VIDEO_PATH="/home/andy/matrix/base_video.mp4"

python inference.py \
--prompt "On a lush green meadow, a white car is driving. From an \
overhead panoramic shot, this car is adorned with blue and red stripes \
on its body, and it has a black spoiler at the rear. The camera follows the \
car as it moves through a field of golden wheat, surrounded by green grass and \
trees. In the distance, a river and some hills can be seen, with a cloudless \
blue sky above." \
--model_path "/mnt/d/model_ckpts_stage4/model_ckpts_stage4/stage3" \
--video_path "/home/andy/matrix/base_video.mp4" \
--control_signal D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,DL,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D \
--output_path "./output_30s.mp4" \
--num_sample_groups 120 --num_inference_steps 4 --lcm_multiplier 1 --dtype float16 --num_frames 17
