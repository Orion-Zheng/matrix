import argparse
from typing import Literal
import numpy as np
import os
import sys
sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))

import torch
from stage3.cogvideox.pipelines.pipeline_cogvideox_dit import CogVideoXStreamingPipeline
from stage3.cogvideox.transformer import CogVideoXTransformer3DModel
from stage3.cogvideox.scheduler import CogVideoXSwinDPMScheduler

from diffusers.utils import export_to_video, load_image, load_video

import decord
import PIL.Image

from torchao.quantization import quantize_, int8_weight_only, float8_weight_only

from sageattention import sageattn
import torch.nn.functional as F

def quantize_model(part, quantization_scheme):
    print(f'====== Using {quantization_scheme} quantized model for streaming ======')
    if quantization_scheme == "int8":
        quantize_(part, int8_weight_only())
    elif quantization_scheme == "fp8":
        quantize_(part, float8_weight_only())
    return part

def enable_sage_attn(attn_type):
    print(f'====== Using {attn_type} attention for streaming ======')
    if attn_type == 'sage':
        F.scaled_dot_product_attention = sageattn
    elif attn_type == 'fa3':
        from sageattention.fa3_wrapper import fa3
        F.scaled_dot_product_attention = fa3
    elif attn_type == 'fa3_fp8':
        from sageattention.fa3_wrapper import fa3_fp8
        F.scaled_dot_product_attention = fa3_fp8
        
def generate_random_control_signal(
        length, seed, repeat_lens=[2, 2, 2], signal_choices=['D', 'DR', 'DL'],
        change_prob_increment=0.2,
    ):
        if not signal_choices or not repeat_lens \
            or len(repeat_lens) != len(signal_choices) \
            or length < 1:
            raise ValueError("Invalid parameters")
        rng = np.random.default_rng(seed)
        result = []
        current_repeat = 0
        current_idx = 0
        change_prob = change_prob_increment
        for i in range(length):
            if current_repeat >= repeat_lens[current_idx]:
                if change_prob >= 1 or rng.uniform(0, 1) < change_prob:
                    if current_idx == 0:
                        current_idx_choices = [j for j in range(1, len(signal_choices))]
                        current_idx = rng.choice(current_idx_choices)
                    else:
                        current_idx = 0
                    current_repeat = 1
                    change_prob = change_prob_increment
                else:
                    current_repeat += 1
                    change_prob = min(1, change_prob + change_prob_increment)
            else:
                current_repeat += 1
            result.append(signal_choices[current_idx])
        return ','.join(result)
    


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: int = 1360,
    height: int = 768,
    output_path: str = "./output.mp4",
    video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 8,
    gpu_id: int = 0,
    control_signal: str = None,
    control_signal_type: str = "downsampled",
    control_seed: int = 42,
    num_noise_groups: int=4,
    num_sample_groups: int = 20,
    init_video_clip_frame: int = 65,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """
    enable_sage_attn('sage')

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )
    # quant_type = 'fp8'
    # transformer = quantize_model(part=transformer, quantization_scheme=quant_type)
    # transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)
    # print("Transformer compiled successfully.")
    scheduler = CogVideoXSwinDPMScheduler.from_config(os.path.join(model_path, "scheduler"), timestep_spacing="trailing")

    pipe = CogVideoXStreamingPipeline.from_pretrained(model_path, transformer=transformer, scheduler=scheduler, torch_dtype=dtype)
    pipe.vae.to("cpu")  # move vae to cpu when we decouple the vae to a separate gpu
    # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)  # raise error
    # pipe.scheduler, pipe.transformer, pip.vae, pipe.vae.transformer, pipe.vae.transformer.transformer
    
    # Init_video should be pillow list.
    video_reader = decord.VideoReader(video_path)
    video_num_frames = len(video_reader)
    video_fps = video_reader.get_avg_fps()
    sampling_interval = video_fps/fps  # 1
    frame_indices = np.round(np.arange(0, video_num_frames, sampling_interval)).astype(int).tolist()
    frame_indices = frame_indices[:init_video_clip_frame]  # frame of base video
    video = video_reader.get_batch(frame_indices).asnumpy()  # (frame=65, height, width, channel)
    video = [PIL.Image.fromarray(frame) for frame in video]
    if sampling_interval > 1:
        control_signal_list = control_signal.split(",")
        control_signal_list = [control_signal_list[i] for i in frame_indices]
        control_signal = ",".join(control_signal_list)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"],
            # lora_scale=1 / lora_rank  # It seems that there are some issues here, removed.
            )

    pipe.to(gpu_id)
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    num_frames = len(video)
    
    # Pad control signal for new frames generation and [the redundancy used by the last window]
    if control_signal_type == "raw":
        control_signal_list = control_signal.split(",")
        control_signal_list = [control_signal_list[i] for i in range(0, len(control_signal_list), 4)]
        control_signal = ",".join(control_signal_list)
    if len(control_signal.split(",")) < (num_frames - 1) / 4 * (num_sample_groups/num_noise_groups + 1) + 1:
        control_padding_length = int(np.ceil((num_frames - 1) / 4 * (num_sample_groups/num_noise_groups + 1))) + 1 - len(control_signal.split(","))
        control_signal = control_signal + "," + generate_random_control_signal(control_padding_length, seed=control_seed)
    print(f"Padded control signal: {control_signal}")
    with torch.no_grad():
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            use_dynamic_cfg=False,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            control_signal=control_signal,
            init_video=video,
            num_noise_groups=num_noise_groups,
            num_sample_groups=num_sample_groups,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, help="The description of the video to be generated")
    parser.add_argument("--model_path", type=str, help="Path of the pre-trained model use")
    parser.add_argument("--video_path", type=str, help="The path of the video to be extend.")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=256, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=41, help="NOT USED HERE")
    parser.add_argument("--width", type=int, default=720, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=480, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=16, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    # control arguments
    parser.add_argument("--control_signal", type=str, required=True, help="control signal of original video (and can be longer which contains the control signal of video to be generated).")
    parser.add_argument("--control_signal_type", type=str, choices=["raw", "downsampled"], default="downsampled", help="Whether the control signal is recorded in video raw fps or downsampled fps (i.e. 4 fps), if raw, its length >= init_video_clip_frame.")
    parser.add_argument("--control_seed", type=int, default=42, help="The seed for reproducibility")
    # swin arguments
    parser.add_argument("--num_noise_groups", type=int, default=4, help="Number of noise groups")
    parser.add_argument("--num_sample_groups", type=int, default=8, help="Number of sampled videos groups")
    parser.add_argument("--init_video_clip_frame", type=int, default=65, help="Frame number of init_video to be clipped, should be 4n+1")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        video_path=args.video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
        gpu_id=args.gpu_id,
        control_signal=args.control_signal,
        control_signal_type=args.control_signal_type,
        control_seed=args.control_seed,
        num_sample_groups=args.num_sample_groups,
        num_noise_groups=args.num_noise_groups,
        init_video_clip_frame=args.init_video_clip_frame,
    )