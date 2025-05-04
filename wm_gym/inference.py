import argparse
from typing import Literal, Optional
from dataclasses import dataclass
import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.abspath('.'))
# import sys
# sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))
# print( '/'.join(os.path.realpath(__file__).split('/')[:-2]))
import torch
from wm_gym.vlm_reward_model import OpenAIRewardModel
from wm_gym.pipelines import CogVideoXStreamingPipeline
from wm_gym.cogvideox.transformer import CogVideoXTransformer3DModel
from wm_gym.cogvideox.scheduler import LCMSwinScheduler

from wm_gym.cogvideox.pipelines.pipeline_output import CogVideoXPipelineOutput
from wm_gym.cogvideox.loader import CogVideoXLoraLoaderMixin
from wm_gym.cogvideox.autoencoder import AutoencoderKLCogVideoX
from wm_gym.cogvideox.transformer import CogVideoXTransformer3DModel
from wm_gym.cogvideox.scheduler import (
    LCMSwinScheduler,
    CogVideoXDPMScheduler,
    CogVideoXSwinDPMScheduler,
    expand_timesteps_with_group,
)
from wm_gym.cogvideox.control_adapter import CONTROL_SIGNAL_TO_PROMPT

from diffusers.utils import export_to_video, load_image, load_video

import decord
import PIL.Image
import datetime


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
    original_inference_steps: int = 50,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 8,
    gpu_id: int = 0,
    transformer_path: str = None,
    control_signal: str = None,
    control_signal_type: str = "downsampled",
    control_seed: int = 42,
    num_noise_groups: int=4,
    num_sample_groups: int = 20,
    init_video_clip_frame: int = 65,
    lcm_multiplier: int = 1,
    do_classifier_free_guidance: bool = False
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
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        torch_dtype=dtype,
        # low_cpu_mem_usage=False , set it false for load sharded weights
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(os.path.join(model_path, 'vae'), torch_dtype=dtype)
    pipe = CogVideoXStreamingPipeline.from_pretrained(model_path, vae=vae, transformer=transformer, torch_dtype=dtype)
    pipe.scheduler = LCMSwinScheduler.from_config(pipe.scheduler.config)

    # Init_video should be pillow list.
    video_reader = decord.VideoReader(video_path)
    video_num_frames = num_frames # len(video_reader)
    video_fps = video_reader.get_avg_fps()
    sampling_interval = video_fps/fps
    frame_indices = np.round(np.arange(0, video_num_frames, sampling_interval)).astype(int).tolist()
    frame_indices = frame_indices[:init_video_clip_frame]
    video = video_reader.get_batch(frame_indices).asnumpy()
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
    
    # Keep_cache feature conflicts with the tiling/slicing feature
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

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
    
    with torch.no_grad():
        start_time = datetime.datetime.now()
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            use_dynamic_cfg=False,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            generator=torch.Generator().manual_seed(seed),
            control_signal=control_signal,
            init_video=video,
            num_noise_groups=num_noise_groups,
            num_sample_groups=num_sample_groups,
            original_inference_steps=original_inference_steps,
            lcm_multiplier = lcm_multiplier,
            do_classifier_free_guidance=False
        ).frames[0]
        print("Time cost: ", datetime.datetime.now() - start_time)
        export_to_video(video_generate, output_path, fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, help="The description of the video to be generated")
    parser.add_argument("--model_path", type=str, help="Path of the pre-trained model use")
    parser.add_argument("--video_path", type=str, help="The path of the video to be extend.")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=256, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=17, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=720, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=480, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=16, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="float16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    # control arguments
    parser.add_argument("--control_signal", type=str, required=True, help="control signal of original video (and can be longer which contains the control signal of video to be generated).")
    parser.add_argument("--control_signal_type", type=str, choices=["raw", "downsampled"], default="downsampled", help="Whether the control signal is recorded in video raw fps or downsampled fps (i.e. 4 fps), if raw, its length >= init_video_clip_frame.")
    parser.add_argument("--control_seed", type=int, default=42, help="The seed for reproducibility")
    # swin arguments
    parser.add_argument("--num_noise_groups", type=int, default=4, help="Number of noise groups")
    parser.add_argument("--num_sample_groups", type=int, default=8, help="Number of sampled videos groups")
    parser.add_argument("--init_video_clip_frame", type=int, default=33, help="Frame number of init_video to be clipped, should be 4n+1")
    # lcm arguments
    parser.add_argument("--original_inference_steps", type=int, default=40, help="Number of DDIM steps for training consistency model")
    parser.add_argument("--lcm_multiplier", type=int, default=1, help="Number of lcm multiplier")

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
        original_inference_steps=args.original_inference_steps,
        lcm_multiplier = args.lcm_multiplier,
    )
@dataclass
class VLMRewardArgs:
    model: str
    api_key: str
    reward_query: str
    reward_criteria: str

@dataclass
class MatrixGenerationArgs:
    prompt: str
    model_path: str
    video_path: str
    lora_path: Optional[str] = None
    lora_rank: int = 256
    output_path: str = "./output.mp4"
    num_inference_steps: int = 4
    num_frames: int = 17
    width: int = 720
    height: int = 480
    fps: int = 16
    num_videos_per_prompt: int = 1
    dtype: str = "float16"
    seed: int = 42
    gpu_id: int = 0
    use_dynamic_cfg: bool = False
    do_classifier_free_guidance: bool = False
    
    # swin arguments
    num_noise_groups: int = 4
    init_video_clip_frame: int = 17

    # lcm arguments
    original_inference_steps: int = 40
    lcm_multiplier: int = 1
    
    # gym arguments
    max_iteractions: int = 1000
    
    def __post_init__(self):
        if self.dtype == "float16":
            self.dtype = torch.float16
        elif self.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

def seed_everything(seed=42):
    """
    Set the random seed for all libraries to a fixed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_wm_env_init_args(wm_gen_config):
    # init_video should be pillow list.
    video_reader = decord.VideoReader(wm_gen_config.video_path)
    video_num_frames = wm_gen_config.num_frames
    video_fps = video_reader.get_avg_fps()
    sampling_interval = video_fps/wm_gen_config.fps
    frame_indices = np.round(np.arange(0, video_num_frames, sampling_interval)).astype(int).tolist()
    frame_indices = frame_indices[:wm_gen_config.init_video_clip_frame]
    video = video_reader.get_batch(frame_indices).asnumpy()
    video = [PIL.Image.fromarray(frame) for frame in video]

    # Generate the video frames based on the prompt.
    wm_gen_config.num_frames = len(video)
    init_args = {
        "prompt": wm_gen_config.prompt,
        "num_videos_per_prompt": wm_gen_config.num_videos_per_prompt,
        "num_inference_steps": wm_gen_config.num_inference_steps,
        "height": wm_gen_config.height,
        "width": wm_gen_config.width,
        "num_frames": wm_gen_config.num_frames,
        "use_dynamic_cfg": wm_gen_config.use_dynamic_cfg,
        "generator": torch.Generator().manual_seed(wm_gen_config.seed),
        "control_signal": None,
        "init_video": video,
        "num_noise_groups": wm_gen_config.num_noise_groups,
        "original_inference_steps": wm_gen_config.original_inference_steps,
        "lcm_multiplier": wm_gen_config.lcm_multiplier,
        "do_classifier_free_guidance": False,
    }
    return init_args

def load_matrix_gym_pipe(wm_gen_config):
    transformer = CogVideoXTransformer3DModel.from_pretrained(
            os.path.join(wm_gen_config.model_path, "transformer"),
            torch_dtype=wm_gen_config.dtype,
        )
    # NOTE: `keep_cache` feature conflicts with the tiling/slicing feature
    vae = AutoencoderKLCogVideoX.from_pretrained(os.path.join(wm_gen_config.model_path, 'vae'), 
                                                    torch_dtype=wm_gen_config.dtype)
    # vae.to(0)
    # v = vae.encode(torch.randn((1, 3, 17, 480, 720), dtype=torch.float16).to(vae.device))
    pipe = CogVideoXStreamingPipeline.from_pretrained(wm_gen_config.model_path, 
                                                        vae=vae, transformer=transformer, 
                                                        torch_dtype=wm_gen_config.dtype)
    pipe.scheduler = LCMSwinScheduler.from_config(pipe.scheduler.config)
    if wm_gen_config.lora_path:  # If you're using with lora, add this code
        pipe.load_lora_weights(wm_gen_config.lora_path, 
                               weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"],)  # lora_scale=1 / lora_rank  # It seems that there are some issues here, removed.
    pipe.to(wm_gen_config.gpu_id)  # pipe._execution_device from cpu --> cuda:0
    pipe.wm_gen_config = wm_gen_config
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    wm_init_args = prepare_wm_env_init_args(wm_gen_config)
    pipe.gym_init(**wm_init_args)
    # pipe.vae.disable_slicing()
    # pipe.vae.disable_tiling()
    return pipe
    
class matrixGym:
    def __init__(self, wm_gym_pipe, vlm_reward_model):
        # TODO: Prepare base video here
        self.wm_gym_pipe = wm_gym_pipe
        self.wm_gen_config = wm_gym_pipe.wm_gen_config
        self.vlm_reward_model = vlm_reward_model
    
    def reset(self):
        self.current_step = 0
        info = {}
        self.current_state, self.current_frames, self.current_pil_list = self.wm_gym_pipe.gym_reset()
        return self.current_state, info
    
    def vlm_feedback(self, observations, action=None):
        # vlm input: observations(pil_list), action(optional)
        # vlm output: reward, terminated, info
        full_analysis, short_answer_reward = self.vlm_reward_model.analyze(observations)
        reward = int(short_answer_reward)
        terminated = False
        info = {"env info": full_analysis}
        return reward, terminated, info
    
    def step(self, action, skip_reward=False):
        assert action in CONTROL_SIGNAL_TO_PROMPT.keys(), f"Invalid action: {action}. Valid actions are: {CONTROL_SIGNAL_TO_PROMPT.keys()}"
        truncated = False  # whether the episode is truncated by max_iteractions
        
        # 1. Generate the next observations based on the action
        self.current_state, self.current_frames, self.current_pil_list = self.wm_gym_pipe.gym_step(action)
        # 2. Get the vlm feedback from observations
        if skip_reward:
            reward, terminated, info = None, False, None
        else:
            reward, terminated, info = self.vlm_feedback(self.current_pil_list, action)
        
        self.current_step += 1
        if self.current_step == self.wm_gen_config.max_iteractions:  
            truncated = terminated = True
            info = {}
            print("Max iterations reached, stop generating.")
        return self.current_state, reward, terminated, truncated, info
    
    def render(self, mode="pil"):
        observation = self.wm_gym_pipe.gym_render(mode=mode)
        return observation


def test():
    matrix_gen_config = MatrixGenerationArgs(
        prompt="On a lush green meadow, a white car is driving. From an overhead panoramic shot, \
                this car is adorned with blue and red stripes on its body, and it has a black spoiler at the rear. \
                The camera follows the car as it moves through a field of golden wheat, surrounded by green grass and trees. \
                In the distance, a river and some hills can be seen, with a cloudless blue sky above.",
        model_path="/home/andy/matrix_stage4_ckpt",
        video_path="/home/andy/matrix/base_video.mp4",
    )  # natural domain randomization support :)
    vlm_rm_config = VLMRewardArgs(
        model="gpt-4o",
        api_key="debug",#os.environ["OPENAI_API_KEY"],
        reward_query = (
            "You are a video analyst. Your task is to analyze a sequence of consecutive images and describe the spatial relationship "
            "between the car and any potential obstacles. Based on this analysis, assess the risk of a possible collision. "
        ),
        reward_criteria = (
            "Here is the video description: {} "
            "Return 1 if there is no risk of collision with any obstacle. "
            "Return 0 if there is a potential risk of collision, but no collision has occurred yet. "
            "Return -1 if you believe a collision has already occurred between the car and an obstacle, regardless of whether there was damage."
            "Your response must only contain one of the following: 1, 0, or -1. Do not include any additional explanation or description."
        ),
    )
    seed_everything(matrix_gen_config.seed)
    
    debug_clip_output_dir = "./debug_clip_output"
    os.makedirs(debug_clip_output_dir, exist_ok=True)
    
    gpt4_rm = OpenAIRewardModel(**vars(vlm_rm_config))
    wm_gym_pipe = load_matrix_gym_pipe(matrix_gen_config)
    
    env = matrixGym(wm_gym_pipe, gpt4_rm)
    state, info = env.reset()
    video_clip = env.render(mode="pil")
    export_to_video(video_clip, fps=matrix_gen_config.fps, 
                    output_video_path=os.path.join(debug_clip_output_dir, "0_reset_video.mp4"))
    
    state, reward, terminated, truncated, info = env.step("DL", skip_reward=True)
    video_clip = env.render(mode="pil")
    print("reward: ", reward)
    print("vlm_response", info)
    export_to_video(video_clip, fps=matrix_gen_config.fps, 
                    output_video_path=os.path.join(debug_clip_output_dir, "1_DL_video.mp4"))
    
    state, reward, terminated, truncated, info = env.step("DR", skip_reward=True)
    video_clip = env.render(mode="pil")
    print("reward: ", reward)
    print("vlm_response", info)
    export_to_video(video_clip, fps=matrix_gen_config.fps, 
                    output_video_path=os.path.join(debug_clip_output_dir, "2_DR_video.mp4"))
    
if __name__ == "__main__":
    # main()
    test()