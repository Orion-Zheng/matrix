import argparse
from typing import Literal, Optional, Union
from dataclasses import dataclass
import numpy as np
import os
import sys
sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))
print( '/'.join(os.path.realpath(__file__).split('/')[:-2]))
import torch
# from stage4.cogvideox.pipelines import CogVideoXStreamingPipeline
from wm_gym.pipelines import CogVideoXStreamingPipeline
from vlm_reward_model import OpenAIRewardModel
from stage4.cogvideox.transformer import CogVideoXTransformer3DModel
from stage4.cogvideox.scheduler import LCMSwinScheduler

from stage4.cogvideox.pipelines.pipeline_output import CogVideoXPipelineOutput
from stage4.cogvideox.loader import CogVideoXLoraLoaderMixin
from stage4.cogvideox.autoencoder import AutoencoderKLCogVideoX
from stage4.cogvideox.transformer import CogVideoXTransformer3DModel
from stage4.cogvideox.scheduler import (
    LCMSwinScheduler,
    CogVideoXDPMScheduler,
    CogVideoXSwinDPMScheduler,
    expand_timesteps_with_group,
)
from stage4.cogvideox.control_adapter import CONTROL_SIGNAL_TO_PROMPT

from diffusers.utils import export_to_video, load_image, load_video

import decord
import PIL.Image
import datetime

import random

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
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    wm_init_args = prepare_wm_env_init_args(wm_gen_config)
    pipe.gym_init(**wm_init_args)
    pipe.vae.disable_slicing()
    pipe.vae.disable_tiling()
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
    
    def step(self, action):
        assert action in CONTROL_SIGNAL_TO_PROMPT.keys(), f"Invalid action: {action}. Valid actions are: {CONTROL_SIGNAL_TO_PROMPT.keys()}"
        truncated = False  # whether the episode is truncated by max_iteractions
        
        # 1. Generate the next observations based on the action
        self.current_state, self.current_frames, self.current_pil_list = self.wm_gym_pipe.gym_step(action)
        # 2. Get the vlm feedback from observations
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
        model_path="/mnt/d/model_ckpts_stage4/model_ckpts_stage4/stage3",
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
    
    state, reward, terminated, truncated, info = env.step("DL")
    video_clip = env.render(mode="pil")
    print("reward: ", reward)
    print("vlm_response", info)
    export_to_video(video_clip, fps=matrix_gen_config.fps, 
                    output_video_path=os.path.join(debug_clip_output_dir, "1_DL_video.mp4"))
    
    state, reward, terminated, truncated, info = env.step("DR")
    video_clip = env.render(mode="pil")
    print("reward: ", reward)
    print("vlm_response", info)
    export_to_video(video_clip, fps=matrix_gen_config.fps, 
                    output_video_path=os.path.join(debug_clip_output_dir, "2_DR_video.mp4"))
    
    
if __name__ == "__main__":
    test()