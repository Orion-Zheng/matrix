import argparse
from typing import Literal, Optional
from dataclasses import dataclass
import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from wm_gym.vlm_reward_model import OpenAIRewardModel
from wm_gym.cogvideox.pipelines import CogVideoXStreamingPipeline
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

@torch.no_grad()
def load_matrix_gym_pipe(wm_gen_config, disable_progress_bar=False):
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
    pipe.set_progress_bar_config(disable=disable_progress_bar) 
    if wm_gen_config.lora_path:  # If you're using with lora, add this code
        pipe.load_lora_weights(wm_gen_config.lora_path, 
                               weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"],)  # lora_scale=1 / lora_rank  # It seems that there are some issues here, removed.
    pipe.to(wm_gen_config.gpu_id)  # pipe._execution_device from cpu --> cuda:0
    pipe.wm_gen_config = wm_gen_config
    wm_init_args = prepare_wm_env_init_args(wm_gen_config)
    pipe.gym_init(**wm_init_args)
    return pipe
    
class matrixGym:
    def __init__(self, wm_gym_pipe, vlm_reward_model):
        # TODO: Prepare base video here
        self.wm_gym_pipe = wm_gym_pipe
        self.wm_gen_config = wm_gym_pipe.wm_gen_config
        self.vlm_reward_model = vlm_reward_model
    
    @torch.no_grad()
    def reset(self):
        self.current_step = 0
        info = {}
        self.current_state, self.current_frames, self.current_pil_list = self.wm_gym_pipe.gym_reset()
        return self.current_state, info
    
    @torch.no_grad()
    def vlm_feedback(self, observations, frame_downsample, action=None):
        # vlm input: observations(pil_list), action(optional)
        # vlm output: reward, terminated, info
        full_analysis, short_answer_reward = self.vlm_reward_model.analyze(observations, down_sample=frame_downsample)
        reward = int(short_answer_reward)
        terminated = False
        info = {"full_analysis": full_analysis}
        return reward, terminated, info
    
    @torch.no_grad()
    def _step(self, action, skip_reward=False, frame_downsample=4):
        assert action in CONTROL_SIGNAL_TO_PROMPT.keys(), f"Invalid action: {action}. Valid actions are: {CONTROL_SIGNAL_TO_PROMPT.keys()}"
        truncated = False  # whether the episode is truncated by max_iteractions
        
        # 1. Generate the next observations based on the action
        self.current_state, self.current_frames, self.current_pil_list = self.wm_gym_pipe.gym_step(action)
        # 2. Get the vlm feedback from observations
        if skip_reward:
            reward, terminated, info = None, False, None
        else:
            reward, terminated, info = self.vlm_feedback(self.current_pil_list, frame_downsample, action)
        
        self.current_step += 1
        if self.current_step == self.wm_gen_config.max_iteractions:  
            truncated = terminated = True
            info = {}
            print("Max iterations reached, stop generating.")
        return self.current_state, reward, terminated, truncated, info
    
    def step(self, action, pad_k_step=0, padding_action="D", skip_reward=False, frame_downsample=4):
        # Action may have some delay, so we need to step k times to ensure the action is applied.
        # We take the action then pad the rest of the steps with padding_action.
        # The reward is calculated from the clip of all the steps.
        assert action in CONTROL_SIGNAL_TO_PROMPT.keys(), f"Invalid action: {action}. Valid actions are: {CONTROL_SIGNAL_TO_PROMPT.keys()}"
        truncated = False  # whether the episode is truncated by max_iteractions
        
        total_pil_list = []
        total_frames = []
        total_states = []
        current_state, current_frames, pil_list = self.wm_gym_pipe.gym_step(action)
        total_states.append(current_state.clone())
        total_frames.append(current_frames.clone())
        total_pil_list.extend(pil_list.copy())
        
        for _ in range(pad_k_step):
            current_state, current_frames, pil_list = self.wm_gym_pipe.gym_step(padding_action)
            total_states.append(current_state.clone())
            total_frames.append(current_frames.clone())
            total_pil_list.extend(pil_list.copy())
        
        self.current_state = total_states    
        self.current_frames = total_frames
        self.current_pil_list = total_pil_list
        
        if skip_reward:
            reward, terminated, info = None, False, None
        else:
            reward, terminated, info = self.vlm_feedback(total_pil_list, frame_downsample, action)
        
        self.current_step += 1
        if self.current_step == self.wm_gen_config.max_iteractions:  
            truncated = terminated = True
            info = {}
            print("Max iterations reached, stop generating.")
        
        return total_states, reward, terminated, truncated, info
        
    @torch.no_grad()
    def render(self, mode="pil"):
        if mode == "pil":
            return self.current_pil_list
        elif mode == "rgb":
            return self.current_frames
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: pil, rgb.")

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
            "Given these consecutive images from a car racing game, analyze the moving direction and the locations of possible obstacles. And answer the question: do you think the car has any collision with obstacles during the process? "
            "In this game, collisions don’t actually cause any damage — even if the car passes through an obstacle, it still counts as a collision."
        ),
        reward_criteria = (
            "Here is the video description: {} "
            "Return 1 if there is not any collision happened between the car and an obstacle, "
            "Return -1 if you believe a collision has already occurred between the car and an obstacle, regardless of whether there was damage."
            "Your response must only contain one of the following: 1 or -1. Do not include any additional explanation or description."
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