import os
import sys
import time
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from itertools import islice, repeat
sys.path.append("/workspace/matrix")  

import torch
import torch.distributed as dist
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video, load_image, load_video

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from xfuser.ray.pipeline.ray_utils import initialize_ray_cluster
from xfuser.ray.pipeline.pipeline_utils import GPUExecutor
from xfuser.ray.worker.worker_wrappers import RayWorkerWrapper
from xfuser.ray.worker.worker import WorkerBase
from xfuser.core.distributed.parallel_state import (init_distributed_environment,
                                                    init_vae_group,
                                                    get_world_group,
                                                    get_vae_parallel_group)

from stage3.cogvideox.autoencoder import AutoencoderKLCogVideoX
from stage3.cogvideox.parallel_vae_utils import VAEParallelState
from ray_pipeline_utils import timer

@dataclass
class ParallelConfig:
    world_size: int = 1
    dit_parallel_size: int = 0
    vae_parallel_size: int = 1 # 0 means the vae is in the same process with diffusion
    def __post_init__(self):
        self.dp_degree = 1
        self.cfg_degree = 1
        self.sp_degree = 1
        self.pp_degree = 1
        self.tp_degree = 1
    
@dataclass(frozen=True)
class EngineConfig:
    parallel_config: ParallelConfig
    # model_config: ModelConfig
    # runtime_config: RuntimeConfig
    # fast_attn_config: FastAttnConfig

class ParallelVAEDecodeWrapper:
    # 1. Add video post processing to 
    # 2. (Optional) Send to Postprocessing Queue
    def __init__(
        self, 
        vae,
    ):
        self.vae = vae
        # vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
        # self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # latents.shape: [batch_size, num_latents, num_channels=16, height, width]
        # frames.shape: [batch_size, num_channels=3, num_frames, height, width]
        vae_scaling_factor_image = self.vae.config.scaling_factor
    
        assert latents.device == self.vae.device
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_frames, num_channels, height, width]
        latents = 1 / vae_scaling_factor_image * latents
        frames = self.vae.decode(latents).sample
        return frames

    def execute(self, **kwargs):
        latents = kwargs.get('latents', None)
        assert latents is not None and self.vae is not None
        rank = get_world_group().rank
        # print(f"Rank {rank} is running the VAE")
        frames = None
        latents = latents.to(self.vae.device)
        # if rank == 0:
        #     print(f"========= Decode {latents.size(1)} Latents Together ============")
        # with timer(f"Decoding {latents.size(1)} latents"):
        print("input of vae worker: ", latents.shape)
        frames = self.decode_latents(latents)
        # with timer(f"Postprocessing {latents.size(1)} latents"):
        #     full_video = self.video_processor.postprocess_video(video=frames, output_type='pil')
        return frames
        
class ParallelVAEWorker(WorkerBase):
    """
    A worker class that executes the VAE on a GPU.
    """
    parallel_config: ParallelConfig
    def __init__(
        self,
        parallel_config: ParallelConfig,
        rank: int,
    ) -> None:
        WorkerBase.__init__(self)
        self.parallel_config = parallel_config
        self.rank = rank
        self.vae = None
    
    def init_worker_distributed_environment(self):
        # print('Arguments when init the worker: ', self.rank, self.parallel_config.world_size)
        # print('Env Variables of worker: ', os.environ)
        assert "MASTER_ADDR" in os.environ, "MASTER_ADDR is not set in the worker"
        assert "MASTER_PORT" in os.environ, "MASTER_PORT is not set in the worker"
        init_distributed_environment(  # `init_process_group` + `set_device` in each worker
            rank=self.rank,
            world_size=self.parallel_config.world_size,
        )
        # if only running the vae parallel, setting `dit_parallel_size` to 0
        init_vae_group(0, self.parallel_config.vae_parallel_size, torch.distributed.Backend.NCCL)
        VAEParallelState.initialize(vae_group=get_vae_parallel_group())
        
    def from_pretrained(
        self, 
        pretrained_model_name_or_path: str,
        **kwargs
    ):
        parallel_decoding_idx = 0
        local_rank = get_world_group().local_rank
        vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_model_name_or_path,  **kwargs)
        vae.enable_parallel_decoding(parallel_decoding_idx)
        vae = vae.to(f"cuda:{local_rank}")
        self.vae = ParallelVAEDecodeWrapper(vae) # Wrapper will automatically apply video_processor
        return
    
    def prepare_run(self, input_config, steps: int = 3, sync_steps: int = 1):
        if self.vae is not None:
            return self.vae.execute()
        return None
    
    def execute(self, **kwargs):
        return self.vae.execute(**kwargs)  # this will run `execute` of xxxWrapper