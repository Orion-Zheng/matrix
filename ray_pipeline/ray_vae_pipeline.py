import os
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from itertools import islice, repeat

import torch
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.video_processor import VideoProcessor

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from xfuser.ray.pipeline.ray_utils import initialize_ray_cluster
from xfuser.ray.pipeline.pipeline_utils import GPUExecutor
from xfuser.ray.worker.worker_wrappers import RayWorkerWrapper

from ray_vae_worker import EngineConfig, ParallelConfig
from ray_pipeline_utils import timer

class RayVAEPipeline(GPUExecutor):
    runtime_env = {}
    total_workers = []
    # dit_workers = []
    vae_workers = []
    def _init_executor(self):
        self._init_ray_workers()
        self._run_workers(self.workers,"init_worker_distributed_environment")

    def _init_ray_workers(self):
        placement_group = initialize_ray_cluster(self.engine_config.parallel_config, 'auto') 
        print("placement_group: ", placement_group)
        # create placement group and worker wrapper instance for lazy load worker
        self.workers = []
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            # print("bundle_id: ", bundle_id)
            # print("bundle: ", bundle)
            # Skip bundles without GPUs
            if not bundle.get("GPU", 0):
                continue

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_id,
                placement_group_capture_child_tasks=True,
            )

            # VAE workers
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=scheduling_strategy,
                runtime_env=self.runtime_env,
            )(RayWorkerWrapper).remote(
                self.engine_config.parallel_config,
                "ray_vae_worker.ParallelVAEWorker",
                bundle_id,  # GPU ID
            )
            
            self.vae_workers.append(worker)
            self.workers.append(worker)

    def _run_workers(
        self,
        workers: List[ray.ObjectRef],
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """

        count = len(workers)
        # If using SPMD worker, all workers are the same, so we should execute
        # the args on all workers. Otherwise, we skip the first worker's args
        # because those args will go to the driver worker.
        first_worker_args_index: int = 0
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, first_worker_args_index, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, first_worker_args_index, None)
        # print("method: ", method)
        # print("all_worker_args: ", all_worker_args)
        # print("all_worker_kwargs: ", all_worker_kwargs)
        # Start the ray workers first.
        ray_workers = workers
        ray_worker_outputs = [
            # `execute_method` is defined in `RayWorkerWrapper`, will call the `method` of the worker
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)  
            for (worker, worker_args, worker_kwargs
                 ) in zip(ray_workers, all_worker_args, all_worker_kwargs)
        ]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return ray_worker_outputs

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return ray_worker_outputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, engine_config: EngineConfig, runtime_env, **kwargs):
        cls.runtime_env = runtime_env
        pipeline = cls(engine_config)
        pipeline._run_workers(pipeline.workers, "from_pretrained", pretrained_model_name_or_path, **kwargs)
        return pipeline

    def __call__(self, **kwargs):
        return self._run_workers(self.workers,"execute", **kwargs)
    
if __name__ == "__main__":
    matrix_ckpt_path = "/matrix_ckpts/stage3/vae"
    video_output_dir = "/workspace/matrix/ray_pipeline"
    video_output_path = os.path.join(video_output_dir, f"video.mp4")
    
    latents = torch.load("/workspace/matrix/latents_100.pt")
    latents = latents[:, :2]  # decode 100 latents at a time will cause OOM on 4090
    
    vae_config_block_out_channels = [128, 256, 256, 512]
    vae_scale_factor_spatial = 2 ** (len(vae_config_block_out_channels) - 1)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
        
    dist_env_var = {
        'env_vars': {"MASTER_ADDR": "localhost", "MASTER_PORT": "12355"}
    }
    
    parallel_config = ParallelConfig(world_size=2, vae_parallel_size=2)
    engine_config = EngineConfig(parallel_config=parallel_config)
    vae_ray_pipline = RayVAEPipeline.from_pretrained(matrix_ckpt_path, engine_config, dist_env_var, torch_dtype=torch.bfloat16)
    frames = vae_ray_pipline(latents=latents)[0]  # only the rank 0 worker will return the results
    frames = frames[:, :, 4:]
    print(frames.shape)
    with timer(f"Postprocessing {latents.size(1)} latents"):
        full_video = video_processor.postprocess_video(video=frames, output_type='pil')
    export_to_video(full_video[0], video_output_path, fps=16)