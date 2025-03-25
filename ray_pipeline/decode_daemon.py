import os

import torch
import ray

from diffusers.utils import export_to_video, load_image, load_video
from diffusers.video_processor import VideoProcessor

from ray_vae_worker import EngineConfig, ParallelConfig
from ray_pipeline.ray_vae_pipeline import RayVAEPipeline
from ray_pipeline_utils import QueueManager, timer

def frame_post_process(frames):
    # Options: Save to disk, send to postprocessing queue, etc
    with timer(f"Postprocessing {frames.size(2)} Frames"):  # TODO: parallelize frames postprocessing
        full_video = video_processor.postprocess_video(video=frames, output_type='pil')
        for frame in full_video[0]:
            post_process_queue.put.remote(frame)
        # video_output_path = os.path.join(video_output_dir, f"video_{counter}.mp4")
        # export_to_video(full_video[0], video_output_path, fps=16)

if __name__ == "__main__":
    matrix_ckpt_path = "/matrix_ckpts/stage3/vae"
    video_output_dir = "/workspace/matrix/ray_pipeline"
    vae_parallel = 1
    
    dist_env_var = {
        'env_vars': {"MASTER_ADDR": "localhost", "MASTER_PORT": "12355"}
    }
    vae_config_block_out_channels = [128, 256, 256, 512]
    vae_scale_factor_spatial = 2 ** (len(vae_config_block_out_channels) - 1)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    
    ray.init(address='auto')  
    dit_queue = QueueManager.options(namespace='vae_decoder', name="latents_queue").remote()  # Create a queue for communicating with DiT process
    post_process_queue = ray.get_actor("postproc_queue", namespace="vae_decoder")
    
    parallel_config = ParallelConfig(world_size=vae_parallel, vae_parallel_size=vae_parallel)
    engine_config = EngineConfig(parallel_config=parallel_config)
    vae_ray_pipline = RayVAEPipeline.from_pretrained(matrix_ckpt_path, engine_config, dist_env_var, torch_dtype=torch.bfloat16)
    
    counter = 0
    latents_window = []
    while True:
        latent = ray.get(dit_queue.get.remote())
        if latent.shape[1] == 1:
            latents_window.append(latent)
            latents_window = latents_window[-2:]
            if len(latents_window) < 2:
                print(f"[Consumer] Not enough latents to decode")
                continue
            latent = torch.cat(latents_window, axis=1)
            
        print(f"[Consumer] Received and processing: {latent.shape}")
        frames = vae_ray_pipline(latents=latent)[0]  # only the rank 0 worker will return the results
        
        frames = frames[:, :, 4:]  # torch.Size([1, 3, num_frames=4, 480, 720])
        print(frames.shape)
        frame_post_process(frames)
        counter += 1