import os
import time 

import torch
import ray

from diffusers.utils import export_to_video, load_image, load_video
from diffusers.video_processor import VideoProcessor

from matrix_ray_worker import EngineConfig, ParallelConfig
from matrix_ray_driver import RayMatrixPipeline
from ray_pipeline_utils import QueueManager, timer
    
def debug_daemon():
    # init_ray()
    matrix_ckpt_path = "/matrix_ckpts/stage3/vae"
    video_output_dir = "/workspace/matrix/ray_pipeline"
    frame_interpolator_model_path = "/workspace/matrix/Practical_RIFE/train_log"
    export_video_for_debug = False
    
    dist_env_var = {
        'env_vars': {"MASTER_ADDR": "localhost", "MASTER_PORT": "12355"}
    }
    actors = ray.util.list_named_actors(all_namespaces=True)
    print([actor_name for actor_name in actors])
    dit2vae_queue = ray.get_actor("dit2vae_queue", namespace='matrix')  # DiT --> VAE
    vae2post_queue = ray.get_actor("vae2post_queue", namespace='matrix')  # VAE --> Postprocessing
    post2web_queue = ray.get_actor("post2web_queue", namespace='matrix')  # Postprocessing --> Web
    
    # Set the parallel configuration in the parallel config
    parallel_config = ParallelConfig(world_size=2, dit_parallel_size=0, vae_parallel_size=1, post_parallel_size=1)
    engine_config = EngineConfig(parallel_config=parallel_config)
    matrix_ray_pipline = RayMatrixPipeline.from_pretrained(matrix_ckpt_path, engine_config, dist_env_var, 
                                                           frame_interpolator_model_path=frame_interpolator_model_path,
                                                           torch_dtype=torch.bfloat16)
    matrix_ray_pipline.connect_pipeline()  # Let each worker try to connect to the recv/send queue
    num_latents = 2  # VAE decode num_latents latents at a time
    counter = 0
    latents_window = []
    while True:
        latents = ray.get(dit2vae_queue.get.remote())
        if latents.shape[1] == 1:
            latents_window.append(latents)
            latents_window = latents_window[-num_latents:]
            if len(latents_window) < num_latents:
                print(f"[Consumer] Not enough latents to decode")
                continue
            latent = torch.cat(latents_window, axis=1)
            
        # print(f"[Consumer] Received and processing: {latent.shape}")
        with timer("Decoding Latents"):
            frames = matrix_ray_pipline.call_vae(latents=latent)[0]  # only the rank 0 worker will return the results
        frames = frames[:, :, 4:]  # torch.Size([1, 3, num_frames=4, 480, 720])
        print(frames.shape)
        
        if parallel_config.post_parallel_size > 0:  # if there is postprocessor workers
            full_video = matrix_ray_pipline.call_postprocessor(frames=frames)
            full_video = full_video[0]  # the return of the ray workers is a list. There is only one postprocessor so we get the first element (also the only element)
        else:
            vae_config_block_out_channels = [128, 256, 256, 512]
            vae_scale_factor_spatial = 2 ** (len(vae_config_block_out_channels) - 1)
            video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
            with timer(f"Postprocessing {latents.size(1)} latents"):
                full_video = video_processor.postprocess_video(video=frames, output_type='pil')
        
        if export_video_for_debug:
            video_output_path = os.path.join(video_output_dir, f"video_{counter}.mp4")
            print("Exporting video to: ", video_output_path)
            export_to_video(full_video, video_output_path, fps=16)

        for frame_img in full_video:
            ray.get(post2web_queue.put.remote(frame_img))
        counter += 1
        
    # TODO: Clean up all process groups

def new_daemon():
    # init_ray()
    matrix_ckpt_path = "/matrix_ckpts/stage3/vae"
    video_output_dir = "/workspace/matrix/ray_pipeline"
    frame_interpolator_model_path = "/workspace/matrix/Practical_RIFE/train_log"
    export_video_for_debug = False
    
    dist_env_var = {
        'env_vars': {"MASTER_ADDR": "localhost", "MASTER_PORT": "12355"}
    }
    
    # Set the parallel configuration in the parallel config
    parallel_config = ParallelConfig(world_size=2, dit_parallel_size=0, vae_parallel_size=1, post_parallel_size=1)
    engine_config = EngineConfig(parallel_config=parallel_config)
    matrix_ray_pipeline = RayMatrixPipeline.from_pretrained(matrix_ckpt_path, engine_config, dist_env_var, 
                                                           frame_interpolator_model_path=frame_interpolator_model_path,
                                                           torch_dtype=torch.bfloat16)
    matrix_ray_pipeline.connect_pipeline()  # Let each worker try to connect to the recv/send queue
    matrix_ray_pipeline.start_postprocessor_loop()
    matrix_ray_pipeline.start_vae_loop()
    try:
        print("Daemon are running. Press Ctrl+C to exit.")
        while True:
            time.sleep(10)  # Keep the process alive to prevent actors from being destroyed
    except KeyboardInterrupt:
        print("Shutting down.")
    
if __name__ == "__main__":
    # debug_daemon()
    new_daemon()