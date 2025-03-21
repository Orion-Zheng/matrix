import os
import time
import torch
from stage3.cogvideox.autoencoder import AutoencoderKLCogVideoX
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
import numpy as np

from contextlib import contextmanager
import time

@contextmanager
def timer(label="Block"):
    start_time = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensures all CUDA operations are completed before measuring time
    end_time = time.perf_counter()
    print(f"{label} took {end_time - start_time:.6f} seconds")
    
def decode_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    # latents.shape: [batch_size, num_latents, num_channels=16, height, width]
    # frames.shape: [batch_size, num_channels=3, num_frames, height, width]
    vae_scaling_factor_image = vae.config.scaling_factor
    n_tokens = latents.shape[1]
    with torch.no_grad():
        latents = latents.to(vae.device)
        assert latents.device == vae.device
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_frames, num_channels, height, width]
        latents = 1 / vae_scaling_factor_image * latents
        frames = vae.decode(latents).sample
    return frames

def frames_to_video(frames: torch.Tensor, output_path: str, video_processor, fps: int = 16):
    # frames.shape: [batch_size, num_channels=3, num_frames, height, width]
    full_video = video_processor.postprocess_video(video=frames, output_type='pil')
    assert len(full_video) == 1
    export_to_video(full_video[0], output_path, fps=fps)
    return full_video[0]

n_tokens = 2
output_dir = "/workspace/matrix/samples/vae_decode_test"
vae_ckpt_path = "/matrix_ckpts/stage3/vae"
latents = torch.load("/workspace/matrix/latents_100.pt")
device = torch.device("cuda:0")
vae = AutoencoderKLCogVideoX.from_pretrained(vae_ckpt_path, torch_dtype=torch.bfloat16).to(device)
vae.eval()
# vae = torch.compile(vae, mode="max-autotune-no-cudagraphs")
vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)

for idx, i in enumerate(range(latents.shape[1]-n_tokens+1)):
    # idx, i = 0, 0
    cur_latents = latents[:, i:i+n_tokens]
    # print(cur_latents.shape)
    with timer(f"Decoding {n_tokens} latents"):
        frames = decode_latents(cur_latents, vae)
    # print(frames.shape)
    new_frames = frames[:, :, 4:]
    with timer("Postprocessing"):
        full_video = video_processor.postprocess_video(video=frames, output_type='pil')
    assert len(full_video) == 1
    output_path = os.path.join(output_dir, f"video_{idx}.mp4")
    export_to_video(full_video[0], output_path, fps=16)
    if idx > 3:
        break