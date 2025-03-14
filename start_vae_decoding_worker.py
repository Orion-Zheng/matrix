import ray
from ray.util.queue import Queue
import os
import time
import torch
from stage3.cogvideox.autoencoder import AutoencoderKLCogVideoX
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.video_processor import VideoProcessor

from post_processors.SuperRes_ESRGAN import ESRGANUpscaler
from post_processors.Interpolator_RIFE import RIFEInterpolator

@ray.remote(num_cpus=1, max_concurrency=2)
class QueueManager:
    def __init__(self):
        self.queue = Queue()

    def put(self, item):
        # print(f"[QueueManager] Added: {item}")
        self.queue.put(item)
        # print(f"[QueueManager] Add Finished")

    def get(self):
        print(f"[LatentQueue] Getting")
        return self.queue.get()
    
    def print_queue(self):
        print(f"[LatentQueue] Current queue: {self.queue}")
    
@ray.remote(num_cpus=4, num_gpus=1)
class LatentDecoder:
    def __init__(self, vae_ckpt_path: str, output_dir: str, 
                 fps=16,
                 super_resolution=True, 
                 frame_interpolation='/workspace/matrix/Practical_RIFE/train_log'):
        self.fps = fps
        self.do_super_resolution = super_resolution
        self.do_frame_interpolation = frame_interpolation
        self.vae_ckpt_path = vae_ckpt_path
        self.output_dir = output_dir
        self.queue = ray.get_actor("latents_queue", namespace='vae_decoder')
        self.device = torch.device("cuda:0")
        
        self.vae = AutoencoderKLCogVideoX.from_pretrained(self.vae_ckpt_path, torch_dtype=torch.bfloat16).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae = torch.compile(self.vae, mode="max-autotune-no-cudagraphs")
        
        self.vae_scale_factor_spatial =  2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scaling_factor_image = self.vae.config.scaling_factor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        
        self.latents = []
        self.last_frame = None
        
        self.sr_upscaler = ESRGANUpscaler(model_scale="2")
        self.sr_upscaler.model.model.eval()
        self.sr_upscaler.model.model.requires_grad_(False)
        self.sr_upscaler.model.model = torch.compile(self.sr_upscaler.model.model, mode="max-autotune-no-cudagraphs")

        self.frame_interpolator = RIFEInterpolator(model_path=frame_interpolation)
        self.frame_interpolator.model.flownet.requires_grad_(False)
        self.frame_interpolator.model.flownet.eval()
        self.frame_interpolator.model.flownet = torch.compile(self.frame_interpolator.model.flownet, mode="max-autotune-no-cudagraphs")
        
    @torch.no_grad()
    def frames_to_video(frames: torch.Tensor, output_path: str, video_processor, fps: int = 16, save=True):
        # frames.shape: [batch_size, num_channels=3, num_frames, height, width]
        full_video = video_processor.postprocess_video(video=frames, output_type='pil')
        assert len(full_video) == 1
        if save:
            export_to_video(full_video[0], output_path, fps=fps)
        return full_video[0]
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, sliced_decode=False) -> torch.Tensor:
        # latents.shape: [batch_size, num_latents, num_channels=16, height, width]
    	# frames.shape: [batch_size, num_channels=3, num_frames, height, width]
        latents = latents.to(self.device)
        assert latents.device == self.vae.device
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        if sliced_decode:
            batch_size, num_channels, num_latent_frames, height, width = latents.size()
            assert batch_size == 1, "Only support one video to decode when using sliced decoding"
            decode_window = 128  # can be any length
            overlap_length = 4
            cur_start = 0
            all_frames = []
            while cur_start + decode_window <= num_latent_frames:
                cur_frames = self.vae.decode(latents[:, :, cur_start:cur_start+decode_window]).sample
                cur_frames_cpu = cur_frames.cpu()
                del cur_frames
                all_frames.append(cur_frames_cpu)
                cur_start += (decode_window - overlap_length)
            if cur_start + overlap_length < num_latent_frames or cur_start == 0:
                cur_frames = self.vae.decode(latents[:, :, cur_start:]).sample
                cur_frames_cpu = cur_frames.cpu()
                del cur_frames
                all_frames.append(cur_frames_cpu)
            frames = [all_frames[0]]
            frames.extend([item[:, :, overlap_length*4:] for item in all_frames[1:]])
            frames = torch.cat(frames, dim=2)
        else:
            frames = self.vae.decode(latents).sample
        return frames
    
    @torch.no_grad()
    def super_resolution(self, frame_list):
        # frame_list: list of PIL images
        sr_video = self.sr_upscaler.process_video(frame_list)
        return sr_video
    
    @torch.no_grad()
    def frame_interpolation(self, frame_list):
        interpolated_frame_list = self.frame_interpolator.interpolate(frame_list, multi=4)
        return interpolated_frame_list
    
    @torch.no_grad()
    def process_data(self):
        counter = 0
        while True:
            latent = ray.get(self.queue.get.remote())
            print(f"[Consumer] Received and processing: {latent.shape}")
            decode_start = time.time()
            self.latents.append(latent)
            self.latents = self.latents[-2:]
            if len(self.latents) < 2:
                print(f"[Consumer] Not enough latents to decode")
                continue
            latent = torch.cat(self.latents, axis=1)
            print(f"[Consumer] Concatenated: {latent.shape}")
            frames = self.decode_latents(latent, sliced_decode=True)
            frames = frames[:, :, 4:, :, :]
            
            print(f"[Consumer] Processed: {frames.shape}")
            video = self.video_processor.postprocess_video(frames, output_type="pil")
            print('Decoded Video Len', len(video[0]))
            decode_end = time.time()
            print(f"1. Decoding time: {decode_end - decode_start}")
            assert len(video) == 1, "Only support one video to decode"
            video = video[0]
                
            # if self.do_frame_interpolation:
            #     start_time = time.time()
            #     if self.last_frame is not None:  # Interpolate between videos
            #         video = [self.last_frame] + video
            #         video = self.frame_interpolation(video)
            #         video = video[1:]
            #     else:
            #         video = self.frame_interpolation(video)
            #     self.last_frame = video[-1]
            #     end_time = time.time()
            #     print(f"2. Interpolation time: {end_time - start_time}")
            
            # if self.do_super_resolution:
            #     start_time = time.time()
            #     video = self.super_resolution(video)
            #     end_time = time.time()
            #     print(f"3. Super resolution time: {end_time - start_time}")
            
            output_path = os.path.join(self.output_dir, f"test_{str(counter)}.mp4")
            export_to_video(video, output_path, fps=self.fps)
            counter += 1
            
if __name__ == "__main__":
    # assert torch.cuda.device_count() == 1
    # output_dir = "/workspace/matrix/samples/decoupled_vae_output_interpolated_superres"
    output_dir = "/workspace/matrix/samples/decoupled_vae_output"
    vae_ckpt_path = "/matrix_ckpts/stage3/vae"
    ray.init(address='auto')  
    # Start the named queue manager
    queue = QueueManager.options(namespace='vae_decoder', name="latents_queue").remote()
    consumer = LatentDecoder.remote(vae_ckpt_path, output_dir)
    ray.get(consumer.process_data.remote())