# ref: https://github.com/ai-forever/Real-ESRGAN
import torch
from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np

class ESRGANUpscaler:
    """
    A class for upscaling images and videos using RealESRGAN.
    """

    def __init__(self, model_scale: str = "4"):
        # Determine the device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"device: {self.device}")
        self.model = RealESRGAN(self.device, scale=int(model_scale))
        self.model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth')
    
    @torch.no_grad()
    def process_image(self, image: Image.Image) -> Image.Image:
        input_array = np.array(image)  # Convert PIL Image to NumPy array for processing
        sr_array = self.model.predict(input_array)
        return sr_array  # 'predict' returns a PIL Image

    @torch.no_grad()
    def process_video(self, frames: list[Image.Image]) -> list[Image.Image]:
        sr_frames = []
        for image in frames:
            sr_frame = self.process_image(image)
            sr_frames.append(sr_frame)
        return sr_frames
    