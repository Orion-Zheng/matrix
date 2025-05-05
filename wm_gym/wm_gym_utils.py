import os
import numpy as np
import decord
import imageio
import tempfile
from PIL import Image
from IPython.display import Video, display

def create_temp_video(pil_images, fps=16):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        writer = imageio.get_writer(temp.name, fps=fps)

        for img in pil_images:
            frame = np.array(img)  # Convert PIL.Image to numpy array
            writer.append_data(frame)

        writer.close()
        return temp.name