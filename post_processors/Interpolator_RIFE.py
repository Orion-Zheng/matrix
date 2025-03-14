# ref: https://github.com/hzwer/Practical-RIFE
# ref: https://colab.research.google.com/drive/1gIAzh8Mn8E7aqDIMtM74e894HpW_TA3S#scrollTo=NqRbvfA7LzhR
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from Practical_RIFE.train_log.RIFE_HDv3 import Model
# from model.pytorch_msssim import ssim_matlab  # If you want SSIM-based logic

def pad_to_multiple_of_64(tensor):
    """
    Pads a [1, C, H, W] tensor so that H and W are multiples of 64.
    Returns:
      padded_tensor, pad_w, pad_h
    where pad_w and pad_h are how many pixels of padding were added
    to the right (width) and bottom (height) respectively.
    """
    B, C, H, W = tensor.shape
    # Figure out how many extra pixels are needed so that W and H become multiples of 64
    pad_w = ((W - 1) // 64 + 1) * 64 - W
    pad_h = ((H - 1) // 64 + 1) * 64 - H

    # (left, right, top, bottom) = (0, pad_w, 0, pad_h)
    padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return padded_tensor, pad_w, pad_h

def unpad_from_multiple_of_64(tensor, pad_w, pad_h):
    """
    Removes the padding that was added on the right (pad_w) and bottom (pad_h).
    `tensor` must have shape [1, C, H, W].
    """
    B, C, H, W = tensor.shape
    # The new H will be H - pad_h
    # The new W will be W - pad_w
    return tensor[:, :, : (H - pad_h), : (W - pad_w)]

class RIFEInterpolator:
    """
    A class for performing frame interpolation using a RIFE HDv3 (or similar) model.
    This version pads frames to multiples of 64 to avoid dimension mismatch in multi-scale.
    
    Usage:
        interpolator = RIFEInterpolator(model_path='train_log')
        out_frames = interpolator.interpolate(some_list_of_pil_frames, multi=2)
    """

    def __init__(self, model_path='train_log', scale=1.0, device=None, fp16=False):
        """
        Initialize the RIFE model and set up parameters.
        
        Args:
            model_path (str): The directory/path containing the trained RIFE HDv3 model weights.
            scale (float): The scaling factor to use within the RIFE model (e.g. 0.5 for 4K).
            device (str, optional): The device to run inference on ('cuda' or 'cpu'). 
                If None, will automatically choose 'cuda' if available, otherwise 'cpu'.
            fp16 (bool): Whether to use half-precision inference (requires GPU support).
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.scale = scale
        self.fp16 = fp16

        # Load the RIFE model (adjust per your file paths / project structure)
        self.model = Model()  # from train_log.RIFE_HDv3 import Model
        self.model.load_model(model_path, -1)
        self.model.eval()
        self.model.device()

        # If fp16 is enabled on cuda, switch default tensor type
        if self.fp16 and self.device.type == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    def _make_inference(self, I0, I1, n):
        """
        Generate n interpolated frames between I0 and I1 using the RIFE model's multi-scale logic.
        
        Args:
            I0 (torch.Tensor): First frame of shape [1, C, H, W].
            I1 (torch.Tensor): Second frame of shape [1, C, H, W].
            n (int): Number of intermediate frames (multi - 1).
        
        Returns:
            list of torch.Tensor: The interpolated frames, each [1, C, H, W].
        """
        with torch.no_grad():
            # If model has version >=3.9, a simpler approach is used
            if hasattr(self.model, 'version') and self.model.version >= 3.9:
                result = []
                for i in range(n):
                    t = (i + 1) / (n + 1)
                    mid = self.model.inference(I0, I1, t, self.scale)
                    result.append(mid)
                return result
            else:
                # Fallback logic: do a single pass, then recursively subdivide
                if n == 1:
                    mid = self.model.inference(I0, I1, scale=self.scale)
                    return [mid]
                else:
                    mid = self.model.inference(I0, I1, scale=self.scale)
                    left_side = self._make_inference(I0, mid, n // 2)
                    right_side = self._make_inference(mid, I1, n // 2)
                    if n % 2 == 1:
                        return left_side + [mid] + right_side
                    else:
                        return left_side + right_side

    def interpolate(self, pil_list, multi=2):
        """
        Interpolate between consecutive frames in a list of PIL Images,
        returning a new list with intermediate frames inserted.
        
        Args:
            pil_list (list of PIL.Image.Image): The input frames to be interpolated.
            multi (int): For each pair of frames, produce (multi - 1) intermediate frames.
        
        Returns:
            list of PIL.Image.Image: The resulting frames after interpolation.
        """
        if len(pil_list) < 2:
            return pil_list  # No interpolation needed for a single frame.

        out_list = []

        # Convert PIL -> Tensor [1, C, H, W]
        def pil_to_tensor(img_pil):
            arr = np.array(img_pil.convert("RGB"))
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device, non_blocking=True)
            # If we're in half-precision mode, cast to half
            if self.fp16 and self.device.type == 'cuda':
                tensor = tensor.half()
            return tensor

        # Convert Tensor [1, C, H, W] -> PIL
        def tensor_to_pil(tensor):
            arr = tensor[0].clamp(0, 1).detach().cpu().numpy() * 255.0
            arr = arr.astype(np.uint8).transpose(1, 2, 0)
            return Image.fromarray(arr)

        for i in range(len(pil_list) - 1):
            frame0_pil = pil_list[i]
            frame1_pil = pil_list[i + 1]

            I0 = pil_to_tensor(frame0_pil)
            I1 = pil_to_tensor(frame1_pil)

            # --- Pad both frames to multiples of 64 ---
            I0_padded, pad_w0, pad_h0 = pad_to_multiple_of_64(I0)
            I1_padded, pad_w1, pad_h1 = pad_to_multiple_of_64(I1)

            # If frames differ in required padding, unify them to the max
            pad_w = max(pad_w0, pad_w1)
            pad_h = max(pad_h0, pad_h1)

            # Re-pad so both are exactly the same shape
            if pad_w != pad_w0 or pad_h != pad_h0:
                I0_padded = F.pad(I0_padded, (0, pad_w - pad_w0, 0, pad_h - pad_h0))
            if pad_w != pad_w1 or pad_h != pad_h1:
                I1_padded = F.pad(I1_padded, (0, pad_w - pad_w1, 0, pad_h - pad_h1))

            n = multi - 1
            if n > 0:
                mids_padded = self._make_inference(I0_padded, I1_padded, n)
            else:
                mids_padded = []

            # Add the original frame0 to output
            out_list.append(frame0_pil)

            # Unpad each interpolated mid, convert to PIL
            for mid_padded in mids_padded:
                mid_unpadded = unpad_from_multiple_of_64(mid_padded, pad_w, pad_h)
                out_list.append(tensor_to_pil(mid_unpadded))

        # Append the final frame
        out_list.append(pil_list[-1])
        return out_list

