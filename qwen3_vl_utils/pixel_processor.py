import math
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Mirror the Qwen2-VL logic so we stay aligned with HF processor outputs."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _ensure_three_channels(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def pil_to_pixel_tensor(image: Image.Image, device: Optional[torch.device] = None) -> torch.Tensor:
    """Converts an RGB PIL image into a [1, 3, H, W] tensor normalized to [0, 1]."""
    array = np.array(image, dtype=np.float32)
    array = _ensure_three_channels(array)
    tensor = torch.from_numpy(array / 255.0).permute(2, 0, 1).contiguous()
    tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def pixel_tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """Converts a batched tensor [B, 3, H, W] back to a list of PIL images."""
    if tensor.dim() != 4:
        raise ValueError(f"Expected a 4D tensor, received {tensor.shape}")
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    tensor = tensor.permute(0, 2, 3, 1).contiguous()
    images: List[Image.Image] = []
    for sample in tensor:
        arr = (sample.numpy() * 255.0).round().clip(0, 255).astype("uint8")
        images.append(Image.fromarray(arr))
    return images


class PixelValueBuilder:
    """Differentiable recreation of Qwen2.5-VL image preprocessing."""

    def __init__(self, image_processor, device: torch.device):
        self.patch_size = getattr(image_processor, "patch_size", 14)
        self.temporal_patch_size = getattr(image_processor, "temporal_patch_size", 2)
        merge_size = getattr(image_processor, "merge_size", None)
        self.merge_size = merge_size if merge_size is not None else getattr(image_processor, "spatial_merge_size", 2)
        self.min_pixels = getattr(image_processor, "min_pixels", 56 * 56)
        self.max_pixels = getattr(image_processor, "max_pixels", 14 * 14 * 4 * 1280)
        image_mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.device = device

    def build_inputs(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Converts a batch of pixel tensors [B, 3, H, W] into the flattened patch tokens expected by Qwen2.5-VL.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError(f"Expected image tensor with 4 dims, received {images.shape}")

        batch_flattened: List[torch.Tensor] = []
        grids: List[torch.Tensor] = []
        for sample in images:
            flatten, grid = self._encode_single(sample.unsqueeze(0))
            batch_flattened.append(flatten)
            grids.append(grid)

        pixel_values = torch.cat(batch_flattened, dim=0)
        image_grid_thw = torch.stack(grids, dim=0)
        return pixel_values, image_grid_thw

    def _encode_single(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        `image` has shape [1, 3, H, W] in [0, 1]. Returns flattened patches and (t, h, w) grid.
        """
        _, _, height, width = image.shape
        resized_h, resized_w = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        if resized_h != height or resized_w != width:
            image = F.interpolate(image, size=(resized_h, resized_w), mode="bicubic", align_corners=False)

        image = image.clamp(0.0, 1.0)
        normalized = (image - self.image_mean).div(self.image_std)
        patches = normalized
        if patches.shape[0] == 1:
            patches = patches.repeat(self.temporal_patch_size, 1, 1, 1)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        flatten = patches.view(
            grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )

        grid = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long, device=self.device)
        return flatten, grid
