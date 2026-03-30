import numpy as np
import torch
from PIL import Image

from .util import BaseTransform

__all__ = ["ToTensor"]


class ToTensor(BaseTransform):
    """Minimal tensor conversion to avoid a hard torchvision dependency."""

    @classmethod
    def apply(cls, data):
        if isinstance(data, torch.Tensor):
            return data.float()

        if isinstance(data, Image.Image):
            data = np.array(data)

        if not isinstance(data, np.ndarray):
            raise TypeError(f"ToTensor only works with numpy arrays, PIL images or tensors [{type(data)}]")

        if data.ndim == 2:
            data = data[:, :, None]

        tensor = torch.from_numpy(np.ascontiguousarray(data.transpose(2, 0, 1))).float()
        if data.dtype == np.uint8:
            tensor /= 255.0
        return tensor
