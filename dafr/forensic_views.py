from __future__ import annotations

import io
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms.functional import normalize


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def _tensor(image: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(image))
    value = value.permute(2, 0, 1).float() / 255.0
    return normalize(value, IMAGENET_MEAN, IMAGENET_STD)


def low_bit_view(rgb: np.ndarray, size: int) -> np.ndarray:
    bits = (rgb & 7).astype(np.uint16)
    return _resize((bits * 255 // 7).astype(np.uint8), size)


def directional_residual_view(rgb: np.ndarray, size: int, clip: int = 32) -> np.ndarray:
    x = rgb.astype(np.int16)
    horizontal = np.zeros_like(x)
    vertical = np.zeros_like(x)
    diagonal = np.zeros_like(x)
    horizontal[:, 1:] = x[:, 1:] - x[:, :-1]
    vertical[1:, :] = x[1:, :] - x[:-1, :]
    diagonal[1:, 1:] = x[1:, 1:] - x[:-1, :-1]
    residual = np.stack(
        [horizontal.mean(2), vertical.mean(2), diagonal.mean(2)], axis=2
    )
    residual = np.clip(residual, -clip, clip)
    view = ((residual + clip) * 255.0 / (2 * clip)).astype(np.uint8)
    return _resize(view, size)


def spectral_view(rgb: np.ndarray, size: int, sigma: float = 3.0) -> np.ndarray:
    resized = _resize(rgb, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    radius = (radius / max(float(radius.max()), 1e-6)).astype(np.float32)
    residual = np.abs(magnitude - cv2.GaussianBlur(magnitude, (0, 0), sigma))
    channels = (magnitude, magnitude * radius, residual)
    return np.stack(
        [
            cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for channel in channels
        ],
        axis=2,
    )


def random_degradation(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    operation = random.choice(("jpeg", "resize", "blur", "noise"))
    if operation == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.choice((95, 85, 75, 60, 45)))
        buffer.seek(0)
        with Image.open(buffer) as restored:
            return restored.convert("RGB")
    if operation == "resize":
        scale = random.choice((0.75, 0.5, 0.35, 0.25))
        width = max(16, int(image.width * scale))
        height = max(16, int(image.height * scale))
        return image.resize((width, height), Image.Resampling.BICUBIC).resize(
            image.size, Image.Resampling.BICUBIC
        )
    if operation == "blur":
        return image.filter(ImageFilter.GaussianBlur(random.choice((0.5, 1.0, 1.5, 2.0))))
    sigma = random.choice((2.0, 4.0, 8.0, 12.0))
    array = np.asarray(image, dtype=np.float32)
    noisy = np.clip(array + np.random.normal(0.0, sigma, array.shape), 0, 255)
    return Image.fromarray(noisy.astype(np.uint8), mode="RGB")


class ForensicTransform:
    def __init__(self, size: int = 256, clip: int = 32, sigma: float = 3.0):
        self.size = size
        self.clip = clip
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> dict[str, torch.Tensor]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return {
            "rgb": _tensor(_resize(rgb, self.size)),
            "lowbit": _tensor(low_bit_view(rgb, self.size)),
            "npr": _tensor(directional_residual_view(rgb, self.size, self.clip)),
            "spectrum": _tensor(spectral_view(rgb, self.size, self.sigma)),
        }
