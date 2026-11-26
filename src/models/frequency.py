"""Differentiable orthonormal 2D DCT and radial low/mid/high frequency bands."""
import math

import torch
from torch import nn


def dct_matrix(size: int, device=None, dtype=None):
    n = torch.arange(size, device=device, dtype=dtype or torch.float32)
    k = n[:, None]
    matrix = torch.cos(math.pi / size * (n[None, :] + 0.5) * k)
    scale = torch.full((size,), math.sqrt(2.0 / size), device=device, dtype=dtype or torch.float32)
    scale[0] = math.sqrt(1.0 / size)
    return matrix * scale[:, None]


class FrequencyBands(nn.Module):
    def __init__(self, low=0.12, high=0.35, method="dct"):
        super().__init__()
        if not 0 < low < high < 1:
            raise ValueError("Require 0 < low < high < 1")
        if method not in {"dct", "fft", "srm"}:
            raise ValueError("method must be dct, fft, or srm")
        self.low, self.high = low, high
        self.method = method

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != x.shape[-2]:
            raise ValueError("Frequency transform expects square BCHW input")
        size = x.shape[-1]
        if self.method == "srm":
            kernels = torch.tensor([
                [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
                [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            ], device=x.device, dtype=x.dtype).div_(4)
            weight = kernels[:, None].repeat(x.shape[1], 1, 1, 1)
            return torch.nn.functional.conv2d(x, weight, padding=1, groups=x.shape[1])
        coords = (torch.arange(size, device=x.device, dtype=x.dtype) - size // 2) / size
        if self.method == "dct":
            coords = coords + 0.5
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square()) / math.sqrt(2)
        masks = torch.stack((radius < self.low, (radius >= self.low) & (radius < self.high),
                             radius >= self.high)).to(x.dtype)
        if self.method == "dct":
            basis = dct_matrix(size, x.device, x.dtype)
            coeff = torch.matmul(torch.matmul(basis, x), basis.T)
            separated = coeff[:, None] * masks[None, :, None, :, :]
        else:
            coeff = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
            radial_masks = masks[:, None, :, :]
            separated = torch.fft.ifft2(torch.fft.ifftshift(
                coeff[:, None] * radial_masks[None], dim=(-2, -1)), norm="ortho").real
        return separated.flatten(1, 2)


class FrequencyCNN(nn.Module):
    def __init__(self, width=16, feature_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(9, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(width * 4, feature_dim)

    def forward(self, x):
        return self.projection(self.features(x).flatten(1))
