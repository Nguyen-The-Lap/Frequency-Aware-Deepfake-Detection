"""Deterministic evaluation corruptions for tensor batches in [0, 1]."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def corrupt_batch(images, kind, severity=1, seed=0):
    if not 1 <= severity <= 5:
        raise ValueError("severity must be from 1 to 5")
    if kind == "blur":
        return F.avg_pool2d(images, kernel_size=2 * severity + 1, stride=1,
                            padding=severity).clamp(0, 1)
    if kind == "resize":
        scale = max(0.1, 1 - 0.16 * severity)
        small = F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
        return F.interpolate(small, size=images.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    if kind == "noise":
        generator = torch.Generator(device=images.device).manual_seed(seed)
        sigma = 0.02 * severity
        return (images + torch.randn(images.shape, generator=generator, device=images.device,
                                     dtype=images.dtype) * sigma).clamp(0, 1)
    if kind == "jpeg":
        quality = max(5, 100 - 19 * severity)
        arrays = []
        for image in images.detach().cpu().permute(0, 2, 3, 1).numpy():
            rgb = np.uint8(np.clip(image * 255, 0, 255))
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("OpenCV JPEG encoding failed")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            arrays.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        result = np.stack(arrays).copy()
        return torch.from_numpy(result).to(images.device, images.dtype).permute(0, 3, 1, 2) / 255
    raise ValueError("kind must be jpeg, blur, resize, or noise")
