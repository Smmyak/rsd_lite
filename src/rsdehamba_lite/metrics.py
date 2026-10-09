from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    channels = pred.shape[1]
    pad = window_size // 2
    window = torch.ones(channels, 1, window_size, window_size, device=pred.device, dtype=pred.dtype)
    window = window / float(window_size * window_size)

    mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_y = F.conv2d(target, window, padding=pad, groups=channels)
    sigma_x = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x * mu_x
    sigma_y = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y * mu_y
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2))
    score = score / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    return score.mean()

