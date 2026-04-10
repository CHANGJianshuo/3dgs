"""
Drop-in replacement for fused_ssim, using pure-PyTorch SSIM (from original 3DGS).
Slower than the CUDA fused version, but no compilation needed.
Install as a fake `fused_ssim` package in site-packages.
"""
import torch
import torch.nn.functional as F
from math import exp


def _gaussian(window_size, sigma):
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


_window_cache = {}


def _ssim(img1, img2, window_size=11, padding="same"):
    """SSIM matching original 3DGS implementation. img: [B, C, H, W]."""
    channel = img1.size(1)
    cache_key = (channel, window_size, img1.device, img1.dtype)
    if cache_key not in _window_cache:
        window = _create_window(window_size, channel).to(img1.device).type(img1.dtype)
        _window_cache[cache_key] = window
    window = _window_cache[cache_key]

    pad = window_size // 2 if padding == "same" else 0
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def fused_ssim(img1, img2, padding="same", train=True):
    """Stub matching fused_ssim API used by gsplat simple_trainer."""
    return _ssim(img1, img2, padding=padding)
