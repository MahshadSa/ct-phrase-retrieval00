# HU -> 3-channel windows (soft tissue, liver, raw)
from __future__ import annotations
import numpy as np


def _window(img: np.ndarray, center: float, width: float) -> np.ndarray:
    """Window a HU-like image to uint8 channel using [center±width/2]."""
    lo = float(center) - float(width) / 2.0
    hi = float(center) + float(width) / 2.0
    x = img.astype(np.float32, copy=False)
    # treat NaNs as low end
    x = np.nan_to_num(x, nan=lo)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def ct3ch(img: np.ndarray) -> np.ndarray:
    """
    Convert a CT slice to 3-channel windowed RGB-like (H,W,3) uint8.

    Channels:
      1) soft tissue: center=40,  width=400
      2) liver:      center=60,  width=150
      3) raw clip:   clip [-1000, 1000] then center=0, width=2000
    """
    # already 3-channel
    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype == np.uint8:
            return img
        # if float, assume [0,1] or HU-like; map to uint8 safely
        x = img.astype(np.float32, copy=False)
        x = np.clip(x, 0.0, 255.0) if x.max() > 1.5 else np.clip(x * 255.0, 0.0, 255.0)
        return x.astype(np.uint8)

    # 2D single-channel inputs
    if img.ndim != 2:
        raise ValueError("ct3ch expects a 2D grayscale or 3-channel image")

    if img.dtype == np.uint8:
        # unknown prior window: replicate the channel
        return np.stack([img, img, img], axis=-1)

    # assume HU-like numeric range (int16/float). Apply standard windows.
    ch1 = _window(img, center=40, width=400)
    ch2 = _window(img, center=60, width=150)
    img_clip = np.clip(img, -1000, 1000).astype(np.float32, copy=False)
    ch3 = _window(img_clip, center=0, width=2000)
    return np.stack([ch1, ch2, ch3], axis=-1)
