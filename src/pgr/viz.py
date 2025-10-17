from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw


def _ensure_uint8_rgb(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("rgb must be (H,W,3)")
    if x.dtype == np.uint8:
        return x
    if np.issubdtype(x.dtype, np.floating):
        return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    raise TypeError(f"unsupported rgb dtype: {x.dtype}")


def _normalize01(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo + eps else np.zeros_like(a, dtype=np.float32)


def _colormap(cam01: np.ndarray) -> np.ndarray:
    c = np.clip(cam01, 0.0, 1.0)[..., None]
    r = np.minimum(1.5 * c[..., 0], 1.0)
    g = np.clip(1.5 * (c[..., 0] - 0.3), 0.0, 1.0)
    b = np.clip(1.5 * (c[..., 0] - 0.7), 0.0, 1.0) * 0.5
    return np.stack([r, g, b], axis=-1)


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5, *, resize_cam: bool = True) -> Image.Image:
    """Blend CAM (H,W float) onto RGB (H,W,3) as PIL Image."""
    rgb_u8 = _ensure_uint8_rgb(rgb)
    H, W = rgb_u8.shape[:2]
    if cam.ndim != 2:
        raise ValueError("cam must be (H,W)")
    cam_f = cam.astype(np.float32, copy=False)
    if cam_f.shape != (H, W):
        if not resize_cam:
            raise ValueError("cam size mismatch")
        cam_f = np.array(Image.fromarray(_normalize01(cam_f)).resize((W, H), Image.BILINEAR), dtype=np.float32)
    cam01 = _normalize01(cam_f)
    heat = (_colormap(cam01) * 255.0 + 0.5).astype(np.uint8)
    a = float(np.clip(alpha, 0.0, 1.0))
    out = rgb_u8.astype(np.float32) * (1.0 - a) + heat.astype(np.float32) * a
    return Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8), "RGB")


def draw_point_or_box(img: Image.Image, point: Optional[Tuple[int, int]] = None, box: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    """Draw a point and/or a box on a copy of the image."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    if point is not None:
        x, y = point
        r = 4
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=2)
    if box is not None:
        d.rectangle(list(box), outline=(0, 255, 0), width=2)
    return out


def grid(images: List[Image.Image], rows: int, cols: int, pad: int = 4, bg: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Compose a rows×cols grid."""
    if len(images) != rows * cols:
        raise ValueError(f"expected {rows*cols} images, got {len(images)}")
    w, h = images[0].size
    if any(im.size != (w, h) for im in images):
        raise ValueError("all images must have same size")
    canvas = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * (w + pad), r * (h + pad)))
    return canvas


def save_gif(frames: List[Image.Image], path: str, fps: int = 8, loop: int = 0) -> None:
    """Write frames as GIF."""
    if not frames:
        raise ValueError("frames must be non-empty")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / max(1, fps)), loop=loop, optimize=False, disposal=2)
