from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict
import json
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


@dataclass
class SampleRow:
    study_id: str
    slice_idx: int
    img_path: str
    body_part: Optional[str] = None
    lesion_type: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    split: Optional[str] = None


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "pgr", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)
    return logger


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_tensor_and_norm(x3: np.ndarray, size: int = 224) -> torch.Tensor:
    """HxWx3 (uint8[0..255] or float[0..1]) → (1,3,size,size), CLIP-normalized."""
    if x3.ndim != 3 or x3.shape[2] != 3:
        raise ValueError("expected (H,W,3)")
    t = torch.as_tensor(x3)
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    elif t.dtype.is_floating_point:
        t = t.float().clamp(0.0, 1.0)
    else:
        raise TypeError(f"unsupported dtype: {t.dtype}")
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    mean = _CLIP_MEAN.to(t.device)
    std = _CLIP_STD.to(t.device)
    return (t - mean) / std


def l2_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)



