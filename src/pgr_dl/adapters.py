# prepare_slice(), encode_phrase()
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn.functional as F
from pgr.utils import to_tensor_and_norm
from . import io_deeplesion as io
from . import windowing, phrases


def prepare_slice(row, size: Optional[int] = None) -> torch.Tensor:
    """Load & preprocess one slice → (1,3,H,W) CLIP-normalized."""
    img = io.load_slice(row.img_path)
    x3 = windowing.ct3ch(img)
    return to_tensor_and_norm(x3, size=size or 224)


def encode_phrase(encoder, phrase: str) -> torch.Tensor:
    """Encode a phrase via prompt templates → (1,D) L2-normalized embedding."""
    p = phrase.strip()
    if not p:
        raise ValueError("phrase must be non-empty")
    texts: List[str] = phrases.make_prompts(p)  # e.g., templated variants
    t = encoder.encode_texts(texts)             # (T,D) float32, L2-normalized
    if not isinstance(t, torch.Tensor) or t.ndim != 2:
        raise TypeError("encode_texts must return a (T,D) torch.Tensor")
    pooled = t.mean(dim=0, keepdim=True)        # (1,D)
    pooled = F.normalize(pooled.float(), dim=-1)
    return pooled.cpu()
