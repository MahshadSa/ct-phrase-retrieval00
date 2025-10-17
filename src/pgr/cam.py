from __future__ import annotations
import numpy as np
import torch
from torch import Tensor


def _l2(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def _to_np01(x: Tensor) -> np.ndarray:
    x = x.float().clamp(0, 1)
    return x.cpu().numpy()


def gradcam_vit(img_tensor: Tensor, phrase_vec: Tensor, model, *, use_cosine: bool = True) -> np.ndarray:
    """Minimal CAM via input-grad saliency on CLIP-style encoders. Returns (H,W) in [0,1]."""
    if img_tensor.ndim != 4 or img_tensor.shape[0] != 1:
        raise ValueError("img_tensor must be (1,3,H,W)")
    if phrase_vec.ndim != 2 or phrase_vec.shape[0] != 1:
        raise ValueError("phrase_vec must be (1,D)")

    device = next(model.parameters()).device if hasattr(model, "parameters") else img_tensor.device
    img = img_tensor.to(device).clone().detach().requires_grad_(True)
    txt = phrase_vec.to(device)

    was_training = getattr(model, "training", False)
    model.eval()
    with torch.enable_grad():
        with torch.cuda.amp.autocast(enabled=False):
            img_emb: Tensor = model(img)
            if img_emb.ndim != 2:
                raise RuntimeError("image encoder must return (B,D)")
            if use_cosine:
                img_emb = _l2(img_emb)
                txt = _l2(txt)
            score: Tensor = (img_emb * txt).sum()
        if img.grad is not None:
            img.grad.zero_()
        score.backward()

    sal = img.grad.detach().abs().mean(dim=1)[0]  # (H,W)
    mn, mx = float(sal.min()), float(sal.max())
    sal = (sal - mn) / (mx - mn) if mx > mn else torch.zeros_like(sal)

    if was_training and hasattr(model, "train"):
        model.train()

    return _to_np01(sal)
