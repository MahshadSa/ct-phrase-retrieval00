from __future__ import annotations
from pathlib import Path
import torch

try:
    import open_clip
except Exception:
    open_clip = None

def create_clip(backbone: str = "ViT-B-16", pretrained: str | None = "openai",
                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                weights_path: str | None = None):
    if open_clip is None:
        raise ImportError("open-clip-torch missing. `pip install open-clip-torch`")

    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=None)
    model.to(device).eval()

    if weights_path:
        p = Path(weights_path)
        sd = torch.load(p, map_location="cpu")
        state = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected: print(f"[warn] unexpected keys: {unexpected}")
        if missing: print(f"[warn] missing keys: {missing}")
    elif pretrained:
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        model.to(device).eval()

    return model, preprocess
