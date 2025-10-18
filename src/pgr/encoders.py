from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F

try:
    import open_clip  # type: ignore
except Exception:
    open_clip = None


def _normalize_model_name(name: str) -> str:
    # OpenCLIP uses dashes, not slashes (e.g., "ViT-B-16")
    return name.replace("ViT-B/16", "ViT-B-16")


class ClipEncoder:
    """OpenCLIP wrapper producing L2-normalized (image,text) embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        device: str | torch.device = "cpu",
        pretrained: str = "openai",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if open_clip is None:
            raise ImportError("Install open-clip-torch")
        self.model_name = _normalize_model_name(model_name)
        self.device = torch.device(device)
        self.pretrained = pretrained
        self.dtype = dtype
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self._embed_dim = self._infer_embed_dim()

    def _infer_embed_dim(self) -> int:
        with torch.no_grad():
            toks = self.tokenizer(["x"]).to(self.device)
            t = self.model.encode_text(toks)
        if t.ndim != 2:
            raise RuntimeError("encode_text must return (B, D)")
        return int(t.shape[-1])

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs (B,3,H,W) → (B,D) float32, L2-normalized."""
        if imgs.ndim != 4 or imgs.shape[1] != 3:
            raise ValueError("imgs must be (B,3,H,W)")
        imgs = imgs.to(self.device, non_blocking=True)
        use_amp = self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                feats = self.model.encode_image(imgs)
        else:
            feats = self.model.encode_image(imgs)
        if feats.ndim != 2:
            raise RuntimeError("encode_image must return (B, D)")
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """texts list[str] → (T,D) float32, L2-normalized."""
        if not texts:
            raise ValueError("texts must be non-empty")
        toks = self.tokenizer(texts).to(self.device, non_blocking=True)
        use_amp = self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                feats = self.model.encode_text(toks)
        else:
            feats = self.model.encode_text(toks)
        if feats.ndim != 2:
            raise RuntimeError("encode_text must return (B, D)")
        return F.normalize(feats.float(), dim=-1)

    # alias to match apps
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        return self.encode_texts(texts)
