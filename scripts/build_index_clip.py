# scripts/build_index_clip.py
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import yaml

from pgr.index import build_index, save_index

# ---- OpenCLIP wrapper (offline-friendly) -------------------------------------
try:
    import open_clip
except Exception as e:
    open_clip = None

def create_clip(backbone: str = "ViT-B-16",
                pretrained: Optional[str] = "openai",
                weights_path: Optional[str] = None,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    if open_clip is None:
        raise ImportError("open-clip-torch not installed. `pip install open-clip-torch`")
    # Always create model w/o weights first, then load from chosen source
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=None)
    model.to(device).eval()

    if weights_path:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"CLIP weights not found at {p}.\n"
                                    "Upload weights as a Kaggle Dataset and set paths.weights_path in YAML.")
        sd = torch.load(p, map_location="cpu")
        state = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected: print(f"[warn] unexpected keys: {len(unexpected)}")
        if missing:   print(f"[warn] missing keys: {len(missing)}")
    else:
        # If internet is ON, this will fetch weights; if OFF, it will error clearly.
        if pretrained:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
                model.to(device).eval()
            except Exception as e:
                raise RuntimeError(
                    "Could not download pretrained weights (internet likely OFF).\n"
                    "Either enable internet or set paths.weights_path in YAML to a local .pt/.bin file."
                ) from e
        else:
            raise RuntimeError(
                "No weights available. Provide paths.weights_path in YAML or set pretrained to a valid tag."
            )
    return model, preprocess

# ---- Image list discovery -----------------------------------------------------
def discover_images(data_root: Path, meta: Optional[pd.DataFrame]) -> List[Path]:
    if meta is not None:
        for col in ["File_name", "filename", "image", "path"]:
            if col in meta.columns:
                paths = [data_root / str(x) for x in meta[col].tolist()]
                # Prefer only existing files
                exist = [p for p in paths if p.exists()]
                if exist:
                    return exist
    # Fallback: glob
    pats = ["*.png", "*.jpg", "*.jpeg", "**/*.png", "**/*.jpg", "**/*.jpeg"]
    seen = set()
    found: List[Path] = []
    for pat in pats:
        for p in data_root.glob(pat):
            if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                if p not in seen:
                    seen.add(p); found.append(p)
    if not found:
        raise FileNotFoundError(
            f"No images found under {data_root}. Supply DL_info.csv with a File_name column or place PNG/JPG files."
        )
    return found

# ---- Encoding loop ------------------------------------------------------------
@torch.no_grad()
def encode_images(model, preprocess, paths: List[Path], batch_size: int, device: str) -> np.ndarray:
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="encoding", ncols=88):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                # robust open: skip broken images
                continue
            imgs.append(preprocess(im))
        if not imgs:
            # all failed in batch; skip
            continue
        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
        z = model.encode_image(x)
        z = z.float()
        # L2-normalise for cosine/IP
        z = torch.nn.functional.normalize(z, dim=1)
        embs.append(z.cpu().numpy())
    if not embs:
        raise RuntimeError("No embeddings produced; check image reading / preprocessing.")
    return np.concatenate(embs, axis=0)

def main():
    ap = argparse.ArgumentParser(description="Build CLIP embeddings + FAISS index (Kaggle-ready).")
    ap.add_argument("--config", required=True, help="YAML config (e.g., configs/deeplesion_kaggle.yaml)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    paths_cfg = cfg.get("paths", {})
    data_root = Path(paths_cfg.get("data_root", "/kaggle/input/nih-deeplesion-subset"))
    csv_name  = paths_cfg.get("csv_name", "DL_info.csv")
    results   = Path(paths_cfg.get("results_dir", "/kaggle/working/results/kaggle_clip"))
    weights_path = paths_cfg.get("weights_path", None)  # optional local .pt/.bin

    enc_cfg = cfg.get("encoder", {})
    backbone   = enc_cfg.get("backbone", "ViT-B-16")
    pretrained = enc_cfg.get("pretrained", "openai")  # ignored if weights_path is set
    batch_size = int(enc_cfg.get("batch_size", 64))
    limit_n    = enc_cfg.get("limit_n", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results.mkdir(parents=True, exist_ok=True)

    # Load metadata (optional)
    meta = None
    csv_path = data_root / csv_name
    if csv_path.exists():
        try:
            meta = pd.read_csv(csv_path)
            print(f"[i] Loaded metadata: {csv_path} shape={meta.shape}")
        except Exception as e:
            print(f"[warn] Could not read CSV at {csv_path}: {e}")

    # Discover images
    img_paths = discover_images(data_root, meta)
    if limit_n is not None:
        img_paths = img_paths[:int(limit_n)]
    if len(img_paths) == 0:
        raise RuntimeError("No images to process after discovery/limit.")

    print(f"[i] Images: {len(img_paths)} (showing 3) ->",
          [str(p.relative_to(data_root)) for p in img_paths[:3]])

    # Create model
    model, preprocess = create_clip(backbone=backbone,
                                    pretrained=pretrained,
                                    weights_path=weights_path,
                                    device=device)

    # Encode
    image_embs = encode_images(model, preprocess, img_paths, batch_size=batch_size, device=device)
    print(f"[i] Embeddings: {image_embs.shape}")

    # IDs table
    ids = pd.DataFrame({
        "row_id": np.arange(len(img_paths)),
        "path":   [str(p.relative_to(data_root)) for p in img_paths]
    })

    # FAISS index (cosine via IP on L2-normed vectors)
    index = build_index(image_embs, metric="ip", kind="flat")
    save_index(index, results / "index.faiss")
    np.save(results / "image_embs.npy", image_embs)
    ids.to_parquet(results / "ids.parquet", index=False)

    manifest = {
        "dim": int(image_embs.shape[1]),
        "n": int(image_embs.shape[0]),
        "metric": "ip",
        "kind": "flat",
        "data_root": str(data_root),
        "csv": csv_name if csv_path.exists() else None,
        "backbone": backbone,
        "pretrained": pretrained if weights_path is None else f"local:{Path(weights_path).name}",
        "batch_size": batch_size
    }
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[ok] Wrote artifacts → {results}")
    for f in ["index.faiss","image_embs.npy","ids.parquet","manifest.json"]:
        print(" -", f)

if __name__ == "__main__":
    main()
