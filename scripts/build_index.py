#!/usr/bin/env python
"""Build embeddings + FAISS index for a DeepLesion Kaggle subset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

this_dir = Path(__file__).resolve().parent
repo_root = this_dir.parent
sys.path.append(str(repo_root / "src"))

from pgr import encoders, index  # noqa: E402
from pgr.utils import to_tensor_and_norm, get_device, seed_everything  # noqa: E402
from pgr_dl import io_deeplesion as io, windowing  # noqa: E402


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    data_root = Path(cfg["paths"]["data_root"])
    res_dir = Path(cfg["paths"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    meta = io.load_metadata(str(data_root))
    split = cfg.get("data", {}).get("split")
    if split:
        meta = meta[meta["split"] == split].copy()
    max_samples = cfg.get("data", {}).get("max_samples")
    if max_samples:
        meta = meta.sample(min(int(max_samples), len(meta)), random_state=seed).reset_index(drop=True)
    if len(meta) == 0:
        print("[warn] no rows after filtering")
        return

    device = get_device(args.device)
    enc = encoders.ClipEncoder(
        model_name=cfg["model"]["image_encoder"],
        pretrained=cfg["model"].get("pretrained", "openai"),
        device=str(device),
    )
    dim = enc.embed_dim
    print(f"[info] rows={len(meta)}  encoder={cfg['model']['image_encoder']}  dim={dim}  device={device}")

    batch = int(cfg.get("batch", 64))
    vecs: list[np.ndarray] = []
    ids: list[tuple[str, int]] = []

    for i in tqdm(range(0, len(meta), batch), desc="embedding"):
        sub = meta.iloc[i : i + batch]
        xs = []
        for _, r in sub.iterrows():
            img = io.load_slice(r.img_path)
            x3 = windowing.ct3ch(img)
            xs.append(to_tensor_and_norm(x3))
            ids.append((str(r.study_id), int(r.slice_idx)))
        X = torch.cat(xs, dim=0).to(device, non_blocking=True)
        V = enc.encode_images(X).cpu().numpy().astype("float32")
        vecs.append(V)

    embs = np.vstack(vecs).astype("float32")
    print(f"[info] embeddings: {embs.shape}")

    np.save(res_dir / "image_embs.npy", embs)
    ids_df = pd.DataFrame(ids, columns=["study_id", "slice_idx"])
    for col in ["img_path", "body_part", "lesion_type"]:
        if col in meta.columns:
            ids_df[col] = meta[col].values
    ids_df.to_parquet(res_dir / "ids.parquet", index=False)

    fa = index.FaissIndex(dim=embs.shape[1], kind=cfg.get("index", {}).get("kind", "flat"), metric=cfg.get("index", {}).get("metric", "ip"))
    fa.add(embs)
    fa.save(res_dir / "index.faiss")

    manifest = {
        "counts": {"rows": int(len(meta)), "dim": int(embs.shape[1])},
        "paths": {
            "embeddings": str(res_dir / "image_embs.npy"),
            "ids": str(res_dir / "ids.parquet"),
            "index": str(res_dir / "index.faiss"),
        },
        "seed": seed,
    }
    with open(res_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg, "manifest": manifest}, f, indent=2, default=str)
    print("[done] image_embs.npy, ids.parquet, index.faiss, manifest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu | None for auto")
    main(ap.parse_args())
