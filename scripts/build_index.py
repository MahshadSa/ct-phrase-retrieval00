#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# repo import path
this_dir = Path(__file__).resolve().parent
repo_root = this_dir.parent
sys.path.append(str(repo_root / "src"))

# minimal import guards (helpful on Kaggle)
try:
    import faiss  # noqa: F401
except Exception as e:
    raise RuntimeError("FAISS not found. On Kaggle: pip install faiss-cpu==1.8.0.post1") from e
try:
    import open_clip  # noqa: F401
except Exception as e:
    raise RuntimeError("open-clip-torch not found. pip install open-clip-torch==2.26.1") from e

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
    csv_name = cfg["paths"].get("csv_name", "DL_info.csv")
    csv_path = data_root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    res_dir = Path(cfg["results"]["dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    meta = io.load_metadata(str(data_root), csv_name=csv_name)

    split = (cfg.get("data", {}) or {}).get("split", None)
    if split and "split" in meta.columns:
        meta = meta[meta["split"].astype(str).str.lower() == str(split).lower()].reset_index(drop=True)

    max_samples = (cfg.get("data", {}) or {}).get("max_samples", None)
    if max_samples:
        n = min(int(max_samples), len(meta))
        meta = meta.sample(n=n, random_state=seed).reset_index(drop=True)

    if len(meta) == 0:
        raise RuntimeError("No rows selected after filters; remove 'split' or increase 'max_samples'.")

    required = {"study_id", "slice_idx"}
    if not required.issubset(set(meta.columns)):
        raise KeyError(f"Missing required columns in metadata: {sorted(required - set(meta.columns))}")

    device = get_device(args.device or cfg.get("runtime", {}).get("device"))
    enc = encoders.ClipEncoder(
        model_name=cfg["encoder"]["name"],
        pretrained=cfg["encoder"].get("pretrained", "openai"),
        device=str(device),
    )
    dim = enc.embed_dim
    print(f"[data] rows={len(meta):,} csv={csv_name}")
    print(f"[enc ] {cfg['encoder']['name']} ({cfg['encoder'].get('pretrained','openai')}) → dim={dim} device={device}")

    batch = int(cfg.get("batch", 64))
    vecs: list[np.ndarray] = []
    ids_rows: list[dict] = []

    for i in tqdm(range(0, len(meta), batch), desc="embedding", unit="rows"):
        sub = meta.iloc[i:i + batch]
        xs = []
        for _, r in sub.iterrows():
            img = io.load_slice(r.img_path)
            x3 = windowing.ct3ch(img)
            xs.append(to_tensor_and_norm(x3))
            ids_rows.append({
                "study_id": str(r.study_id),
                "slice_idx": int(r.slice_idx),
                **({k: r[k]} if (k := "img_path") in sub.columns else {}),
            })
        X = torch.cat(xs, dim=0).to(device, non_blocking=True)
        V = enc.encode_images(X).cpu().numpy().astype("float32")
        vecs.append(V)

    embs = np.vstack(vecs).astype("float32")
    print(f"[emb] shape={embs.shape}")

    np.save(res_dir / "image_embs.npy", embs)
    ids_df = pd.DataFrame(ids_rows)
    # append extra known columns if present in meta
    for col in ("img_path", "body_part", "lesion_type"):
        if col in meta.columns and col not in ids_df.columns:
            ids_df[col] = meta.loc[ids_df.index, col].values
    ids_df.to_parquet(res_dir / "ids.parquet", index=False)

    fa = index.FaissIndex(
        dim=embs.shape[1],
        kind=cfg.get("index", {}).get("kind", "flat"),
        metric=cfg.get("index", {}).get("metric", "ip"),
    )
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
        "config_path": str(Path(args.config).resolve()),
        "encoder": cfg.get("encoder", {}),
        "index_cfg": cfg.get("index", {}),
    }
    with open(res_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"paths": manifest["paths"], "manifest": manifest}, f, indent=2, default=str)

    print("[done] image_embs.npy, ids.parquet, index.faiss, manifest.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="cuda | cpu | None(auto)")
    main(ap.parse_args())
