from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from pgr.index import load_index, search

def _require(path: Path, msg: str):
    if not path.exists(): raise FileNotFoundError(f"{msg}: {path}")

def main():
    ap = argparse.ArgumentParser(description="Phrase→top-k search (Kaggle-safe)")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--phrase", required=True)
    ap.add_argument("-k", type=int, default=9)
    ap.add_argument("--save_panel", action="store_true")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    idx_p, embs_p, ids_p = rd/"index.faiss", rd/"image_embs.npy", rd/"ids.parquet"
    _require(idx_p, "Missing FAISS index (run build_index.py)")
    _require(embs_p, "Missing embeddings (run build_index.py)")
    _require(ids_p, "Missing ids parquet (run build_index.py)")

    index = load_index(idx_p)
    dim = np.load(embs_p).shape[1]
    q = np.zeros((1, dim), dtype="float32")
    q[0, hash(args.phrase) % dim] = 1.0  # deterministic toy vector

    D, I = search(index, q, k=args.k)
    ids = pd.read_parquet(ids_p)
    hits = ids.iloc[I[0]].assign(score=D[0])
    print(hits.to_string(index=False))

    if args.save_panel:
        out = rd / f"panel_{args.phrase.replace(' ','_')}.txt"
        out.write_text(hits.to_csv(index=False))
        print(f"[ok] Saved panel list → {out}")

if __name__ == "__main__":
    main()
