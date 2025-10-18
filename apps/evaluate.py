from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from pgr.index import load_index, search

def main():
    ap = argparse.ArgumentParser(description="Toy Recall@k")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    idx_p, embs_p = rd/"index.faiss", rd/"image_embs.npy"
    if not idx_p.exists() or not embs_p.exists():
        raise FileNotFoundError("Run scripts/build_index.py first.")

    index = load_index(idx_p)
    X = np.load(embs_p)
    N = min(100, X.shape[0])
    D, I = search(index, X[:N], k=args.k)
    recall = (np.arange(N)[:,None] == I).any(axis=1).mean()
    print(f"Recall@{args.k} on {N} queries: {recall:.3f}")

if __name__ == "__main__":
    main()
