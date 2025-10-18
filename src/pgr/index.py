from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

Metric = Literal["ip", "l2"]
Kind = Literal["flat", "ivf"]

def _require_faiss():
    if faiss is None:
        raise ImportError("faiss is required (pip install faiss-cpu).")

def build_index(x: np.ndarray, metric: Metric = "ip", kind: Kind = "flat", nlist: int = 512):
    _require_faiss()
    x = np.ascontiguousarray(x.astype("float32"))
    dim = x.shape[1]
    if metric == "ip":
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x = x / norms
        if kind == "flat":
            index = faiss.IndexFlatIP(dim)
        else:
            quant = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(x)
    else:
        if kind == "flat":
            index = faiss.IndexFlatL2(dim)
        else:
            quant = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
            index.train(x)
    index.add(x)
    return index

def search(index, q: np.ndarray, k: int = 9):
    _require_faiss()
    q = np.ascontiguousarray(q.astype("float32"))
    return index.search(q, k)

def save_index(index, path: Path):
    _require_faiss()
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def load_index(path: Path):
    _require_faiss()
    if not path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {path}")
    return faiss.read_index(str(path))
