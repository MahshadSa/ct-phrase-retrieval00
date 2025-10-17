from __future__ import annotations
from typing import Tuple, Optional, Literal
import os
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore[assignment]

Metric = Literal["ip", "l2"]
Kind = Literal["flat", "ivf"]


def _require_faiss() -> None:
    if faiss is None:
        raise ImportError("Install faiss-cpu (pip) or faiss-gpu")


def _faiss_metric(metric: Metric) -> int:
    if metric == "ip":
        return faiss.METRIC_INNER_PRODUCT
    if metric == "l2":
        return faiss.METRIC_L2
    raise ValueError("metric must be 'ip' or 'l2'")


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


class FaissIndex:
    """Flat/IVF FAISS wrapper. For metric='ip' inputs are L2-normalized (cosine)."""

    def __init__(
        self,
        dim: int,
        kind: Kind = "flat",
        metric: Metric = "ip",
        nlist: int = 4096,
        nprobe: int = 16,
    ) -> None:
        _require_faiss()
        self.dim = int(dim)
        self.kind = kind
        self.metric = metric
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self._index = self._build()
        self._ext_ids: Optional[np.ndarray] = None

    def _build(self):
        m = _faiss_metric(self.metric)
        if self.kind == "flat":
            return faiss.IndexFlatIP(self.dim) if m == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dim)
        if self.kind == "ivf":
            q = faiss.IndexFlatIP(self.dim) if m == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dim)
            idx = faiss.IndexIVFFlat(q, self.dim, self.nlist, m)
            idx.nprobe = self.nprobe
            return idx
        raise ValueError("kind must be 'flat' or 'ivf'")

    def set_nprobe(self, nprobe: int) -> None:
        if isinstance(self._index, faiss.IndexIVF):
            self._index.nprobe = int(nprobe)
            self.nprobe = int(nprobe)

    def add(self, X: np.ndarray, ids: Optional[np.ndarray] = None, train_subset: int = 100_000, seed: int = 42) -> None:
        """Add (N,D) to index; auto-trains IVF. IP path L2-normalizes."""
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"X must be (N,{self.dim})")
        X = X.astype(np.float32, copy=False)
        if self.metric == "ip":
            X = _l2norm(X)

        N = int(X.shape[0])
        if ids is not None:
            ids = np.asarray(ids)
            if ids.shape[0] != N:
                raise ValueError("ids length must match N")

        if isinstance(self._index, faiss.IndexIVF) and not self._index.is_trained:
            take = min(train_subset, N)
            rs = np.random.RandomState(seed)
            sel = rs.choice(N, size=take, replace=False)
            self._index.train(X[sel])
            self._index.nprobe = self.nprobe

        if ids is not None and np.issubdtype(ids.dtype, np.integer):
            self._index.add_with_ids(X, ids.astype(np.int64, copy=False))
            self._ext_ids = None
        else:
            self._index.add(X)
            self._ext_ids = ids if ids is not None else None

    def search(self, Q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Q (B,D) → scores (B,k), I (B,k), ext (B,k|None). IP path L2-normalizes."""
        if Q.ndim != 2 or Q.shape[1] != self.dim:
            raise ValueError(f"Q must be (B,{self.dim})")
        Q = Q.astype(np.float32, copy=False)
        if self.metric == "ip":
            Q = _l2norm(Q)
        if isinstance(self._index, faiss.IndexIVF):
            self._index.nprobe = self.nprobe
        scores, I = self._index.search(Q, int(k))
        ext: Optional[np.ndarray] = None
        if self._ext_ids is not None:
            ext = np.full(I.shape, None, dtype=object)
            valid = I >= 0
            if valid.any():
                ext[valid] = self._ext_ids[I[valid]]
        return scores, I, ext

    def save(self, path: str | os.PathLike, ext_ids_path: Optional[str | os.PathLike] = None) -> None:
        """Save index to path; external ids to path or path+'.ext_ids.npy'."""
        _require_faiss()
        path = str(path)
        idx = self._index
        if hasattr(faiss, "get_num_gpus") and "Gpu" in type(idx).__name__:
            idx = faiss.index_gpu_to_cpu(idx)  # type: ignore[attr-defined]
        faiss.write_index(idx, path)
        if self._ext_ids is not None:
            p = str(ext_ids_path) if ext_ids_path is not None else path + ".ext_ids.npy"
            np.save(p, self._ext_ids, allow_pickle=(self._ext_ids.dtype == object))

    @classmethod
    def load(cls, path: str | os.PathLike, *, ext_ids_path: Optional[str | os.PathLike] = None, nprobe: Optional[int] = None) -> "FaissIndex":
        """Load index and optional external ids."""
        _require_faiss()
        path = str(path)
        idx = faiss.read_index(path)
        dim = int(idx.d)
        metric: Metric = "ip" if idx.metric_type == faiss.METRIC_INNER_PRODUCT else "l2"
        kind: Kind = "ivf" if isinstance(idx, faiss.IndexIVF) else "flat"
        obj = cls(dim=dim, kind=kind, metric=metric)
        obj._index = idx
        if nprobe is not None and isinstance(obj._index, faiss.IndexIVF):
            obj.set_nprobe(int(nprobe))
        p = str(ext_ids_path) if ext_ids_path is not None else path + ".ext_ids.npy"
        obj._ext_ids = np.load(p, allow_pickle=True) if os.path.exists(p) else None
        return obj
