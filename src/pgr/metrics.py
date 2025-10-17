from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

IdLike = Union[int, str]
Box = Tuple[int, int, int, int]


def recall_at_k(
    results: Sequence[Sequence[IdLike]],
    positives: Sequence[Iterable[IdLike]],
    ks: Iterable[int] = (1, 5, 10),
    *,
    ignore_empty: bool = True,
) -> Dict[int, float]:
    """Recall@k over queries; binary relevance."""
    if len(results) != len(positives):
        raise ValueError("length mismatch")
    ks = sorted({int(k) for k in ks if k > 0})
    if not ks:
        raise ValueError("ks must be positive")
    Q = len(results)
    pos_sets: List[Set[IdLike]] = [set(p) for p in positives]
    denom = 0
    hits = {k: 0 for k in ks}
    for q in range(Q):
        pos = pos_sets[q]
        if not pos:
            if ignore_empty:
                continue
            denom += 1
            continue
        denom += 1
        row = results[q]
        if len(row) < max(ks):
            raise ValueError(f"row {q} shorter than max(k)")
        for k in ks:
            if any(r in pos for r in row[:k]):
                hits[k] += 1
    if denom == 0:
        return {k: float("nan") for k in ks}
    return {k: hits[k] / denom for k in ks}


def binary_hits_at_k(
    results: Sequence[Sequence[IdLike]],
    positives: Sequence[Iterable[IdLike]],
    k: int,
    *,
    ignore_empty: bool = True,
) -> np.ndarray:
    """Per-query hit flag for top-k."""
    if len(results) != len(positives):
        raise ValueError("length mismatch")
    if k <= 0:
        raise ValueError("k must be positive")
    Q = len(results)
    out = np.zeros(Q, dtype=bool)
    for q in range(Q):
        pos = set(positives[q])
        if not pos and ignore_empty:
            out[q] = False
            continue
        row = results[q]
        if len(row) < k:
            raise ValueError(f"row {q} shorter than k")
        out[q] = any(r in pos for r in row[:k])
    return out


def iou(box_a: Optional[Box], box_b: Optional[Box]) -> float:
    """IoU for (x1,y1,x2,y2); 0 if invalid."""
    if not _valid_box(box_a) or not _valid_box(box_b):
        return 0.0
    ax1, ay1, ax2, ay2 = box_a  # type: ignore[misc]
    bx1, by1, bx2, by2 = box_b  # type: ignore[misc]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cam_peak(cam: np.ndarray) -> Tuple[int, int, float]:
    """Return (x,y,value) of the peak."""
    if cam.ndim != 2:
        raise ValueError("cam must be 2D")
    cam_f = cam.astype(np.float32, copy=False)
    idx = int(np.argmax(cam_f))
    H, W = cam_f.shape
    y, x = divmod(idx, W)
    return x, y, float(cam_f[y, x])


def peak_in_box(point: Tuple[int, int], box: Optional[Box]) -> bool:
    """True if (x,y) lies in box [x1,x2)×[y1,y2)."""
    if not _valid_box(box):
        return False
    x, y = point
    x1, y1, x2, y2 = box  # type: ignore[misc]
    return (x1 <= x < x2) and (y1 <= y < y2)


def peak_in_box_rate(cams: Sequence[np.ndarray], boxes: Sequence[Optional[Box]]) -> float:
    """Fraction of cams whose peak falls inside the box; NaN if none evaluable."""
    if len(cams) != len(boxes):
        raise ValueError("length mismatch")
    total = hits = 0
    for cam, box in zip(cams, boxes):
        if not _valid_box(box):
            continue
        total += 1
        x, y, _ = cam_peak(cam)
        if peak_in_box((x, y), box):
            hits += 1
    return hits / total if total else float("nan")


def _valid_box(box: Optional[Box]) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return (x2 > x1) and (y2 > y1)
