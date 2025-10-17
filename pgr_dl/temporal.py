# tiny longitudinal toy (pairing, features, probe)
# pgr_dl/temporal.py
from __future__ import annotations
from typing import Iterable, List, Literal, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from . import io_deeplesion as io
from . import windowing
from pgr.viz import overlay_cam


Reduce = Literal["max", "mean"]


def _valid_window(radius: int) -> int:
    r = int(radius)
    if r < 0:
        raise ValueError("radius must be >= 0")
    return r


def temporal_smooth_scores(
    df: pd.DataFrame,
    *,
    radius: int = 1,
    reduce: Reduce = "max",
    by: Sequence[str] = ("query_phrase", "study_id"),
) -> pd.DataFrame:
    """
    Smooth slice-level scores over neighboring slices.

    Expects columns: by..., 'slice_idx', 'score'.
    Returns df with a new column 'score_smooth'.
    """
    for col in list(by) + ["slice_idx", "score"]:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    r = _valid_window(radius)
    if r == 0:
        out = df.copy()
        out["score_smooth"] = out["score"].astype(np.float32)
        return out

    def _smooth_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("slice_idx").reset_index(drop=True)
        idx = g["slice_idx"].to_numpy()
        s = g["score"].to_numpy().astype(np.float32)
        sm = np.empty_like(s, dtype=np.float32)
        for i in range(len(s)):
            lo = idx[i] - r
            hi = idx[i] + r
            m = (idx >= lo) & (idx <= hi)
            if reduce == "max":
                sm[i] = float(s[m].max())
            elif reduce == "mean":
                sm[i] = float(s[m].mean())
            else:
                raise ValueError("reduce must be 'max' or 'mean'")
        g["score_smooth"] = sm
        return g

    parts = []
    for _, g in df.groupby(list(by), sort=False):
        parts.append(_smooth_group(g))
    return pd.concat(parts, axis=0, ignore_index=True)


def best_slice_per_study(
    df: pd.DataFrame,
    *,
    use_smooth: bool = True,
    k_per_query: int = 10,
) -> pd.DataFrame:
    """
    Collapse to one row per (query_phrase, study_id) using 'score_smooth' if present.

    Returns rows ranked within each query (descending).
    """
    cols = {"query_phrase", "study_id", "slice_idx", "score"}
    if not cols.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(cols - set(df.columns))}")
    score_col = "score_smooth" if use_smooth and "score_smooth" in df.columns else "score"

    df = df.sort_values(["query_phrase", "study_id", score_col], ascending=[True, True, False])
    best = df.groupby(["query_phrase", "study_id"], as_index=False, sort=False).head(1)
    best = best.sort_values(["query_phrase", score_col], ascending=[True, False])
    if k_per_query is not None:
        best = best.groupby("query_phrase", as_index=False, sort=False).head(int(k_per_query))
    return best.reset_index(drop=True)


def neighbor_indices(center_idx: int, radius: int) -> List[int]:
    """Return relative window [center - r, ..., center + r]."""
    r = _valid_window(radius)
    return list(range(center_idx - r, center_idx + r + 1))


def load_cine(
    meta: pd.DataFrame,
    study_id: str,
    anchor_slice: int,
    *,
    radius: int = 2,
) -> List[np.ndarray]:
    """
    Load a small cine stack (as 3-channel uint8) around an anchor slice.

    Expects meta to contain rows with ('study_id','slice_idx','img_path').
    """
    required = {"study_id", "slice_idx", "img_path"}
    if not required.issubset(meta.columns):
        raise ValueError(f"meta missing: {sorted(required - set(meta.columns))}")

    sub = meta[meta["study_id"] == str(study_id)]
    if sub.empty:
        return []

    wanted = set(neighbor_indices(anchor_slice, radius))
    frames: List[np.ndarray] = []
    for _, r in sub.iterrows():
        sidx = int(r["slice_idx"])
        if sidx in wanted:
            img = io.load_slice(str(r["img_path"]))
            frames.append(windowing.ct3ch(img))  # HxWx3 uint8
    frames.sort(key=lambda x: x.shape[0] * 10_000 + x.shape[1])  # stable op; not critical
    # Ensure input order by slice index
    frames = [f for _, f in sorted(zip([int(r["slice_idx"]) for _, r in sub.iterrows() if int(r["slice_idx"]) in wanted], frames))]
    return frames


def overlay_cine_with_cam(
    cine_rgb: Sequence[np.ndarray],
    cams: Optional[Sequence[np.ndarray]] = None,
    *,
    alpha: float = 0.5,
) -> List[np.ndarray]:
    """
    Optionally overlay per-slice CAMs onto a cine stack.

    If cams is None or shorter than cine, missing entries are skipped (no overlay).
    """
    out: List[np.ndarray] = []
    N = len(cine_rgb)
    cams = list(cams) if cams is not None else [None] * N
    for i in range(N):
        rgb = cine_rgb[i]
        cam = cams[i] if i < len(cams) else None
        if cam is None:
            out.append(rgb)
        else:
            pil = overlay_cam(rgb, cam, alpha=alpha)
            out.append(np.asarray(pil, dtype=np.uint8))
    return out


def study_rank_table(
    df: pd.DataFrame,
    *,
    radius: int = 1,
    reduce: Reduce = "max",
    k_per_query: int = 10,
) -> pd.DataFrame:
    """
    Convenience: smooth → pick best slice per study → return ranked table.

    Expects columns: 'query_phrase','study_id','slice_idx','score'.
    """
    sm = temporal_smooth_scores(df, radius=radius, reduce=reduce)
    return best_slice_per_study(sm, use_smooth=True, k_per_query=k_per_query)
