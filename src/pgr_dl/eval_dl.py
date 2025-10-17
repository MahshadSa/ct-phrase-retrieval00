# dataset-specific matching rules
from __future__ import annotations
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from .phrases import tags_match_phrase


def recall_at_k(results: pd.DataFrame, k: int = 5) -> float:
    """
    Recall@k from a ranked results DataFrame.

    Expects columns:
      - 'query_phrase' : str
      - 'is_positive'  : bool (per hit)
    Optional rank hints (used if present):
      - 'score' (higher is better) or 'rank' (lower is better)

    Returns:
      Recall@k in [0,1], or NaN if no queries.
    """
    if "query_phrase" not in results or "is_positive" not in results:
        raise ValueError("results must include 'query_phrase' and 'is_positive' columns")

    # Sort within each query if rank hints exist; otherwise keep current order
    if "score" in results.columns:
        results = results.sort_values(["query_phrase", "score"], ascending=[True, False])
    elif "rank" in results.columns:
        results = results.sort_values(["query_phrase", "rank"], ascending=[True, True])

    hit = []
    for _, group in results.groupby("query_phrase", sort=False):
        topk = group.head(int(k))
        hit.append(bool(topk["is_positive"].any()))
    if not hit:
        return float("nan")
    return float(np.mean(hit))


def cam_peak_in_box(cam: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> bool:
    """
    True if CAM peak (argmax) lies inside bbox [x1,x2)×[y1,y2).

    Args:
      cam  : (H,W) heatmap
      bbox : (x1,y1,x2,y2) or None
    """
    if bbox is None:
        return False
    if cam.ndim != 2:
        raise ValueError("cam must be 2D (H,W)")
    y, x = np.unravel_index(int(np.argmax(cam.astype(np.float32, copy=False))), cam.shape)
    x1, y1, x2, y2 = bbox
    return (x1 <= x < x2) and (y1 <= y < y2)


def label_row_match(row: pd.Series, phrase: str) -> bool:
    """Dataset-specific tag→phrase match using (body_part, lesion_type)."""
    return tags_match_phrase(str(row.get("body_part", "")), str(row.get("lesion_type", "")), phrase)
def aggregate_best_slice_per_study(results: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the best slice per (query_phrase, study_id).

    Uses 'score' (higher is better) if present, else 'rank' (lower is better),
    else keeps the first occurrence in input order.
    """
    required = {"query_phrase", "study_id"}
    if not required.issubset(results.columns):
        missing = required - set(results.columns)
        raise ValueError(f"missing columns: {sorted(missing)}")

    df = results
    if "score" in df.columns:
        df = df.sort_values(["query_phrase", "study_id", "score"], ascending=[True, True, False])
    elif "rank" in df.columns:
        df = df.sort_values(["query_phrase", "study_id", "rank"], ascending=[True, True, True])

    # take first row within each (query_phrase, study_id) after sorting
    best = df.groupby(["query_phrase", "study_id"], as_index=False, sort=False).head(1)
    return best


def recall_at_k_study_level(results: pd.DataFrame, k: int = 5) -> float:
    """
    Study-level Recall@k: aggregate best slice per study, then compute recall@k.

    Expects columns:
      - 'query_phrase' (str), 'study_id' (hashable), 'is_positive' (bool)
      - optional 'score' (desc) or 'rank' (asc) to choose best slice.
    """
    best = aggregate_best_slice_per_study(results)
    # Within each query, (re)rank by score/rank if available
    if "score" in best.columns:
        best = best.sort_values(["query_phrase", "score"], ascending=[True, False])
    elif "rank" in best.columns:
        best = best.sort_values(["query_phrase", "rank"], ascending=[True, True])

    hits = []
    for _, group in best.groupby("query_phrase", sort=False):
        topk = group.head(int(k))
        hits.append(bool(topk["is_positive"].any()))
    if not hits:
        return float("nan")
    return float(np.mean(hits))
