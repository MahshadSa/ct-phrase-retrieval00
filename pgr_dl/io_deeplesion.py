# metadata loader, image reader (Kaggle subset)
from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from pgr.utils import SampleRow


def _parse_bbox_row(r: pd.Series) -> Optional[Tuple[int, int, int, int]]:
    # separate columns
    if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(r.index):
        try:
            x1 = int(r["bbox_x1"])
            y1 = int(r["bbox_y1"])
            x2 = int(r["bbox_x2"])
            y2 = int(r["bbox_y2"])
            return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None
        except Exception:
            return None
    # single string column: "x1,y1,x2,y2"
    if "bbox" in r.index and isinstance(r["bbox"], str) and r["bbox"].strip():
        try:
            parts = [int(v) for v in r["bbox"].replace(" ", "").split(",")]
            if len(parts) == 4:
                x1, y1, x2, y2 = parts
                return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None
        except Exception:
            return None
    return None


def load_metadata(root: str, csv_name: str = "metadata.csv") -> pd.DataFrame:
    """
    Load Kaggle DeepLesion subset metadata.

    Expected (after normalization):
      study_id, slice_idx, img_path, body_part?, lesion_type?, bbox_*?, split?
    """
    root_path = Path(root)
    csv_path = root_path / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # common aliases
    rename_map = {
        "study": "study_id",
        "series_uid": "study_id",
        "instance": "slice_idx",
        "slice": "slice_idx",
        "path": "img_path",
        "image_path": "img_path",
        "bodypart": "body_part",
        "lesion": "lesion_type",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"study_id", "slice_idx", "img_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns in {csv_path.name}: {sorted(missing)}")

    # coerce dtypes
    df["study_id"] = df["study_id"].astype(str)
    df["slice_idx"] = df["slice_idx"].astype(int)
    if "split" in df.columns:
        df["split"] = df["split"].astype(str)

    # resolve img_path to absolute under root if relative
    def _resolve_path(p: str) -> str:
        pth = Path(p)
        return str(pth if pth.is_absolute() else (root_path / pth))

    df["img_path"] = df["img_path"].astype(str).map(_resolve_path)

    # optional bbox parsing → four columns
    if any(c in df.columns for c in ["bbox", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]):
        boxes = df.apply(_parse_bbox_row, axis=1)
        # expand to columns (int or NaN)
        bx = boxes.apply(lambda b: b[0] if b else np.nan).astype("Int64")
        by = boxes.apply(lambda b: b[1] if b else np.nan).astype("Int64")
        bw = boxes.apply(lambda b: b[2] if b else np.nan).astype("Int64")
        bh = boxes.apply(lambda b: b[3] if b else np.nan).astype("Int64")
        df["bbox_x1"], df["bbox_y1"], df["bbox_x2"], df["bbox_y2"] = bx, by, bw, bh

    # ensure optional columns exist for downstream code
    for col in ["body_part", "lesion_type", "split"]:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype="object")

    return df.reset_index(drop=True)


def iter_rows(df: pd.DataFrame, split: Optional[str] = None):
    """Yield SampleRow from a metadata dataframe (optionally filtered by split)."""
    view = df if split is None else df[df["split"] == split]
    cols = set(view.columns)
    has_bbox = {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(cols)

    for _, r in view.iterrows():
        bbox = None
        if has_bbox and pd.notna(r["bbox_x1"]) and pd.notna(r["bbox_y1"]) and pd.notna(r["bbox_x2"]) and pd.notna(r["bbox_y2"]):
            bbox = (int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"]))
        yield SampleRow(
            study_id=str(r["study_id"]),
            slice_idx=int(r["slice_idx"]),
            img_path=str(r["img_path"]),
            body_part=str(r["body_part"]) if "body_part" in cols and pd.notna(r["body_part"]) else None,
            lesion_type=str(r["lesion_type"]) if "lesion_type" in cols and pd.notna(r["lesion_type"]) else None,
            bbox=bbox,
            split=str(r["split"]) if "split" in cols and pd.notna(r["split"]) else None,
        )


def load_slice(img_path: str) -> np.ndarray:
    """
    Load a single image slice from disk.

    Returns:
      - HxW (uint8) for grayscale images
      - HxWx3 (uint8) if the source is RGB
    """
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {img_path}")

    with Image.open(p) as im:
        im.load()  # ensure data is read
        # strip alpha if present
        if im.mode == "RGBA":
            im = im.convert("RGB")
        # prefer grayscale for CT-like inputs
        if im.mode in ("I;16", "I"):
            # 16-bit -> scale to 8-bit for downstream simplicity
            arr16 = np.array(im, dtype=np.uint16)
            if arr16.size == 0:
                raise ValueError(f"empty image: {img_path}")
            # linear rescale to [0,255]
            lo, hi = int(arr16.min()), int(arr16.max())
            if hi > lo:
                arr = ((arr16 - lo) * (255.0 / (hi - lo))).astype(np.uint8)
            else:
                arr = np.zeros_like(arr16, dtype=np.uint8)
            return arr
        if im.mode in ("L",):
            return np.array(im, dtype=np.uint8)
        if im.mode in ("RGB",):
            return np.array(im, dtype=np.uint8)
        # fallback: convert to grayscale
        return np.array(im.convert("L"), dtype=np.uint8)
