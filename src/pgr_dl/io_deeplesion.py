from __future__ import annotations
from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from PIL import Image


def _pick(df: pd.DataFrame, *cands: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        lc = c.lower()
        if lc in cols:
            return cols[lc]
    return None


def _coerce_int(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    except Exception:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")


def _extract_int_from_name(name: str) -> int | None:
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else None


def _resolve_image_path(root: Path, name_or_rel: str) -> Path:
    p = Path(name_or_rel)
    if p.is_absolute():
        return p
    # try as relative to root first
    cand = root / p
    if cand.exists():
        return cand
    # common DeepLesion layout: Images_png/<file_name>
    cand2 = root / "Images_png" / p.name
    if cand2.exists():
        return cand2
    # sometimes stored under images/ or pngs/
    for sub in ("images", "pngs", "png", "imgs"):
        cand3 = root / sub / p.name
        if cand3.exists():
            return cand3
    # fall back to root/<name>
    return root / p.name


def load_metadata(root: str | Path, csv_name: str = "DL_info.csv") -> pd.DataFrame:
    """
    Load a DeepLesion subset CSV and normalize to columns:
      - study_id : str  (e.g., "{patient_index}_{study_index}" or a UID)
      - slice_idx: int  (from Slice_index / InstanceNumber / extracted from file name)
      - img_path : str  (absolute path to PNG)

    The function is robust to typical Kaggle/DeepLesion column variants:
    Patient_index, Study_index, Slice_index, File_name, Series_UID, etc.
    """
    root = Path(root)
    csv_path = root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"DeepLesion CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path.name} is empty")

    # original -> lower map
    rename_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    # candidates
    patient_col = _pick(df, "patient_index", "patientid", "patient_id")
    study_col   = _pick(df, "study_index", "studyid", "study_id", "series_uid", "study_uid", "seriesid", "studyuid")
    slice_col   = _pick(df, "slice_idx", "slice_index", "slice", "instance_number", "image_index", "imagenumber", "z_index")
    file_col    = _pick(df, "file_name", "filename", "png_name", "image_path", "path", "png_path")

    # ---- build study_id
    study_id = None
    if patient_col is not None and study_col is not None:
        study_id = df[patient_col].astype(str).str.strip() + "_" + df[study_col].astype(str).str.strip()
    elif study_col is not None:
        study_id = df[study_col].astype(str).str.strip()
    elif patient_col is not None:
        study_id = df[patient_col].astype(str).str.strip()
    else:
        # last resort: parent folder of file_name (if paths carry structure)
        if file_col is not None:
            study_id = df[file_col].astype(str).apply(lambda p: Path(p).parent.name or "unknown")
        else:
            study_id = pd.Series(["unknown"] * len(df), index=df.index, dtype="string")
    study_id = study_id.astype("string")

    # ---- build slice_idx
    if slice_col is not None:
        slice_idx = _coerce_int(df[slice_col])
    else:
        # try to extract from file name digits
        if file_col is None:
            raise ValueError("Could not infer 'slice_idx' (no slice-like column and no file_name/path present).")
        slice_idx = df[file_col].apply(_extract_int_from_name).astype("Int64")

    # fill remaining NA slice_idx from filename digits if possible
    if slice_idx.isna().any() and file_col is not None:
        fill = df[file_col][slice_idx.isna()].apply(_extract_int_from_name).astype("Int64")
        slice_idx.loc[fill.index] = fill

    if slice_idx.isna().all():
        raise ValueError("Failed to construct 'slice_idx' from CSV. Provide a column like Slice_index/InstanceNumber or file names with numeric tokens.")

    # ---- build absolute img_path
    if file_col is not None:
        paths = df[file_col].astype(str).apply(lambda p: str(_resolve_image_path(root, p)))
    else:
        # sometimes DeepLesion provides only an index; try Images_png/<index>.png
        idx_for_name = slice_idx.fillna(0).astype(int).astype(str).str.zfill(7)  # common 7-digit pad
        paths = idx_for_name.apply(lambda s: str(_resolve_image_path(root, f"{s}.png")))

    out = pd.DataFrame({
        "study_id": study_id,
        "slice_idx": slice_idx.astype("Int64"),
        "img_path": paths.astype("string"),
    })

    # pass through a few optional columns if present
    for extra in ("body_part", "lesion_type", "split"):
        if extra in df.columns:
            out[extra] = df[extra].astype("string")

    # keep only rows that actually exist on disk (prevents later I/O errors)
    exists = out["img_path"].apply(lambda p: os.path.exists(p))
    if exists.any():
        out = out[exists].reset_index(drop=True)

    return out


def load_slice(path: str | Path) -> np.ndarray:
    """Load a single PNG slice as grayscale float32 [0,255] (H,W)."""
    p = Path(path)
    im = Image.open(p).convert("L")
    arr = np.asarray(im, dtype=np.float32)
    return arr
