from __future__ import annotations
from pathlib import Path
import os, re, glob
import pandas as pd
import numpy as np
from PIL import Image


# ---------- small helpers ----------

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

def _split_study_and_file(name: str) -> tuple[str, str]:
    """
    Accepts "001274_01_02_043.png" or "001274_01_02/043.png" and returns:
      study="001274_01_02", file="043.png"
    """
    s = str(name).replace("\\", "/").strip("/")
    if "/" in s:
        study, file = s.rsplit("/", 1)
        return study, file
    parts = s.split("_")
    if len(parts) >= 4 and "." in parts[-1]:
        study = "_".join(parts[:-1])
        file = parts[-1]
        return study, file
    return "", s


# ---------- path resolution ----------

def _resolve_image_path(root: Path, name_or_rel: str) -> Path:
    p = Path(name_or_rel)
    if p.is_absolute() and p.exists():
        return p

    study, file = _split_study_and_file(p.as_posix())

    roots = [
        root,
        root / "minideeplesion",   # main location in your dataset
        root / "Images_png",
        root / "images_png",
        root / "images",
        root / "pngs",
        root / "png",
        root / "imgs",
    ]

    candidates: list[Path] = []
    for r in roots:
        candidates += [
            r / p,                          # keep any subpath provided
            r / p.name,                     # basename at each root
            (r / study / file) if study else (r / file),
        ]

    for c in candidates:
        if c.exists():
            return c

    hits = glob.glob(str(root / "**" / p.name), recursive=True)
    if hits:
        return Path(hits[0])

    return root / p.name


# ---------- public API ----------

def load_metadata(root: str | Path, csv_name: str = "DL_info.csv") -> pd.DataFrame:
    """
    Normalize DeepLesion CSV to:
      - study_id : str   (e.g., "001274_01_02")
      - slice_idx: int   (e.g., 43)
      - img_path : str   (absolute PNG path)
    """
    root = Path(root)
    csv_path = root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"DeepLesion CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path.name} is empty")

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    patient_col = _pick(df, "patient_index", "patientid", "patient_id")
    study_col   = _pick(df, "study_index", "studyid", "study_id", "series_uid", "study_uid", "seriesid", "studyuid")
    slice_col   = _pick(df, "slice_idx", "slice_index", "slice", "instance_number", "image_index", "imagenumber", "z_index")
    file_col    = _pick(df, "file_name", "filename", "png_name", "image_path", "path", "png_path", "png")

    # study_id
    if patient_col is not None and study_col is not None:
        study_id = df[patient_col].astype(str).str.strip() + "_" + df[study_col].astype(str).str.strip()
    elif study_col is not None:
        study_id = df[study_col].astype(str).str.strip()
    elif patient_col is not None:
        study_id = df[patient_col].astype(str).str.strip()
    elif file_col is not None:
        study_id = df[file_col].astype(str).apply(lambda p: Path(p).parent.name or "unknown")
    else:
        study_id = pd.Series(["unknown"] * len(df), index=df.index, dtype="string")
    study_id = study_id.astype("string")

    # slice_idx
    if slice_col is not None:
        slice_idx = _coerce_int(df[slice_col])
    else:
        if file_col is None:
            raise ValueError("Could not infer 'slice_idx' (no slice-like column and no file/path column).")
        slice_idx = df[file_col].apply(_extract_int_from_name).astype("Int64")

    if slice_idx.isna().any() and file_col is not None:
        fill = df[file_col][slice_idx.isna()].apply(_extract_int_from_name).astype("Int64")
        slice_idx.loc[fill.index] = fill

    if slice_idx.isna().all():
        raise ValueError("Failed to construct 'slice_idx' from CSV.")

    # absolute img_path
    if file_col is not None:
        paths = df[file_col].astype(str).apply(lambda p: str(_resolve_image_path(root, p)))
    else:
        # fallback: use zero-padded slices under detected roots
        names = slice_idx.fillna(0).astype(int).astype(str).str.zfill(3) + ".png"
        paths = names.apply(lambda s: str(_resolve_image_path(root, s)))

    out = pd.DataFrame({
        "study_id": study_id,
        "slice_idx": slice_idx.astype("Int64"),
        "img_path": paths.astype("string"),
    })

    for extra in ("body_part", "lesion_type", "split"):
        if extra in df.columns:
            out[extra] = df[extra].astype("string")

    exists = out["img_path"].apply(lambda p: os.path.exists(p))
    if exists.any():
        out = out[exists].reset_index(drop=True)

    return out


def load_slice(path: str | Path) -> np.ndarray:
    """Load a single PNG slice as grayscale float32 [0,255] (H,W)."""
    p = Path(path)
    if not p.exists():
        # try to resolve lazily if a bad path slips through
        guessed_root = p.parents[2] if len(p.parents) > 2 else Path("/kaggle/input/nih-deeplesion-subset")
        p = _resolve_image_path(guessed_root, p.as_posix())
    im = Image.open(p).convert("L")
    return np.asarray(im, dtype=np.float32)
