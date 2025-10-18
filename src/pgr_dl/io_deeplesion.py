from __future__ import annotations
from pathlib import Path
import os, re, glob
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



from pathlib import Path
import os
import pandas as pd

def load_metadata(root: str | Path, csv_name: str = "DL_info.csv") -> pd.DataFrame:
    """
    Load DeepLesion-style metadata with a Kaggle-safe CSV override.
    """
    root = Path(root)
    csv_name = os.getenv("PGR_DL_CSV_NAME", csv_name)
    csv_path = root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {csv_path}\n"
            "Tip: In Kaggle, add your dataset in the sidebar and ensure "
            "configs/deeplesion_kaggle.yaml -> paths.csv_name matches the file name."
        )
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        df["split"] = "all"
    return df


def load_slice(path: str | Path) -> np.ndarray:
    """Load a single PNG slice as grayscale float32 [0,255] (H,W)."""
    p = Path(path)
    if not p.exists():
        # try to resolve lazily if a bad path slips through
        guessed_root = p.parents[2] if len(p.parents) > 2 else Path("/kaggle/input/nih-deeplesion-subset")
        p = _resolve_image_path(guessed_root, p.as_posix())
    im = Image.open(p).convert("L")
    return np.asarray(im, dtype=np.float32)
