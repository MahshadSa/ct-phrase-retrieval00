from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from pgr import encoders, index
from pgr_dl.eval_dl import recall_at_k_dataframe
from pgr_dl.phrases import default_phrase_bank

def _load_phrases_from_yaml(manifest: dict) -> list[str] | None:
    cfg_p = Path(manifest.get("config_path", ""))
    if cfg_p.exists():
        try:
            import yaml
            with open(cfg_p, "r") as f:
                y = yaml.safe_load(f)
            return y.get("phrases", {}).get("set", None)
        except Exception:
            return None
    return None

def main(args: argparse.Namespace) -> None:
    res_dir = Path(args.results_dir)
    man = json.loads((res_dir / "manifest.json").read_text())
    paths = man["manifest"]["paths"] if "manifest" in man else man["paths"]

    ids = pd.read_parquet(paths["ids"])
    if len(ids) == 0:
        raise RuntimeError("ids.parquet is empty.")

    fa = index.FaissIndex.load(paths["index"])
    enc = encoders.ClipEncoder(model_name=args.model or "ViT-B-16",
                               pretrained="openai",
                               device=args.device or "cpu")

    if args.phrases_csv is not None:
        phrases = pd.read_csv(args.phrases_csv)["phrase"].tolist()
    else:
        phrases = _load_phrases_from_yaml(man) or default_phrase_bank()

    k = int(args.k)
    rows = []
    for ph in phrases:
        qv = enc.encode_text([ph]).cpu().numpy().astype("float32")
        kk = min(k, len(ids))
        I, _ = fa.search(qv, k=kk)
        hits = ids.iloc[I[0]].reset_index(drop=True)
        hits["query_phrase"] = ph
        hits["rank"] = np.arange(len(hits))
        rows.append(hits[["query_phrase", "study_id", "slice_idx", "rank"]])

    df = pd.concat(rows, ignore_index=True)
    r_at_k = float(recall_at_k_dataframe(df, k=k, ids_df=ids))
    print({f"R@{k}": round(r_at_k, 4)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--phrases_csv", default=None, help="CSV with a 'phrase' column (optional)")
    ap.add_argument("--device", default=None)
    main(ap.parse_args())
