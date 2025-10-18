from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from pgr import encoders, index
from pgr.utils import get_device
from pgr.viz import grid_panel

def main(args: argparse.Namespace) -> None:
    res_dir = Path(args.results_dir)
    manifest_p = res_dir / "manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"manifest.json not found in {res_dir}")

    man = json.loads(manifest_p.read_text())
    paths = man["manifest"]["paths"] if "manifest" in man else man["paths"]

    fa = index.FaissIndex.load(paths["index"])
    ids = pd.read_parquet(paths["ids"])
    if len(ids) == 0:
        raise RuntimeError("ids.parquet is empty.")

    device = get_device(args.device)
    enc = encoders.ClipEncoder(model_name=args.model or "ViT-B-16",
                               pretrained="openai",
                               device=str(device))

    qv = enc.encode_text([args.phrase]).cpu().numpy().astype("float32")
    k = min(args.k, len(ids))
    I, D = fa.search(qv, k=k)
    hits = ids.iloc[I[0]].reset_index(drop=True)
    hits["score"] = D[0]

    print(hits.head(k).to_string(index=False))

    if args.save_panel and "img_path" in hits.columns:
        out = res_dir / f'panel_k{k}_{args.phrase.replace(" ","_")}.png'
        panel = grid_panel([Path(p) for p in hits["img_path"].tolist()[:min(k, 9)]],
                           rows=3, cols=3, title=f'"{args.phrase}" top-{k}')
        panel.save(out)
        print("[panel]", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="dir with index.faiss + ids.parquet + manifest.json")
    ap.add_argument("--phrase", required=True)
    ap.add_argument("-k", type=int, default=9)
    ap.add_argument("--model", default=None, help="CLIP backbone (default ViT-B-16)")
    ap.add_argument("--device", default=None, help="cuda|cpu|None(auto)")
    ap.add_argument("--save_panel", action="store_true")
    main(ap.parse_args())
