from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml, numpy as np, pandas as pd
from pgr_dl.io_deeplesion import load_metadata
from pgr.index import build_index, save_index

def main():
    ap = argparse.ArgumentParser(description="Headless build of embeddings/index")
    ap.add_argument("--config", required=True, help="YAML config (Kaggle-safe)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    csv_name  = cfg["paths"].get("csv_name", "DL_info.csv")
    results   = Path(cfg["paths"].get("results_dir", "results/kaggle_v1"))
    results.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(data_root, csv_name=csv_name)

    # bootstrap: toy embeddings so pipeline runs end-to-end offline
    n = min(512, len(meta)); dim = 256
    rng = np.random.default_rng(0)
    image_embs = rng.standard_normal((n, dim)).astype("float32")
    ids = pd.DataFrame({"row_id": np.arange(n), "path": meta.get("File_name", meta.index)[:n]})

    index = build_index(image_embs, metric="ip", kind="flat")
    save_index(index, results / "index.faiss")

    np.save(results / "image_embs.npy", image_embs)
    ids.to_parquet(results / "ids.parquet", index=False)
    (results / "manifest.json").write_text(json.dumps({
        "dim": dim, "n": int(n), "metric": "ip", "kind": "flat",
        "csv": str(csv_name), "data_root": str(data_root.resolve())
    }, indent=2))
    print(f"[ok] Built index with {n} items @ {results}")

if __name__ == "__main__":
    main()
