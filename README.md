# ct-phrase-retrieval

Mini demo: **phrase-grounded retrieval for CT slices** on a small DeepLesion subset.
Shows end-to-end skills: CT 3-window preprocessing → CLIP embeddings → FAISS search → quick metrics and 3×3 visual panels.

> **Why this repo (portfolio context):** A compact, Kaggle-runnable project meant to demonstrate practical, teachable foundations for NKI AI Lab’s “foundation models for oncology” direction — intentionally small but complete.

---

## Quickstart (Kaggle, ~5 minutes)

1. **Open a Kaggle Notebook**, add:

   * **Dataset:** `nih-deeplesion-subset` (must contain `DL_info.csv`)
   * **This repo** as a dataset or via `git clone`
   * **GPU runtime:** optional (CPU is fine for the demo)

2. **Install**

```bash
pip install -r requirements.txt
pip install -e .
```

3. **Configure**
   Check `configs/deeplesion_kaggle.yaml`:

```yaml
paths:
  data_root: "/kaggle/input/nih-deeplesion-subset"
  csv_name: "DL_info.csv"
results:
  dir: "/kaggle/working/results/kaggle_v1"
phrases:
  set:
    - "liver lesion"
    - "renal mass"
    - "splenic lesion"
    - "lung nodule"
    - "enlarged lymph node"
    - "bone lesion"
```

4. **Build index**

```bash
python scripts/build_index.py --config configs/deeplesion_kaggle.yaml
```

**Outputs (in `results/...`)**

* `image_embs.npy` — image embeddings
* `ids.parquet` — per-slice metadata/IDs
* `index.faiss` — FAISS index (Flat/IP)
* `manifest.json` — paths + config summary

5. **Query a phrase (top-k + optional panel)**

```bash
python apps/query.py --results_dir /kaggle/working/results/kaggle_v1 \
  --phrase "liver lesion" -k 9 --save_panel
```

Prints top-k hits and saves a 3×3 panel PNG.

6. **Evaluate (toy Recall@k)**

```bash
# Uses YAML phrase set if available, else default tiny bank
python apps/evaluate.py --results_dir /kaggle/working/results/kaggle_v1 -k 5
# Or your own CSV:
# python apps/evaluate.py --results_dir ... --phrases_csv phrases/example_phrases.csv -k 5
```

---

## Method (short)

* **Preprocess:** Window CT to 3 channels (lung/soft/bone), resize to 224, CLIP normalization.
* **Encoder:** OpenCLIP ViT-B/16 (`openai` weights) for image & text.
* **Index:** FAISS Flat with inner-product over L2-normalized embeddings (cosine-equivalent).
* **Query:** Encode phrase → top-k nearest neighbors.
* **Viz:** Simple 3×3 grid; optional CAM overlay for qualitative checks.
* **Metric:** Toy **Recall@k** against a small phrase/tag mapping.

---

## Repo structure

```
configs/    # YAML (Kaggle-friendly)
notebooks/  # 00_build_index_dl.ipynb, 01_query_panel.ipynb, 02_eval_recallk.ipynb
scripts/    # build_index.py (headless pipeline)
src/
  pgr/      # encoders, FAISS wrapper, CAM, viz, utils
  pgr_dl/   # DeepLesion glue: I/O, windowing, phrase bank, eval helpers
apps/       # CLI tools: query.py, evaluate.py
phrases/    # (optional) example_phrases.csv
results/    # artifacts (gitignored)
```

---

## Results (thumbnail)

After a run, save a sample panel to:

```
assets/results_panel.png
```

Then it renders here:

![Top-9 panel for "liver lesion"](assets/results_panel.png)

---

## Honest limits & next steps

**Deliberately minimal** to stay teachable:

* Phrase side is small; no heavy report mining.
* Grounding via simple CAM; not a detector/segmenter.
* Recall@k uses a lightweight phrase→tag matching.

**Next I would add:**

* IVF-PQ index for scale, and saliency-aware re-ranking.
* Organ-specific phrase banks and harder negatives.
* Proper grounding head (e.g., ViT token maps or a tiny detector).

---

## Acknowledgements

* **DeepLesion** dataset (NIH).
* **FAISS** (Meta AI) for nearest-neighbor search.
* **OpenCLIP** for CLIP-style encoders.

---

## License

MIT — see [LICENSE](LICENSE).
