# ct-phrase-retrieval

Mini demo: **phrase‑grounded retrieval for CT slices** on a tiny DeepLesion subset. Shows end‑to‑end skills: CT 3‑window preprocessing → CLIP embeddings → FAISS search → quick metrics and 3×3 visual panels.

> **Why this repo (portfolio context):** A compact, Kaggle‑runnable project to demonstrate practical, teachable foundations for NKI AI Lab’s *Foundation Models for Oncology* direction — intentionally small but complete.

---

## Quickstart (Kaggle, ~5 minutes)

```bash
# 1) Install (avoid pinning numpy on Kaggle)
pip install -r requirements.txt
pip install -e .

# 2) Verify the dataset mount (adjust if you used a different name/file)
python - <<'PY'
from pathlib import Path
p = Path('/kaggle/input/nih-deeplesion-subset/DL_info.csv')
assert p.exists(), f"Missing: {p}. Add the Kaggle dataset or edit configs/deeplesion_kaggle.yaml"
print('OK: data present')
PY

# 3) Build (headless; creates FAISS index + small artifacts)
python scripts/build_index.py --config configs/deeplesion_kaggle.yaml

# 4) Query (save a simple 3×3 panel listing)
python apps/query.py \
  --results_dir /kaggle/working/results/kaggle_v1 \
  --phrase "liver lesion" -k 9 --save_panel

# 5) Evaluate (toy Recall@k)
python apps/evaluate.py --results_dir /kaggle/working/results/kaggle_v1 -k 5
```

**No internet on Kaggle?** Upload CLIP weights as a private Kaggle Dataset and pass the local path via `weights_path` in `src/pgr/encoders.py` (or keep the toy embeddings to show end‑to‑end flow).

**Config used above:** `configs/deeplesion_kaggle.yaml`

```yaml
paths:
  data_root: /kaggle/input/nih-deeplesion-subset
  csv_name: DL_info.csv
  results_dir: /kaggle/working/results/kaggle_v1
```

**Troubleshooting**

* `ImportError: cannot import 'pgr_dl'` → Run `pip install -e .` and ensure `src/pgr_dl/__init__.py` exists.
* `FileNotFoundError: DL_info.csv` → Add the dataset in Kaggle *Add data* or edit the YAML `paths.csv_name`.
* `RuntimeError: pretrained weights not found` → Enable internet or supply a weights dataset and set `weights_path`.

---

## Method (short)

* **Preprocess:** Window CT to 3 channels (lung/soft/bone), resize to 224, CLIP normalisation.
* **Encoder:** OpenCLIP ViT‑B/16 (`openai` weights) for image & text (or toy embeddings to stay offline).
* **Index:** FAISS Flat with inner‑product over L2‑normalised embeddings (cosine‑equivalent).
* **Query:** Encode phrase → top‑k nearest neighbours.
* **Viz:** Simple 3×3 grid; optional CAM overlay for qualitative checks.
* **Metric:** Toy **Recall@k** against a small phrase/tag mapping.

---

## Repo structure

```
configs/    # YAML (Kaggle‑friendly)
notebooks/  # 00_build_index_dl.ipynb, 01_query_panel.ipynb, 02_eval_recallk.ipynb
scripts/    # build_index.py (headless pipeline)
src/
  pgr/      # encoders, FAISS wrapper, CAM, viz, utils
  pgr_dl/   # DeepLesion glue: I/O, windowing, phrase bank, eval helpers
apps/       # CLI tools: query.py, evaluate.py
phrases/    # (optional) example_phrases.csv
results/    # artifacts (gitignored)
assets/     # (optional) thumbnails for README
```

---

## Results (thumbnail)

After a run, save a sample panel to:

```
assets/results_panel.png
```

Then it renders here:

![Top‑9 panel for "liver lesion"](assets/results_panel.png)

---

## Honest limits & next steps

**Deliberately minimal** to stay teachable:

* Phrase side is small; no heavy report mining yet.
* Grounding via simple CAM; not a detector/segmenter.
* Recall@k uses a lightweight phrase→tag matching.

**Next I would add:**

* IVF‑PQ index for scale, plus saliency‑aware re‑ranking.
* Organ‑specific phrase banks and harder negatives.
* A proper grounding head (e.g., ViT token maps or a tiny detector), and basic pointing‑accuracy metrics.

---

## Acknowledgements

* **DeepLesion** dataset (NIH).
* **FAISS** (Meta AI) for nearest‑neighbour search.
* **OpenCLIP** for CLIP‑style encoders.

---

## Licence

MIT — see [LICENSE](LICENSE).
