# Current Model

| Field | Value |
|---|---|
| Model | TF-IDF + LogisticRegression (two-stage cascade) |
| Stage 1 | `artifacts/spam_model.joblib` (binary spam classifier) |
| Stage 2 | `artifacts/normal_model.joblib` (work / personal / notification) |
| Dataset | `data-v2` (102 rows) |
| MLflow experiment | `email-cascade-classification` |
| MLflow run | `improved-lr` |
| Owner | Zhan Linzhe (ML) |
| Status | production (served by `app/main.py`) |

## Why this model

The improved run has the best macro-F1 on the normal-email sub-categories and keeps spam
recall high. It is the model exported to `artifacts/` and shipped in the Docker image.
See `ml/README_experiments.md` for the metric comparison and `ml/PROVENANCE.md` for the
data/commit linkage.
