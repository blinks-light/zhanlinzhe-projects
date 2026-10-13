"""Build ml/README_experiments.md from the MLflow run summaries.

Run after both training commands:
    python ml/train.py --model nb --run-name baseline-nb
    python ml/train.py --model lr --run-name improved-lr --save-final
    python ml/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "ml" / "metrics"
OUTPUT = ROOT / "ml" / "README_experiments.md"
MODEL_NAMES = {"nb": "MultinomialNB", "lr": "LogisticRegression"}


def load_summary(name: str) -> dict:
    path = METRICS_DIR / name
    if not path.exists():
        raise SystemExit(f"missing {path}; run both training commands before this script")
    return json.loads(path.read_text(encoding="utf-8"))


def table_row(label: str, summary: dict) -> str:
    metrics = summary["metrics"]
    data_hash = str(summary["data_version"])[:8]
    return (
        f"| {label} | {MODEL_NAMES.get(summary['model'], summary['model'])} | `{data_hash}` | "
        f"{metrics['spam_accuracy']:.3f} | {metrics['spam_f1']:.3f} | "
        f"{metrics['normal_accuracy']:.3f} | {metrics['normal_f1_macro']:.3f} | "
        f"{metrics['cascade_accuracy']:.3f} | {metrics['cascade_f1_macro']:.3f} |"
    )


def main() -> None:
    baseline = load_summary("baseline.json")
    improved = load_summary("improved.json")

    parts = [
        "# Experiment Tracking",
        "",
        "Two experiments are tracked in MLflow, both using the same TF-IDF features and",
        "an 80/20 stratified train/test split (`random_state=42`):",
        "",
        "- **Baseline** — TF-IDF + MultinomialNB (`alpha=0.5`).",
        "- **Improved** — TF-IDF + LogisticRegression (`C=2.0`, `class_weight=balanced`).",
        "",
        "Each run logs the git commit SHA, the DVC dataset hash, all hyperparameters,",
        "accuracy and F1 metrics, confusion matrices and (for the improved run) both",
        "cascade models as joblib artifacts.",
        "",
        "## Results",
        "",
        "| Run | Model | Data hash | Spam acc | Spam F1 | Normal acc | Normal F1 | Cascade acc | Cascade F1 |",
        "|---|---|---|---|---|---|---|---|---|",
        table_row("Baseline", baseline),
        table_row("Improved", improved),
        "",
        "`Normal F1` and `Cascade F1` are macro-averaged; `Spam F1` is binary with spam as",
        "the positive class.",
        "",
        "## Which run is production-worthy",
        "",
        "The **improved LogisticRegression run** is the production model. It keeps spam",
        "detection at the same level as the baseline while giving the second stage a",
        "better separation between `work`, `personal` and `notification`, which is the",
        "harder part of the cascade. It is exported to `artifacts/spam_model.joblib` and",
        "`artifacts/normal_model.joblib` and is the model shipped in the Docker image.",
        "",
        "## Metric being optimized",
        "",
        "The primary metric is **macro-F1 over the four final categories** (spam, work,",
        "personal, notification). Accuracy alone would hide a weak class, and the normal",
        "sub-categories are unbalanced in the toy dataset; macro-F1 weights every class",
        "equally, which matches the API contract that all four labels must be reliable.",
        "Binary spam F1 is tracked separately because a missed spam email is the most",
        "visible failure for a user.",
        "",
        "## Viewing the runs",
        "",
        "```bash",
        "export MLFLOW_ALLOW_FILE_STORE=true   # PowerShell: $env:MLFLOW_ALLOW_FILE_STORE='true'",
        "mlflow ui --backend-store-uri mlruns",
        "# open http://127.0.0.1:5000",
        "```",
        "",
        "The committed summary of each run is also available in",
        "`ml/metrics/baseline.json` and `ml/metrics/improved.json`.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "dvc pull",
        "python ml/train.py --model nb --run-name baseline-nb",
        "python ml/train.py --model lr --run-name improved-lr --save-final",
        "```",
        "",
    ]
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
