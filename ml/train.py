"""Train the two-stage email classifier and track experiments with MLflow.

Experiment 1 (baseline): TF-IDF + MultinomialNB.
Experiment 2 (improved): TF-IDF + LogisticRegression.

The final run writes both production models to artifacts/:
    artifacts/spam_model.joblib
    artifacts/normal_model.joblib
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# MLflow 3.x keeps the local file store behind an explicit opt-in flag.
# The course allows a local file backend, so enable it for every run.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "emails.csv"
DVC_FILE = ROOT / "data.dvc"
PARAMS_FILE = ROOT / "ml" / "configs" / "params.yaml"
ARTIFACT_DIR = ROOT / "artifacts"
METRICS_DIR = ROOT / "ml" / "metrics"
EXPERIMENT_NAME = "email-cascade-classification"
RUN_FILE_NAMES = {"nb": "baseline", "lr": "improved"}


def git_sha() -> str:
    """Return the short commit hash of the code that produced this run."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_version() -> str:
    """Read the DVC hash of the tracked dataset from data.dvc."""
    if not DVC_FILE.exists():
        return "unknown"
    for line in DVC_FILE.read_text(encoding="utf-8").splitlines():
        if "md5:" in line:
            return line.split("md5:", 1)[1].strip()
    return "unknown"


def load_params() -> dict[str, Any]:
    """Load the TF-IDF and model hyperparameters from params.yaml."""
    return yaml.safe_load(PARAMS_FILE.read_text(encoding="utf-8"))


def build_pipeline(model_kind: str, params: dict[str, Any]) -> Pipeline:
    tfidf_params = dict(params["tfidf"])
    tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])
    if model_kind == "nb":
        classifier: Any = MultinomialNB(**params["nb"])
    else:
        classifier = LogisticRegression(**params["lr"])
    return Pipeline([("tfidf", TfidfVectorizer(**tfidf_params)), ("clf", classifier)])


def score(model: Pipeline, texts: pd.Series, labels: pd.Series, average: str) -> dict[str, Any]:
    predictions = model.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average=average)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "labels": sorted({str(label) for label in labels}),
    }


def cascade_predict(spam_model: Pipeline, normal_model: Pipeline, texts: list[str]) -> list[str]:
    spam_flags = spam_model.predict(texts)
    results: list[str | None] = ["spam" if int(flag) == 1 else None for flag in spam_flags]

    normal_indexes = [index for index, result in enumerate(results) if result is None]
    if normal_indexes:
        normal_texts = [texts[index] for index in normal_indexes]
        normal_predictions = normal_model.predict(normal_texts)
        for index, prediction in zip(normal_indexes, normal_predictions):
            results[index] = str(prediction)
    return [str(result) for result in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["nb", "lr"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Export the two cascade models to artifacts/ for the API service.",
    )
    args = parser.parse_args()

    params = load_params()
    frame = pd.read_csv(DATA_FILE)
    required_columns = {"text", "is_spam", "category"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    train_frame, test_frame = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame["category"]
    )

    spam_model = build_pipeline(args.model, params)
    spam_model.fit(train_frame["text"], train_frame["is_spam"])

    normal_train = train_frame[train_frame["is_spam"] == 0]
    normal_test = test_frame[test_frame["is_spam"] == 0]
    normal_model = build_pipeline(args.model, params)
    normal_model.fit(normal_train["text"], normal_train["category"])

    spam_scores = score(spam_model, test_frame["text"], test_frame["is_spam"], average="binary")
    normal_scores = score(
        normal_model, normal_test["text"], normal_test["category"], average="macro"
    )
    cascade_labels = cascade_predict(spam_model, normal_model, test_frame["text"].tolist())
    cascade_accuracy = float(accuracy_score(test_frame["category"], cascade_labels))
    cascade_f1 = float(f1_score(test_frame["category"], cascade_labels, average="macro"))

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(
            {
                "git_sha": git_sha(),
                "data_version": data_version(),
                "model_family": "naive_bayes" if args.model == "nb" else "logistic_regression",
            }
        )
        run_params = {
            "model": args.model,
            "tfidf_max_features": params["tfidf"]["max_features"],
            "tfidf_ngram_range": str(tuple(params["tfidf"]["ngram_range"])),
            "tfidf_min_df": params["tfidf"]["min_df"],
            "test_size": 0.2,
            "random_state": 42,
        }
        family_params = params["nb"] if args.model == "nb" else params["lr"]
        run_params.update({f"{args.model}_{key}": value for key, value in family_params.items()})
        mlflow.log_params(run_params)

        metrics = {
            "spam_accuracy": spam_scores["accuracy"],
            "spam_f1": spam_scores["f1"],
            "normal_accuracy": normal_scores["accuracy"],
            "normal_f1_macro": normal_scores["f1"],
            "cascade_accuracy": cascade_accuracy,
            "cascade_f1_macro": cascade_f1,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"spam": spam_scores, "normal": normal_scores}, "confusion_matrices.json")

        if args.save_final:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            spam_path = ARTIFACT_DIR / "spam_model.joblib"
            normal_path = ARTIFACT_DIR / "normal_model.joblib"
            joblib.dump(spam_model, spam_path)
            joblib.dump(normal_model, normal_path)
            mlflow.log_artifact(str(spam_path))
            mlflow.log_artifact(str(normal_path))

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_name": args.run_name,
            "run_id": run.info.run_id,
            "model": args.model,
            "git_sha": git_sha(),
            "data_version": data_version(),
            "params": run_params,
            "metrics": metrics,
        }
        metrics_path = METRICS_DIR / f"{RUN_FILE_NAMES[args.model]}.json"
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))

    print(f"run {args.run_name} logged to {EXPERIMENT_NAME}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
