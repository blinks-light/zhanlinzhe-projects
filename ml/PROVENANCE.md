# Model Provenance

This file links each production model back to the code, data and MLflow run that produced it.

| Stage | Dataset | Dataset commit | Git tag | MLflow run | Artifact |
|---|---|---|---|---|---|
| Baseline (MultinomialNB) | v1 | `DATASET_V1_COMMIT` | `data-v1` | `baseline-nb` | not exported (experiment only) |
| Improved (LogisticRegression) | v2 | `DATASET_V2_COMMIT` | `data-v2` | `improved-lr` | `artifacts/spam_model.joblib`, `artifacts/normal_model.joblib` |

## How the chain is verified

1. `git log --oneline --decorate` shows the commit that produced each dataset version.
2. `data.dvc` at that commit stores the DVC hash of the dataset.
3. `ml/train.py` logs that hash as the `data_version` tag and the short git SHA as the
   `git_sha` tag on every MLflow run.
4. `mlruns/` stores the run parameters, metrics and model artifacts locally.
5. `ml/metrics/improved.json` is committed as a static copy of the production run summary.

## Reproduce the production model

```bash
git checkout main
dvc pull
python ml/train.py --model lr --run-name improved-lr --save-final
```

The resulting joblib files are byte-comparable in behaviour to the committed artifacts
because the data split and all hyperparameters are fixed (`random_state=42`).
