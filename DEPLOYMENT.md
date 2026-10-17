# Deployment Guide

This project uses GitHub Actions for CI and CD. There is no runtime deployment target:
the delivery stage stops after building the environment-specific Docker image.

## Environments

| Environment | Branch | GitHub Environment | Image tag | Env file secret |
|---|---|---|---|---|
| staging | `staging` | `staging` | `email-cascade-classifier:staging` | `STAGING_ENV_FILE` |
| production | `main` | `production` | `email-cascade-classifier:prod` | `PROD_ENV_FILE` |

## Required repository secrets

Create the secrets under **Settings → Environments → staging / production** so each
environment only sees its own values:

| Secret | Example value | Used by |
|---|---|---|
| `STAGING_ENV_FILE` | `APP_ENV=staging` | staging CD job |
| `PROD_ENV_FILE` | `APP_ENV=production` | production CD job |

The workflows write the secret to a `.env` file before building the image.

## CI workflow (`.github/workflows/ci.yml`)

Triggers on every pull request targeting `dev`, `staging` or `main`.

| Job | Command | Purpose |
|---|---|---|
| `Lint (ruff)` | `ruff check .` | static analysis and import order |
| `Tests (pytest)` | `pytest -q` | functional and boundary tests |
| `Docker build` | `docker build -t email-cascade-classifier:ci .` | verify the image builds |

All three jobs are required status checks. A pull request cannot be merged while any of
them is failing.

## CD workflow (`.github/workflows/cd.yml`)

Triggers on pushes to `staging` and `main`.

Staging job (`if: github.ref == 'refs/heads/staging'`, environment `staging`):

1. checkout the repository;
2. `echo "$STAGING_ENV_FILE" > .env`;
3. `docker build --build-arg APP_ENV=staging -t email-cascade-classifier:staging .`;
4. print the image ID.

Production job (`if: github.ref == 'refs/heads/main'`, environment `production`):

1. checkout the repository;
2. `echo "$PROD_ENV_FILE" > .env`;
3. `docker build --build-arg APP_ENV=production -t email-cascade-classifier:prod .`;
4. print the image ID.

## Branch protection

Configure once in **Settings → Branches**:

1. Add a rule for `main`, `staging` and `dev`.
2. Enable **Require a pull request before merging**.
3. Enable **Require status checks to pass before merging** and select:
   `Lint (ruff)`, `Tests (pytest)`, `Docker build`.
4. Enable **Require branches to be up to date before merging** (optional but recommended).

With these rules the merge path is `feature/* → dev → staging → main` and no failing
pull request can reach a protected branch.

## Running the image locally

```bash
docker build -t email-cascade-classifier:local .
docker run --rm -p 8000:8000 --env-file .env email-cascade-classifier:local
curl http://localhost:8000/health
```

The container runs as the non-root user `app` and exposes a `HEALTHCHECK` that calls
`/health` every 30 seconds.

## Rollback

Images are tagged per environment and every release is a merge commit. To roll back:

```bash
git checkout main
git revert -m 1 <merge-commit-sha>
git push origin main
```

The revert triggers the production CD job again and rebuilds the previous code state.
Because model artifacts are versioned in Git together with the code, the rebuilt image
contains exactly the models that shipped with that commit.

## Security notes

- `.env` files are git-ignored and are only generated inside the CI runner from secrets.
- The DVC Azure SAS token is stored in `.dvc/config.local`, which is git-ignored and must
  never be committed. See `ml/README_data.md` for the remote configuration.
- The image contains only `app/` and `artifacts/`; `ml/`, `data/`, `tests/` and `.git`
  are excluded through `.dockerignore`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| CI test job fails on `/predict` | model artifacts missing from the checkout | ensure `artifacts/*.joblib` is committed |
| Docker build fails on `COPY artifacts` | artifacts were deleted locally | rerun `python ml/train.py --model lr --run-name improved-lr --save-final` |
| CD job fails reading the secret | secret not set on the environment | add `STAGING_ENV_FILE` / `PROD_ENV_FILE` to the matching environment |
| `/predict` returns 503 | models failed to load at startup | check the container logs and the `artifacts/` paths |
