# Deploying the Streamlit dashboard

The dashboard entry point is the repository-root `Home.py`. Streamlit discovers
the analysis pages from the sibling `pages/` directory.

## Required files and folders

The app reads prepared Parquet data from `processed/` and optional exported
artifacts from `artifacts/` and `reports/` (all git-ignored, not bundled).

- **Minimum for analytics:** at least one readable Parquet file with a
  timestamp column in `processed/` (e.g. `btc_features.parquet`).
- **Forecasting (`pages/2_Forecasting.py`):** `processed/split_meta.yaml` and
  LightGBM/CatBoost artifacts in `artifacts/models/` (`lgbm_return.pkl`,
  `catboost_return.pkl`); optional exports `reports/metrics/walk_forward_summary.json`,
  `artifacts/models/*_test_predictions.csv`.
- **Volatility (`pages/3_Volatility.py`):** `artifacts/models/garch_cond_vol.csv`
  (optional realized-vol comparison uses overlapping feature columns).
- **Regimes (`pages/4_Regimes.py`):** GMM/HMM `.pkl` in `artifacts/models/`,
  `artifacts/scalers/regime_scaler.pkl`, `processed/split_meta.yaml`; optional
  `reports/metrics/regime_diagnostics.json`.
- **Anomalies (`pages/5_Anomalies.py`):** `artifacts/models/anomaly_scores.csv`;
  optional `artifacts/models/matrix_profile.csv` and `matrix_profile_meta.json`.
- **Change points (`pages/6_ChangePoints.py`):** `artifacts/models/changepoints.csv`.

Pages degrade gracefully when data/artifacts are missing, but analyses are only
meaningful with artifacts regenerated from the current code and notebooks.

## Regenerating artifacts (local/research environment)

Install research dependencies and run the notebook pipeline in order
`01_data_pipeline` → `02_feature_engineering` → `03_modelling` → `04_evaluation_and_export`,
or use the helper:

```bash
python -m pip install -r requirements-research.txt
python scripts/prepare_data.py --full
```

Notes:
- `--no-notebooks` only creates directories.
- Full Spark-backed processing requires a compatible Java installation as
  configured in [`configs/pipeline.yaml`](configs/pipeline.yaml) (Spark/Java
  paths are not modified by the script).
- Re-run regeneration after changes to data schema, features, models, configs,
  or notebooks; existing outputs in `processed/`, `artifacts/`, `reports/`, and
  `configs/experiments.yaml` may be stale.

## Streamlit Community Cloud

1. Push the application source to a GitHub repository.
2. Set **Main file path** to `Home.py` and use Python 3.10+.
3. Provision `processed/`, required `artifacts/`, and any `reports/metrics`/figure
   exports **before** deploying (the app does not download them). Options:
   - Regenerate locally and publish versioned artifacts via controlled object
     storage or a persistent data store approved by your hosting platform.
   - Avoid committing large generated files to git unless size/terms permit;
     review `.gitignore`.
4. Add credentials via Streamlit app secrets or the platform secret manager.
   Never commit `.streamlit/secrets.toml` or secrets in source.
5. Validate each page that depends on optional artifacts after deploy.

## Local dashboard check

```bash
python -m pip install -r requirements.txt
streamlit run Home.py
```

## Operational notes

- Use the smallest dependency set for deployment; research/notebook deps are in
  `requirements-research.txt` (includes optional `statsmodels`, `arch`, `stumpy`,
  `ruptures`, Spark/HF/MLflow stack).
- Load only trusted serialized artifacts (`pickle`/`joblib` can execute code).
- Short-horizon BTC return forecasts are noisy; treat walk-forward/baseline
  comparisons and risk/diagnostic views as primary decision context.
- Check dataset terms and redistribution permissions before hosting source data.
