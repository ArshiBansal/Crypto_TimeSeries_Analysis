# Deploying the Streamlit dashboard

The dashboard entry point is the repository-root `Home.py`. Streamlit discovers
the analysis pages from the sibling `pages/` directory.

## Required files and folders

The dashboard reads prepared Parquet files from `processed/`. Some analysis
pages also read optional model files and exported CSVs from
`artifacts/models/`. These generated directories are git-ignored, so they are
not automatically present in a fresh clone or a Streamlit Community Cloud
deployment.

The Home page can start without prepared data, but its analytics require at
least one readable Parquet file with a timestamp column. Pages that use saved
models or scores require their corresponding artifacts.

## Streamlit Community Cloud

1. Push the application source to a GitHub repository.
2. In Streamlit Community Cloud, select that repository and set **Main file
   path** to `Home.py`.
3. Configure the app's Python version to 3.10 or newer, consistent with the
   project dependencies.
4. Arrange secure access to the prepared Parquet data and any optional
   artifacts. Suitable approaches include downloading versioned files from
   controlled object storage during startup or using an approved persistent
   data store. Keep large datasets and model files out of Git.
5. Add credentials through Streamlit's app secrets or the hosting platform's
   secret manager. Never place secrets in source code or commit
   `.streamlit/secrets.toml`.
6. Confirm the app can read its data and navigate to each page that depends on
   optional artifacts.

The current app expects files in local `processed/` and `artifacts/models/`
directories. **It does not automatically download these files from remote
storage.** Configure and test any provisioning step before relying on a hosted
deployment.

## Local deployment check

```bash
python -m pip install -r requirements.txt
streamlit run Home.py
```

For the complete data-generation workflow, install `requirements-research.txt`
and run the notebooks in order. Spark-backed processing also requires a
compatible Java installation; see [`configs/pipeline.yaml`](configs/pipeline.yaml).

## Operational notes

- Use the smallest dependency set required by the deployed app. The research
  environment includes heavier notebook and modelling dependencies.
- Deploy only trusted serialized model artifacts: loading pickle/joblib files
  can execute code.
- Rebuild and validate generated artifacts when the data schema or feature
  pipeline changes.
- Check dataset terms and redistribution permissions before storing source data
  in a hosted environment.
