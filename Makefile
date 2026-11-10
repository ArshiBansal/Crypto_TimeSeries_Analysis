# ───────────────────────────────────────────────────────────
# Bitcoin Market Microstructure & Predictive Intelligence
# Quick-start commands
# ───────────────────────────────────────────────────────────

# 1. Install dependencies
install:
	pip install -r requirements.txt

# 2. Run all 4 notebooks in sequence (requires nbconvert + jupyter)
run:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_pipeline.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_engineering.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_modelling.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation_and_export.ipynb

# 3. Launch Streamlit dashboard
app:
	streamlit run Home.py

# 4. Open notebooks interactively
nb:
	jupyter notebook notebooks/

# 5. Clean generated outputs (keeps src/ and notebooks/)
clean:
	rm -rf processed/*.parquet reports/figures/*.html reports/metrics/*.csv
	rm -rf artifacts/models/*.pkl artifacts/scalers/*.pkl
