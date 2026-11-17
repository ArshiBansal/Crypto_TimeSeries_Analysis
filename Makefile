# ───────────────────────────────────────────────────────────
# Bitcoin Market Microstructure & Predictive Intelligence
# Quick-start commands
# ───────────────────────────────────────────────────────────

# 1. Install dependencies
install:
	pip install -r requirements.txt

# 1b. Install research dependencies (notebooks + Spark + models)
install-research:
	pip install -r requirements-research.txt

# 2. Run all 4 notebooks in sequence (requires nbconvert + jupyter)
#    Prerequisite: Java 17 or 21 on PATH (or set spark.java_home in configs/pipeline.yaml)
run:
	@python scripts/prepare_data.py --full

# 2b. Run notebooks only (alias for the provisioning script)
run-notebooks:
	python scripts/prepare_data.py --full

# 2c. Quick audit run (streaming sample, fast)
run-audit:
	python scripts/prepare_data.py

# 3. Launch Streamlit dashboard
app:
	streamlit run Home.py

# 4. Open notebooks interactively
nb:
	jupyter notebook notebooks/

# 5. Run tests
test:
	pytest

# 6. Clean generated outputs (keeps src/ and notebooks/)
clean:
	@if exist "processed" rmdir /s /q processed
	@if exist "artifacts" rmdir /s /q artifacts
	@if exist "reports" rmdir /s /q reports
	@echo "Cleaned generated directories."

# 7. Check Java version (precondition for Spark)
check-java:
	@java -version 2>&1 || echo "Java not found. Install Java 17 or 21."