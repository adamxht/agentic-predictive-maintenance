# Agentic Predictive Maintenance

A polished end-to-end machine learning project for predictive maintenance and anomaly detection using the NASA CMAPSS dataset. The repository combines data science, MLOps, and an LLM-friendly workflow for exploring descriptive, diagnostic, and predictive analytics.

## ✨ What this project includes

- A complete data science workflow for remaining useful life (RUL) prediction
- Structured notebooks for exploratory data analysis and modeling
- A config-driven data preprocessing pipeline that reproduces the notebooks' logic with
  adjustable knobs (target, splits, feature engineering, feature selection) via YAML
- A config-driven model training pipeline (Optuna tuning, SHAP explainability, MLflow
  experiment tracking and model registry) for RandomForest and XGBoost
- DVC-based data versioning with a local MinIO remote for development
- Modular Python utilities and configuration for repeatable experimentation

## 📁 Project structure

```text
.
├── configs/                       # YAML configs for pipeline runs
│   ├── data_transformation/       # Data preprocessing pipeline configs
│   └── model_training/            # Model training pipeline configs
├── data/                          # Versioned datasets and DVC metadata
├── notebooks/                     # EDA and modeling notebooks
├── scripts/                       # Entry-point scripts (run_data_preparation.py, run_model_training.py)
├── src/                           # Reusable Python modules
│   ├── components/                # Individual pipeline steps (ingestion, feature engineering,
│   │                               # model training, evaluation, explainability)
│   ├── models/                    # Model factory + interfaces
│   ├── pipeline/                  # Orchestrators that chain components together
│   └── plots.py                   # Evaluation/explainability plotting functions
├── tests/                         # Unit tests
├── training_logs/                 # Generated plots per training run (gitignored)
├── mlruns/ , mlflow.db             # MLflow tracking store and artifacts (gitignored)
├── minio-data/                    # Local MinIO storage for development
├── README.md                      # Project overview and setup guide
└── PLAN.md                        # Project planning notes
```

## 🚀 Getting started

### 1. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2. Install the required tools

```bash
pip install "dvc[s3]" jupyter
```

### 3. Start a local MinIO server

```bash
docker run -p 9000:9000 -p 9001:9001 \
  -v ./minio-data:/data \
  -e MINIO_ROOT_USER=<USERNAME> \
  -e MINIO_ROOT_PASSWORD=<PASSWORD> \
  minio/minio server /data --console-address ":9001"
```

### 4. Configure DVC with the local MinIO remote

```bash
dvc remote add -d minio s3://nasa-cmapss
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio --local access_key_id <USERNAME>
dvc remote modify minio --local secret_access_key <PASSWORD>
dvc remote modify minio use_ssl false
```

### 5. Track and push data

```bash
dvc add data/raw/
dvc push
```

After pushing, the `nasa-cmapss` bucket in the MinIO console should show the tracked data:

![MinIO object browser showing the nasa-cmapss bucket](images/minio_example.png)

## 📓 Notebooks

- [notebooks/step1_eda_RUL.ipynb](notebooks/step1_eda_RUL.ipynb) — exploratory data analysis and feature understanding
- [notebooks/step2_modeling_RUL.ipynb](notebooks/step2_modeling_RUL.ipynb) — model training and evaluation, predicting raw RUL
- [notebooks/step3_modeling_life_ratio.ipynb](notebooks/step3_modeling_life_ratio.ipynb) — same workflow, predicting `life_ratio` (RUL normalized to [0, 1]) instead

## 🧹 Data preprocessing pipeline

The notebooks' data preparation logic is also available as a config-driven pipeline under
`src/`, so it can be re-run with different settings without editing notebook cells. It
operates on independent per-engine time series (grouped by `engine_id`, ordered by `cycle`),
and every step that fits a statistic (the scaler, feature selection) fits on the training
split only to avoid leakage into validation.

### Pipeline steps (run in the order listed in the config)

1. **`train_validation_split`** — loads `data/raw/train_FD001.txt` and splits by
   `engine_id` (not by row), so no engine's cycles leak across the train/validation split.
2. **`preprocessing`** — computes the target column (`RUL`, or `life_ratio = RUL / max_cycle`,
   bounded [0, 1]) and drops configured columns (operating settings, redundant/constant sensors).
3. **`missing_value_handling`** — fills missing sensor readings per engine: linear
   interpolation for interior gaps, then forward/backward-fill for any leading/trailing gaps
   interpolation can't reach. The current raw data has no missing values, but this keeps the
   pipeline robust if future data does.
4. **`scaling`** — fits a `StandardScaler` on the training split's sensor columns only, then
   applies it to both splits.
5. **`feature_engineering`** — adds per-engine rolling-mean and lag features for each sensor,
   then drops the rows left with missing values from the lag window.
6. **`feature_selection`** — ranks features by `|correlation| * variance` against the target
   (fit on train only) and keeps the top-k.

### Configuring a run

Edit [configs/data_transformation/default.yaml](configs/data_transformation/default.yaml)
(or copy it and point `--config` at your copy). Key knobs:

```yaml
target:
  type: "life_ratio"  # or "rul" to predict raw remaining cycles instead

train_validation_split:
  test_size: 0.2
  random_state: 42

feature_selection:
  top_k: 10

pipeline:
  steps:               # comment a step out to skip it, or reorder them
    - "train_validation_split"
    - "preprocessing"
    - "missing_value_handling"
    - "scaling"
    - "feature_engineering"
    - "feature_selection"
```

### Running it

```bash   # or any environment with requirements.txt installed
python scripts/run_data_preparation.py --config configs/data_transformation/default.yaml
```

This writes (paths configurable under `paths:` in the YAML):

- `data/processed/train.csv`, `data/processed/val.csv` — the processed splits
- `data/processed/artifacts/scaler.pkl` — the fitted `StandardScaler`
- `data/processed/artifacts/selected_features.json` — the selected feature names

Progress, warnings (e.g. missing values filled, zero-variance features), and errors are
logged to both the console and `logs/<timestamp>.log`.

## 🤖 Model training pipeline

Config-driven hyperparameter tuning, evaluation, explainability, and experiment tracking,
built on top of the processed train/val CSVs from the data preprocessing pipeline above.

### What it does, per configured model (`random_forest` and `xgboost` by default)

1. Runs an Optuna hyperparameter search (`n_trials` per model), fitting on train and
   minimizing validation RMSE.
2. Computes regression metrics (RMSE, MAE, R²) on train and validation, plus precision,
   recall, f1, and ROC-AUC on validation by thresholding the continuous prediction into a
   "near failure" binary label (`life_ratio <= threshold`, mirroring the notebook).
3. Computes SHAP values for a sample of validation rows.
4. Generates plots (below) and saves them to `training_logs/<run_name>/<model_name>/plots/`.
5. Logs the model config, hyperparameters, train/validation metrics, dataset lineage, plots,
   and the fitted model to MLflow, with optional Model Registry registration.

Plots, written by [src/plots.py](src/plots.py):

- Actual vs predicted (train and validation)
- Train vs validation RMSE per Optuna trial (tuning convergence / overfit check)
- SHAP beeswarm and bar plots
- Residuals vs true value, absolute error vs cycle, mean absolute error by engine
- Confusion matrix and ROC curve for the near-failure binary classification

### Configuring a run

Edit [configs/model_training/default.yaml](configs/model_training/default.yaml). Key knobs:

```yaml
run_name: "life_ratio_rf_xgb"

models:
  - name: "random_forest"
    n_trials: 100
    search_space:
      n_estimators: { type: "int", low: 100, high: 600 }
      # ...

binary_classification:
  threshold: 0.1 # life_ratio <= threshold => "near failure"
  pred_offset: 0.0

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "cmapss_life_ratio"
  registered_model_name: null # set a name to register the best model
```

### Running it

```bash
python scripts/run_model_training.py --config configs/model_training/default.yaml
```

Each model gets its own MLflow run under the configured experiment. Explore results with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

On this dataset both models land around RMSE ≈ 0.057, R² ≈ 0.96 on the `life_ratio` scale,
closely matching `step3_modeling_life_ratio.ipynb` — exact numbers vary run to run since the
Optuna search isn't seeded.

A single run's logged metrics and parameters:

![MLflow run overview showing validation metrics and XGBoost parameters](images/mlflow_xgb.png)

Comparing runs side by side (RandomForest vs. XGBoost, train and validation metrics):

![MLflow training runs comparison across models](images/mlflow_compare_runs.png)

## ✅ Running the tests

```bash
pytest tests/unit_test.py
```

## 📝 Notes

This repository uses a local MinIO instance for development to keep data storage lightweight and inexpensive while preserving a realistic MLOps workflow.