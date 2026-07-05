# Agentic Predictive Maintenance

A polished end-to-end machine learning project for predictive maintenance and anomaly detection using the NASA CMAPSS dataset. The repository combines data science, MLOps, and an LLM-friendly workflow for exploring descriptive, diagnostic, and predictive analytics.

## ✨ What this project includes

- A complete data science workflow for remaining useful life (RUL) prediction
- Structured notebooks for exploratory data analysis and modeling
- A config-driven data preprocessing pipeline that reproduces the notebooks' logic with
  adjustable knobs (target, splits, feature engineering, feature selection) via YAML
- DVC-based data versioning with a local MinIO remote for development
- Modular Python utilities and configuration for repeatable experimentation

## 📁 Project structure

```text
.
├── configs/                       # YAML configs for pipeline runs
│   └── data_transformation/       # Data preprocessing pipeline configs
├── data/                          # Versioned datasets and DVC metadata
├── notebooks/                     # EDA and modeling notebooks
├── scripts/                       # Entry-point scripts (e.g. run_data_preparation.py)
├── src/                           # Reusable Python modules
│   ├── components/                # Individual pipeline steps (ingestion, feature engineering)
│   └── pipeline/                  # Orchestrators that chain components together
├── tests/                         # Unit tests
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

```bash
conda activate jabil   # or any environment with requirements.txt installed
python scripts/run_data_preparation.py --config configs/data_transformation/default.yaml
```

This writes (paths configurable under `paths:` in the YAML):

- `data/processed/train.csv`, `data/processed/val.csv` — the processed splits
- `data/processed/artifacts/scaler.pkl` — the fitted `StandardScaler`
- `data/processed/artifacts/selected_features.json` — the selected feature names

Progress, warnings (e.g. missing values filled, zero-variance features), and errors are
logged to both the console and `logs/<timestamp>.log`.

### Running the tests

```bash
conda activate jabil
pytest tests/unit_test.py
```

## 📝 Notes

This repository uses a local MinIO instance for development to keep data storage lightweight and inexpensive while preserving a realistic MLOps workflow.