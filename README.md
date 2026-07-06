# Agentic Predictive Maintenance

A polished end-to-end machine learning project for predictive maintenance and anomaly detection using the NASA CMAPSS dataset. The repository combines data science, MLOps, and an LLM-friendly workflow for exploring descriptive, diagnostic, and predictive analytics.

## ✨ What this project includes

- A complete data science workflow for remaining useful life (RUL) prediction
- Structured notebooks for exploratory data analysis and modeling
- A config-driven data preprocessing pipeline that reproduces the notebooks' logic with
  adjustable knobs (target, splits, feature engineering, feature selection) via YAML
- A config-driven model training pipeline (Optuna tuning, SHAP explainability, MLflow
  experiment tracking and model registry) for RandomForest and XGBoost
- A test-set evaluation entry point that scores a trained model (from MLflow or a local
  path) against the held-out CMAPSS test set, with the same metrics and plots as training
- A stateless real-time inference API (FastAPI) and a Streamlit demo that replays raw
  engine sensor readings as a live feed, with SHAP explanations and a simulated
  sensor-drift button
- Two Docker Compose stacks under [docker/](docker/) -- a lean one for just the demo (Model Serving API +
  Streamlit) and a full one that also runs this project's MinIO and MLflow services -- with
  no local Python environment needed
- DVC-based data versioning with a local MinIO remote for development
- Modular Python utilities and configuration for repeatable experimentation

## 📁 Project structure

```text
.
├── app/                            # FastAPI inference service + Streamlit demo
├── configs/                       # YAML configs for pipeline runs
│   ├── data_transformation/       # Data preprocessing pipeline configs
│   ├── model_training/            # Model training pipeline configs
│   └── deployment/                # Inference-serving config (model, DB path)
├── data/                          # Versioned datasets and DVC metadata
├── docker/                        # Dockerfiles + Compose stacks (inference-only, full)
├── notebooks/                     # EDA and modeling notebooks
├── scripts/                       # Entry-point scripts (run_data_preparation.py,
│                                  # run_model_training.py, run_test_set_eval.py)
├── src/                           # Reusable Python modules
│   ├── components/                # Individual pipeline steps (ingestion, feature engineering,
│   │                               # test-set ingestion, model training/loading, evaluation,
│   │                               # explainability, inference logging)
│   ├── configs/                   # Pydantic config schemas + YAML loaders
│   ├── models/                    # Model factory + interfaces
│   ├── pipeline/                  # Orchestrators that chain components together
│   └── plots.py                   # Evaluation/explainability plotting functions
├── tests/                         # Unit tests + in-memory integration tests
├── training_logs/ , test_logs/    # Generated plots per run (gitignored)
├── mlruns/ , mlflow.db             # MLflow tracking store and artifacts (gitignored)
├── monitor/                       # Real-time inference log database (gitignored)
├── trained_model/                 # Locally saved models, opt-in (tracked via Git LFS)
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

### 6. Fetch the trained model (Git LFS)

`trained_model/` (the RandomForest/XGBoost artifacts used by the test-set evaluation script,
the inference API, and the Streamlit demo) is tracked with [Git LFS](https://git-lfs.com),
not DVC. Install Git LFS once per machine, then pull the actual model files:

```bash
# One-time per machine
sudo apt install git-lfs
git lfs install

# Fetch the real content:
git lfs pull
```

Verify it worked -- `git lfs ls-files` should list every tracked file, and
`trained_model/life_ratio_rf_xgb/*/model.pkl` should be their real size (tens of KB+), not a
tiny ~130-byte Git LFS pointer text file.

## 📓 Notebooks

- [notebooks/step1_eda_RUL.ipynb](notebooks/step1_eda_RUL.ipynb) - exploratory data analysis and feature understanding
- [notebooks/step2_modeling_RUL.ipynb](notebooks/step2_modeling_RUL.ipynb) - model training and evaluation, predicting raw RUL
- [notebooks/step3_modeling_life_ratio.ipynb](notebooks/step3_modeling_life_ratio.ipynb) - same workflow, predicting `life_ratio` (RUL normalized to [0, 1]) instead

## 🧹 Data preprocessing pipeline

The notebooks' data preparation logic is also available as a config-driven pipeline under
`src/`, so it can be re-run with different settings without editing notebook cells. It
operates on independent per-engine time series (grouped by `engine_id`, ordered by `cycle`),
and every step that fits a statistic (the scaler, feature selection) fits on the training
split only to avoid leakage into validation.

### Pipeline steps (run in the order listed in the config)

1. **`train_validation_split`** - loads `data/raw/train_FD001.txt` and splits by
   `engine_id` (not by row), so no engine's cycles leak across the train/validation split.
2. **`preprocessing`** - computes the target column (`RUL`, or `life_ratio = RUL / max_cycle`,
   bounded [0, 1]) and drops configured columns (operating settings, redundant/constant sensors).
3. **`missing_value_handling`** - fills missing sensor readings per engine: linear
   interpolation for interior gaps, then forward/backward-fill for any leading/trailing gaps
   interpolation can't reach. The current raw data has no missing values, but this keeps the
   pipeline robust if future data does.
4. **`scaling`** - fits a `StandardScaler` on the training split's sensor columns only, then
   applies it to both splits.
5. **`feature_engineering`** - adds per-engine rolling-mean and lag features for each sensor,
   then drops the rows left with missing values from the lag window.
6. **`feature_selection`** - ranks features by `|correlation| * variance` against the target
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
# any environment with requirements.txt installed
python scripts/run_data_preparation.py --config configs/data_transformation/default.yaml
```

This writes (paths configurable under `paths:` in the YAML):

- `data/processed/train.csv`, `data/processed/val.csv` - the processed splits
- `data/processed/artifacts/scaler.pkl` - the fitted `StandardScaler`
- `data/processed/artifacts/selected_features.json` - the selected feature names

Progress, warnings (e.g. missing values filled, zero-variance features), and errors are
logged to both the console and `logs/<timestamp>.log`.

### The held-out test set

The same script also prepares `data/raw/test_FD001.txt` + `data/raw/RUL_FD001.txt` into
`data/processed/test.csv`, via the `test_set:` section of the same config. This set is
*censored* (engines don't run to failure), so it can't reuse `train_validation_split` or
`preprocessing` as-is, instead `test_set_ingestion` reconstructs the target from the
provided terminal RUL answer key (`RUL = rul_at_last_cycle + (last_cycle - cycle)`), and
`scaling`/`feature_selection` reuse the scaler and selected-feature list already fit/chosen
on the training split (`paths.scaler_path` / `paths.selected_features_path`) rather than
refitting on test data:

```yaml
test_set:
  raw_data_path: "data/raw/test_FD001.txt"
  raw_rul_path: "data/raw/RUL_FD001.txt"
  processed_test_path: "data/processed/test.csv"
  pipeline:
    steps:
      - "test_set_ingestion"
      - "missing_value_handling"
      - "scaling"
      - "feature_engineering"
      - "feature_selection"
```

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
    # registered_model_name: "life_ratio_rf_xgb_random_forest"  # optional override
    save_locally: true # opt-in: also save to trained_model/<run_name>/<name>/model.pkl
    search_space:
      n_estimators: { type: "int", low: 100, high: 600 }
      # ...

binary_classification:
  threshold: 0.1 # life_ratio <= threshold => "near failure"
  pred_offset: 0.0

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "cmapss_life_ratio"
```

Each model is registered in the MLflow Model Registry under its own name - if
`registered_model_name` isn't set, it defaults to `<run_name>_<model_name>` (e.g.
`life_ratio_rf_xgb_random_forest`), so RandomForest and XGBoost don't collide under one
name. Load a registered model directly with `mlflow.pyfunc.load_model("models:/<name>/<version>")`.

`save_locally` is off by default; when enabled it *also* writes the fitted model to
`trained_model/<run_name>/<model_name>/model.pkl` (or `save_model_path` if set), so the
test-set evaluation below can score a model without any MLflow dependency at all.

### Running it

```bash
python scripts/run_model_training.py --config configs/model_training/default.yaml
```

Each model gets its own MLflow run under the configured experiment. Explore results with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

On this dataset both models land around RMSE ≈ 0.057, R² ≈ 0.96 on the `life_ratio` scale,
closely matching `step3_modeling_life_ratio.ipynb` - exact numbers vary run to run since the
Optuna search isn't seeded.

A single run's logged metrics and parameters:

![MLflow run overview showing validation metrics and XGBoost parameters](images/mlflow_xgb.png)

Comparing runs side by side (RandomForest vs. XGBoost, train and validation metrics):

![MLflow training runs comparison across models](images/mlflow_compare_runs.png)

## 🧪 Test-set evaluation

Scores an already-trained model against `data/processed/test.csv` (produced by the data
preprocessing pipeline above), computing the same regression + near-failure classification
metrics and plots the training pipeline does - just on the held-out test set instead of
validation, and for one model instead of a list.

The model can come from either MLflow or a local path, so this script has no MLflow
dependency at all when using the latter:

```bash
# From MLflow (registered model)
python scripts/run_test_set_eval.py --model "models:/life_ratio_xgboost/1"

# From a local path (requires save_locally: true during training)
python scripts/run_test_set_eval.py --model trained_model/life_ratio_rf_xgb/xgboost
```

Metrics are printed as a table:

```text
╒═══════════╤═════════╕
│ Metric    │   Value │
╞═══════════╪═════════╡
│ RMSE      │  0.0667 │
├───────────┼─────────┤
│ MAE       │  0.0479 │
├───────────┼─────────┤
│ R2        │  0.9142 │
├───────────┼─────────┤
│ ACCURACY  │  0.9947 │
├───────────┼─────────┤
│ PRECISION │  0.7025 │
├───────────┼─────────┤
│ RECALL    │  0.7265 │
├───────────┼─────────┤
│ F1        │  0.7143 │
├───────────┼─────────┤
│ ROC_AUC   │  0.9968 │
╘═══════════╧═════════╛
```

Every setting is available as a CLI flag (`--threshold`, `--pred-offset`, `--target-type`,
`--sample-size`, `--plots-output-dir`, `--no-plots`, ...) with sensible
defaults. An optional `--config path/to.yaml` can override any subset of them - **any field
present in that YAML takes precedence over the matching CLI flag**, so a config only needs
to set the values it wants to change:

```yaml
# overrides.yaml
model: "models:/life_ratio_xgboost/1"
threshold: 0.15
```

```bash
python scripts/run_test_set_eval.py --threshold 0.1 --config overrides.yaml
# runs with threshold=0.15 (config wins), not 0.1
```

Plots are written to `test_logs/<model>/plots/`, where `<model>` is derived from `--model`
itself (e.g. `trained_model/life_ratio_rf_xgb/xgboost` -> `life_ratio_rf_xgb/xgboost`, or
`models:/life_ratio_xgboost/1` -> `life_ratio_xgboost/1`) via
[src/plots.py](src/plots.py), the same functions the training pipeline uses. Metrics are
logged to the console and `logs/<timestamp>.log`.

## 🔌 Real-time inference (API + demo)

[src/pipeline/inference_pipeline.py](src/pipeline/inference_pipeline.py) turns a raw sensor
reading window into a model-ready feature row, reusing the same
[src/components/feature_engineering.py](src/components/feature_engineering.py) functions
as training/evaluation. It's **stateless**: every call takes the full window of recent raw
readings for one engine (oldest to newest) and recomputes rolling/lag features from
scratch, holding no per-engine history between calls, the client owns that buffer. The
scaler and selected-feature list come from the serving model's own bundled `preprocessor/`
folder (see the "Model training pipeline" section above), not a standalone path, so a
`model` reference is the single point of truth, fully decoupled from
[configs/data_transformation/default.yaml](configs/data_transformation/default.yaml).

### The API

[app/api.py](app/api.py) exposes it over FastAPI, fully configured via
[configs/deployment/default.yaml](configs/deployment/default.yaml) - which model to load,
MLflow tracking URI, log database path, preprocessing steps, and the rolling-window/lag
settings (which must match what the model was trained with):

```bash
uvicorn app.api:app --port 8000
```

- `GET /config` - the settings a client needs to build valid requests: `required_window_length`,
  `sensor_columns`, `selected_features`.
- `POST /predict` - body is `{"engine_id": int, "readings": [{"cycle": int, "values": {...}}, ...]}`;
  returns the predicted `life_ratio` plus a per-feature SHAP breakdown for that cycle.
- `GET /health` - liveness check.

Every prediction is logged to a SQLite database (`monitor/inference_log.db` by default) via
[src/components/inference_store.py](src/components/inference_store.py): a wide
`inference_readings` table (raw, unscaled sensor values + prediction, one row per cycle) for
plotting, and a long/normalized `inference_shap_values` table (one row per feature per
prediction) so an LLM agent can query SHAP trends with plain SQL later without pivoting.
That's the intended next step this design sets up for - an agent that watches this database
for drift and explains it via SHAP - but it's out of scope here; this piece only covers the
inference pipeline, API, and demo.

### The demo

[app/streamlit.py](app/streamlit.py) replays three engines (75, 25, 26) as a live feed (1
cycle every N seconds, adjustable in the UI, defaulting to 10s) against the API. It reads
them from the raw *training* file rather than the test set, because the test set is
censored - e.g. engine 75 only has data up to cycle 88 in the test set - so it never
actually reaches failure. The training file has each engine's full, uncensored run to
actual failure:

```bash
streamlit run app/streamlit.py
```

It shows the live predicted life-ratio, a SHAP bar chart for the current cycle, and each raw
sensor's trend as its own small chart (unscaled, so actual magnitudes are visible). When the
predicted life ratio drops below `life_ratio_threshold` (set in
[configs/deployment/default.yaml](configs/deployment/default.yaml), default `0.1`, matching
the training config's near-failure threshold), the UI flags it as a predicted engine
failure. A **drift** control forces one chosen sensor to 0.0 in every subsequent reading --
watch its SHAP contribution and its own trend chart react in real time.

### Running the demo with Docker Compose

No local Python environment needed -- just Docker and the model artifacts. Everything
Docker-related lives under [docker/](docker/); run these from the repo root:

```bash
# One-time: fetch the real trained_model/ files (see "Fetch the trained model" above)
git lfs pull

# Just the demo (api + streamlit):
docker compose -f docker/docker-compose.inference.yml up --build

# Or the whole local dev stack (also minio + mlflow):
docker compose -f docker/docker-compose.full.yml up --build
```

Either way, open **http://localhost:8501** for the Streamlit demo. The `full` compose file
additionally starts `minio` (http://localhost:9001, this project's DVC remote) and `mlflow`
(http://localhost:5000, an MLflow UI for past training runs). For pure inferencing, just run the `docker-compose.inference.yaml` version.

There are two Dockerfiles: [docker/Dockerfile.inference](docker/Dockerfile.inference) is a
lean, serving-only image (built from
[docker/requirements_inference.txt](docker/requirements_inference.txt), a trimmed dependency
set with the DVC/Jupyter/training-only tooling stripped out) used by the `model-server` and
`streamlit` services in *both* compose files. [docker/Dockerfile.full](docker/Dockerfile.full)
(the whole root [requirements.txt](requirements.txt)) is only used by `mlflow`, since that
service mirrors the local dev stack rather than serving anything. Both Dockerfiles build
with the repo root as context (`context: ..` in the compose files), since that's where
`.dockerignore` and the source it copies actually live. `streamlit` waits for
`model-server`'s `/health` check before starting, via
`INFERENCE_API_URL=http://model-server:8000` (the service name as hostname on the Compose
network).

## ✅ Running the tests

```bash
pytest tests/unit_test.py         # component-level unit tests
pytest tests/integration_test.py  # full data-prep -> train -> eval flow, gatekeeping
pytest tests/                     # everything
```

**Unit tests** (`unit_test.py`) exercise individual `src/components/` functions in
isolation with small, hand-built DataFrames - fast, no I/O, pinpoint exactly which
transformation broke.

**Integration tests** (`integration_test.py`) run the real `DataPreparationPipeline` and
`TestSetPreparationPipeline` against the actual `data/raw/*.txt` files end-to-end (outputs
redirected to a temp dir, never touching `data/processed/`), then fit RandomForest/XGBoost
with their real best hyperparameters (hardcoded, not re-searched) and assert the metrics
against a baseline captured the same way. These exist to gatekeep a model before release —
representative of actual performance, not just correctness of the code path - so they
require `data/raw/{train,test,RUL}_FD001.txt` to be present.

> **Note:** production would have CI run `dvc pull` against a persistent remote, but this
> project's MinIO is local-dev-only with no server for CI to reach. Since FD001 is small
> (a few MB), it's committed directly to git as a temporary workaround.
>
> This is really a special case of a general pattern: real datasets are often too large
> (>TB) for CI to ever pull, so production teams commit a small, curated fixture subset
> instead - which must still preserve whatever structure the pipeline depends on (here,
> complete per-engine traces, not sampled rows) to stay representative of real performance.

## Preliminary results

Test-set evaluation (`life_ratio` target) for both models trained on FD001:

| Metric    | RandomForest | XGBoost    |
| --------- | ------------ | ---------- |
| RMSE      | 0.0668       | **0.0667** |
| MAE       | **0.0476**   | 0.0479     |
| R2        | 0.9140       | **0.9142** |
| Accuracy  | 0.9946       | **0.9947** |
| Precision | **0.7117**   | 0.7025     |
| Recall    | 0.6752       | **0.7265** |
| F1        | 0.6930       | **0.7143** |
| ROC AUC   | 0.9968       | 0.9968     |

XGBoost edges out RandomForest on RMSE, R2, and the near-failure classification metrics
(recall, F1), while RandomForest is marginally better on MAE and precision. Since RUL
prediction is fundamentally a regression problem, **XGBoost** is the better overall model
here -- its plots are shown below.

**Error by engine** - MAE per engine on the test set, worst and best:

![XGBoost mean absolute error by engine](images/xgboost_error_by_engine.png)

Engines 25, 75, and 26 have the lowest errors.

**SHAP feature importance** - mean absolute SHAP value per feature:

![XGBoost SHAP bar plot](images/xgboost_shap_bar.png)

Aside from `cycle`, `Ps30` (static pressure at HPC outlet) is the most important sensor
feature, based on 10 samples.

## 📝 Notes

This repository uses a local MinIO instance for development to keep data storage lightweight and inexpensive while preserving a realistic MLOps workflow.