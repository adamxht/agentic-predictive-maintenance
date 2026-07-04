# Agentic Predictive Maintenance

A polished end-to-end machine learning project for predictive maintenance and anomaly detection using the NASA CMAPSS dataset. The repository combines data science, MLOps, and an LLM-friendly workflow for exploring descriptive, diagnostic, and predictive analytics.

## ✨ What this project includes

- A complete data science workflow for remaining useful life (RUL) prediction
- Structured notebooks for exploratory data analysis and modeling
- DVC-based data versioning with a local MinIO remote for development
- Modular Python utilities and configuration for repeatable experimentation

## 📁 Project structure

```text
.
├── data/                # Versioned datasets and DVC metadata
├── notebooks/           # EDA and modeling notebooks
├── src/                 # Reusable Python modules
├── minio-data/          # Local MinIO storage for development
├── README.md            # Project overview and setup guide
└── PLAN.md              # Project planning notes
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
dvc remote modify minio access_key_id <USERNAME>
dvc remote modify minio secret_access_key <PASSWORD>
dvc remote modify minio use_ssl false
```

### 5. Track and push data

```bash
dvc add data/raw/
dvc push
```

## 📓 Notebooks

- [notebooks/step1_eda_RUL.ipynb](notebooks/step1_eda_RUL.ipynb) — exploratory data analysis and feature understanding
- [notebooks/step2_modeling_RUL.ipynb](notebooks/step2_modeling_RUL.ipynb) — model training and evaluation

## 📝 Notes

This repository uses a local MinIO instance for development to keep data storage lightweight and inexpensive while preserving a realistic MLOps workflow.