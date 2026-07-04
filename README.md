# agentic-predictive-maintenance
End to end machine learning showcasing Data Science and MLOps applied to a predictive maintenance and anomaly detection dataset. LLM is integrated for users to query the different stages of analytics with regards to this dataset. Namely, descriptive analytics, diagnostic analytics, and predictive analytics.

# 1. Track dataset with DVC
For the sake of this project, I will use a local minio (s3 compatible object storage) for development to reduce costs.

Pip install the dvc library:
`pip install dvc[s3]`

Spin up the local minio server:
`docker run -p 9000:9000 -p 9001:9001 -v ./minio-data:/data -e MINIO_ROOT_USER=<USERNAME> -e MINIO_ROOT_PASSWORD=<PASSWORD> minio/minio server /data --console-address ":9001"`

Setup local minio server as remote (Assuming the bucket name is `nasa-cmapss`):
`dvc remote add -d minio s3://nasa-cmapss`

Overwrite s3 endpoint:
`dvc remote modify minio endpointurl http://localhost:9000`

Setup credentials:
`dvc remote modify minio access_key_id <USERNAME>`
`dvc remote modify minio secret_access_key <PASSWORD>`
`dvc remote modify minio use_ssl false`

Push raw datasets (Assuming the datasets are inside the `data/raw/` directory):
`dvc add data/raw/`
`dvc push`