# Production ML System: Demand Forecasting + Full MLOps Platform (AWS)

## Local stack
- MLflow: http://localhost:5001
- Airflow: http://localhost:8080 (admin/admin)

## Run (local)
```bash

# Demand Forecast MLOps (AWS-ready)

This project is a production-style demand forecasting workflow designed to demonstrate both data science and MLOps fundamentals. It includes time-aware validation, strong baseline benchmarking, experiment tracking, and an AWS deployment path.

## What this project does

- Produces demand forecasts at entity level (currently item)
- Uses time-aware train and validation splits
- Benchmarks against common baselines
  - Lag-1 baseline (yesterday’s value)
  - Seasonal naive baseline (same day last week, t-7)
- Tracks metrics and artefacts using MLflow
- Generates resume-ready summary metrics (rows, number of series, MAPE, RMSE, baseline comparisons)

## Current results (baseline benchmark)

Dataset and scope
- Rows (M): 90,600
- Time series (N): 50 item-level series
- Daily entities forecasted (K): 50
- Validation rows used: 18,250
- Latest date in data: 2017-12-31

Validation performance
- Lag-1 baseline: MAPE 2.02%, RMSE 19.46
- Seasonal naive (t-7): MAPE 0.024%, RMSE 1.70

A note on baselines
Seasonal naive performs extremely well on this dataset, which suggests strong weekly seasonality. The next step is to train a learning-based model (for example LightGBM) to compete with or improve on this benchmark.

## Tech stack

Core
- Python, pandas, NumPy, scikit-learn
- MLflow for experiment tracking and artefact logging

Planned production components
- Docker packaging for training and inference
- CI/CD with GitHub Actions
- AWS deployment using S3, ECR and ECS Fargate
- Monitoring and drift reporting

## Repository structure

src/demand_forecast/models/train_baseline.py
Baseline training script with MLflow logging.

src/demand_forecast/metrics/cv_metrics.py
Metrics collector used to report dataset scale and baseline comparisons.

data/processed/splits/train.parquet
data/processed/splits/valid.parquet
Prepared dataset splits.

artifacts/baseline/valid_predictions.csv
Saved predictions for the validation split and logged to MLflow.

## How to run

Activate your environment
```bash
source .venv/bin/activate
