# Production ML System: Demand Forecasting + Full MLOps Platform (AWS)

## Local stack
- MLflow: http://localhost:5001
- Airflow: http://localhost:8080 (admin/admin)

## Run (local)
```bash
cp .env.example .env
docker compose up --build
# Demand Forecast MLOps (AWS-ready)

Production-style demand forecasting project with reproducible experimentation, baseline benchmarking, and an AWS deployment path.  
Built to demonstrate **Data Science + MLOps fundamentals**: time-aware validation, strong baselines, experiment tracking, artefact logging, and clean project structure.

---

## What this project does

- Creates **time-series demand forecasts** at an entity level (currently: `item`)
- Implements **time-aware train/validation splits**
- Benchmarks against strong baselines:
  - **Lag-1 baseline** (predict yesterday)
  - **Seasonal naïve (t−7)** baseline (predict same day last week)
- Tracks metrics and artefacts using **MLflow**
- Produces **resume-ready metrics** (rows, number of series, MAPE/RMSE, baseline comparisons)

---

## Current results (baseline benchmark)

Dataset & scope:
- **Rows (M):** 90,600  
- **Time series (N):** 50 item-level series  
- **Daily entities forecasted (K):** 50  
- **Validation rows used:** 18,250  

Validation performance:
- **Lag-1 baseline:** **MAPE 2.02%**, **RMSE 19.46**
- **Seasonal naïve (t−7):** **MAPE 0.024%**, **RMSE 1.70**

> Note: Seasonal naïve performs extremely strongly on this dataset, indicating strong weekly seasonality.  
> The next step is to build a learning-based model (e.g., LightGBM) that competes with / improves on the seasonal benchmark.

---

## Tech stack

**Core**
- Python, pandas, NumPy, scikit-learn
- MLflow (experiment tracking + artefact logging)

**MLOps foundations (implemented / planned)**
- Reproducible pipelines and metrics collection (implemented)
- CI/CD with GitHub Actions (planned)
- Docker packaging (planned)
- AWS deployment: S3 + ECR + ECS/Fargate (planned)
- Monitoring & drift reporting (planned)

---

## Repository structure

```text
.
├── src/
│   └── demand_forecast/
│       ├── models/
│       │   └── train_baseline.py          # baseline training + MLflow logging
│       └── metrics/
│           └── cv_metrics.py              # resume-friendly metrics + seasonal baseline benchmarking
├── data/
│   └── processed/
│       └── splits/
│           ├── train.parquet
│           └── valid.parquet
├── artifacts/
│   └── baseline/
│       └── valid_predictions.csv          # saved + logged to MLflow
└── README.md
