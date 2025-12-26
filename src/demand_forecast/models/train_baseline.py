# src/demand_forecast/models/train_baseline.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np


SPLIT_DIR = Path("data/processed/splits")
ARTIFACT_DIR = Path("artifacts/baseline")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return pd.read_parquet(path)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": rmse, "mape": mape}


def main() -> None:
    mlflow.set_experiment("demand-forecast-baseline")

    train = load_split("train")
    valid = load_split("valid")

    # Baseline: predict using lag-1
    y_train = train["demand_total"].values
    yhat_train = train["demand_total_lag_1"].values

    y_valid = valid["demand_total"].values
    yhat_valid = valid["demand_total_lag_1"].values

    with mlflow.start_run(run_name="lag1_baseline"):
        mlflow.log_param("model_type", "lag1_baseline")
        mlflow.log_param("target", "demand_total")
        mlflow.log_param("feature", "demand_total_lag_1")

        train_metrics = evaluate(y_train, yhat_train)
        valid_metrics = evaluate(y_valid, yhat_valid)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"valid_{k}": v for k, v in valid_metrics.items()})

        # Save predictions as artefact
        preds = valid[["date", "item"]].copy()
        preds["y_true"] = y_valid
        preds["y_pred"] = yhat_valid

        pred_path = ARTIFACT_DIR / "valid_predictions.csv"
        preds.to_csv(pred_path, index=False)
        mlflow.log_artifact(str(pred_path))

        print("✅ Baseline training completed")
        print("Train metrics:", train_metrics)
        print("Valid metrics:", valid_metrics)


if __name__ == "__main__":
    main()
