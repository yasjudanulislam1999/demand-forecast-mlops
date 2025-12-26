# src/demand_forecast/models/train_lgbm.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


SPLIT_DIR = Path("data/processed/splits")
ARTIFACT_DIR = Path("artifacts/lgbm")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLS = [
    "day_of_week",
    "month",
    "day_of_month",
    "demand_total_lag_1",
    "demand_total_lag_7",
    "demand_total_lag_14",
    "demand_total_rollmean_7",
    "demand_total_rollmean_14",
]

TARGET_COL = "demand_total"


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return pd.read_parquet(path)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {"rmse": rmse, "mape": mape}


def main() -> None:
    # log to your docker MLflow server (already exported in terminal)
    mlflow.set_experiment("demand-forecast-lgbm-s3")

    train = load_split("train")
    valid = load_split("valid")

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]

    X_valid = valid[FEATURE_COLS]
    y_valid = valid[TARGET_COL]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": 42,
        "verbose": -1,
    }

    with mlflow.start_run(run_name="lgbm_global_model"):
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("feature_cols", ",".join(FEATURE_COLS))
        mlflow.log_param("target_col", TARGET_COL)

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        # predict
        yhat_train = model.predict(X_train, num_iteration=model.best_iteration)
        yhat_valid = model.predict(X_valid, num_iteration=model.best_iteration)

        train_metrics = evaluate(y_train.values, yhat_train)
        valid_metrics = evaluate(y_valid.values, yhat_valid)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"valid_{k}": v for k, v in valid_metrics.items()})

        # log feature importance
        fi = pd.DataFrame(
            {
                "feature": FEATURE_COLS,
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)

        fi_path = ARTIFACT_DIR / "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        # log predictions sample
        preds = valid[["date", "item"]].copy()
        preds["y_true"] = y_valid.values
        preds["y_pred"] = yhat_valid
        pred_path = ARTIFACT_DIR / "valid_predictions.csv"
        preds.to_csv(pred_path, index=False)
        mlflow.log_artifact(str(pred_path))

        # log model to MLflow
        mlflow.lightgbm.log_model(model, artifact_path="model")

        print("✅ LightGBM training completed")
        print("Best iteration:", model.best_iteration)
        print("Train metrics:", train_metrics)
        print("Valid metrics:", valid_metrics)


if __name__ == "__main__":
    main()
