# src/demand_forecast/features/build_features.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/store_item_demand_train.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "item_daily_features.parquet"


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def aggregate_item_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw store-item demand into item-level daily demand.

    Raw:  date, store, item, demand
    Out:  date, item, demand
    """
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")

    agg = (
        df.groupby(["date", "item"], as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "demand_total"})
        .sort_values(["item", "date"])
        .reset_index(drop=True)
    )
    return agg


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple calendar features derived from date.
    These are safe (they don't look into the future).
    """
    out = df.copy()
    out["day_of_week"] = out["date"].dt.dayofweek  # 0=Mon, 6=Sun
    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day
    return out


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Leakage-safe lag features per item.
    demand_total_lag_k = demand_total at time (t-k)
    """
    out = df.copy()
    for k in lags:
        out[f"demand_total_lag_{k}"] = out.groupby("item")["demand_total"].shift(k)
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Leakage-safe rolling means per item.

    We compute rolling stats on *past* values only by shifting by 1 day first.
    That ensures the current day's target is never included.
    """
    out = df.copy()
    for w in windows:
        past = out.groupby("item")["demand_total"].shift(1)
        out[f"demand_total_rollmean_{w}"] = (
            past.groupby(out["item"]).rolling(window=w, min_periods=w).mean().reset_index(level=0, drop=True)
        )
    return out


def drop_na_created_by_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lags/rolling windows create NaNs at the start of each item series.
    In training, we usually drop those rows (or handle differently).
    """
    feature_cols = [c for c in df.columns if c not in ["date", "item", "demand_total"]]
    return df.dropna(subset=feature_cols).reset_index(drop=True)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(RAW_PATH)
    item_daily = aggregate_item_daily(raw)

    feats = add_time_features(item_daily)
    feats = add_lag_features(feats, lags=[1, 7, 14])
    feats = add_rolling_features(feats, windows=[7, 14])

    feats = drop_na_created_by_features(feats)

    feats.to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Features built: {OUTPUT_PATH}")
    print(f"Rows: {len(feats):,} | Cols: {len(feats.columns)}")
    print("Columns:", list(feats.columns))


if __name__ == "__main__":
    main()
