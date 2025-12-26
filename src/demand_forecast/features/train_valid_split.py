# src/demand_forecast/features/train_valid_split.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


PROCESSED_PATH = Path("data/processed/item_daily_features.parquet")
OUTPUT_DIR = Path("data/processed/splits")

# Validation horizon (days)
VALID_DAYS = 365


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    return pd.read_parquet(path)


def time_based_split(df: pd.DataFrame, valid_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by time:
    - Train: everything before the cutoff date
    - Valid: last `valid_days`
    """
    df = df.sort_values("date").reset_index(drop=True)

    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=valid_days)

    train_df = df[df["date"] <= cutoff_date].copy()
    valid_df = df[df["date"] > cutoff_date].copy()

    return train_df, valid_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features(PROCESSED_PATH)

    train_df, valid_df = time_based_split(df, VALID_DAYS)

    train_path = OUTPUT_DIR / "train.parquet"
    valid_path = OUTPUT_DIR / "valid.parquet"

    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)

    print("✅ Time-aware split completed")
    print(f"Train rows: {len(train_df):,}")
    print(f"Valid rows: {len(valid_df):,}")
    print(f"Train date range: {train_df['date'].min()} → {train_df['date'].max()}")
    print(f"Valid date range: {valid_df['date'].min()} → {valid_df['date'].max()}")


if __name__ == "__main__":
    main()
