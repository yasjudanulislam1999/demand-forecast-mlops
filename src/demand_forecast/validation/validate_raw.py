# src/demand_forecast/validation/validate_raw.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


RAW_EXPECTED_COLUMNS = ["date", "store", "item", "demand"]


def build_raw_schema() -> DataFrameSchema:
    """
    Define the rules our raw dataset must satisfy.

    Think of this as a contract:
    - if the contract is broken, we stop the pipeline early
    - because bad data can silently poison training/inference
    """
    return DataFrameSchema(
        {
            "date": Column(
                pa.String,
                nullable=False,
                checks=[
                    # date must parse as YYYY-MM-DD
                    Check.str_matches(r"^\d{4}-\d{2}-\d{2}$", error="date must be YYYY-MM-DD"),
                ],
            ),
            "store": Column(
                pa.Int64,
                nullable=False,
                checks=[
                    Check.ge(1, error="store must be >= 1"),
                ],
            ),
            "item": Column(
                pa.Int64,
                nullable=False,
                checks=[
                    Check.ge(1, error="item must be >= 1"),
                ],
            ),
            "demand": Column(
                pa.Int64,
                nullable=False,
                checks=[
                    Check.ge(0, error="demand must be >= 0"),
                ],
            ),
        },
        strict=True,  # no unexpected columns allowed
        coerce=True,  # try to coerce types (e.g., "1" -> 1) then validate
    )


def validate_time_series_uniqueness(df: pd.DataFrame) -> None:
    """
    Time-series sanity check:
    For a given (date, store, item), there should be exactly one row.
    """
    dup_mask = df.duplicated(subset=["date", "store", "item"], keep=False)
    if dup_mask.any():
        examples = df.loc[dup_mask, ["date", "store", "item"]].head(5)
        raise ValueError(
            "Duplicate rows found for (date, store, item). "
            f"Examples:\n{examples.to_string(index=False)}"
        )


def load_raw_csv(path: Path) -> pd.DataFrame:
    """
    Load raw CSV exactly as-is (no cleaning here).
    Raw should be immutable; fixes belong in processed steps later.
    """
    df = pd.read_csv(path)
    return df


def validate_raw_dataset(path: Path) -> None:
    """
    Main entrypoint: load + schema validate + time-series sanity checks.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = load_raw_csv(path)

    # quick column presence check (friendly error)
    missing = [c for c in RAW_EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    schema = build_raw_schema()
    schema.validate(df)

    validate_time_series_uniqueness(df)

    # If we reached here, validation succeeded.
    print(f"✅ Raw data validation passed for: {path}")
    print(f"Rows: {len(df):,} | Columns: {list(df.columns)}")


def main() -> None:
    raw_path = Path("data/raw/store_item_demand_train.csv")
    validate_raw_dataset(raw_path)


if __name__ == "__main__":
    main()
