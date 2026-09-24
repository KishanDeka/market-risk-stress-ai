"""
scripts/process_data.py

Responsibility:
    - Ingest raw market CSVs produced by `download_market_data.py`.
    - Clean string inputs, handle FRED missing markers ('.'), and drop nulls.
    - Align dates across multiple series.
    - Calculate daily Simple Returns, Log Returns, and Yield Changes.
    - Document data lineage in a transformation manifest.

Outputs:
    - data/processed/cleaned_prices.csv
    - data/processed/market_returns.csv
    - data/processed/process_manifest.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_DIR = Path("../data/raw")
PROCESSED_DATA_DIR = Path("../data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean_series(file_path: Path, series_id: str) -> pd.DataFrame:
    """Loads a raw FRED CSV file, handles missing string markers, and parses numeric values."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file missing: {file_path}")

    # Read all as string first to safely parse raw FRED text formats
    df = pd.read_csv(file_path, dtype=str)

    # Standardize column names
    df.columns = [c.upper().strip() for c in df.columns]
    if "DATE" not in df.columns:
        raise ValueError(f"File {file_path} lacks a 'DATE' column.")

    # Convert FRED missing value flag '.' or whitespace into NaN
    df[series_id] = df[series_id].str.strip()
    df[series_id] = df[series_id].replace([".", "", "nan", "None", "null"], np.nan)

    # Coerce to numeric float
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    # Parse dates and drop invalid date rows
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    # Sort chronologically and drop duplicate dates if any
    df = df.sort_values("DATE").drop_duplicates(subset=["DATE"], keep="first")

    return df[["DATE", series_id]]


def main() -> None:
    logger.info("Starting market data processing pipeline...")
    process_time = datetime.now(timezone.utc).isoformat()

    # Discover raw files
    raw_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        logger.error(f"No raw CSV files found in {RAW_DATA_DIR}. Aborting.")
        return

    cleaned_series: List[pd.DataFrame] = []
    lineage_metrics: Dict[str, Any] = {}

    for file_path in raw_files:
        series_id = file_path.stem
        logger.info(f"Cleaning raw series: {series_id}")

        df_cleaned = load_and_clean_series(file_path, series_id)

        raw_count = len(pd.read_csv(file_path))
        clean_count = df_cleaned[series_id].notnull().sum()

        lineage_metrics[series_id] = {
            "raw_rows": raw_count,
            "valid_numeric_rows": int(clean_count),
            "dropped_rows": int(raw_count - clean_count),
        }

        cleaned_series.append(df_cleaned)

    # 1. Merge all series on DATE (Outer join first to inspect total range)
    merged_df = cleaned_series[0]
    for df in cleaned_series[1:]:
        merged_df = pd.merge(merged_df, df, on="DATE", how="outer")

    merged_df = merged_df.sort_values("DATE").reset_index(drop=True)

    # Forward fill up to 2 consecutive missing days (e.g., minor holiday misalignments), then drop remaining NAs
    cleaned_prices = merged_df.ffill(limit=2).dropna().reset_index(drop=True)

    # Save cleaned merged price dataset
    cleaned_prices_path = PROCESSED_DATA_DIR / "cleaned_prices.csv"
    cleaned_prices.to_csv(cleaned_prices_path, index=False)
    logger.info(f"Saved cleaned price series to {cleaned_prices_path}")

    # 2. Calculate Returns
    returns_df = pd.DataFrame({"DATE": cleaned_prices["DATE"]})

    for col in cleaned_prices.columns:
        if col == "DATE":
            continue

        # Check if series is an interest rate/yield (e.g., DGS10, DGS2) or a Price/Index (e.g., SP500)
        if col.startswith("DGS") or "YIELD" in col.upper():
            # For Yields: Calculate absolute change in percentage points (and basis points)
            returns_df[f"{col}_yield_decimal"] = cleaned_prices[col] / 100.0
            returns_df[f"{col}_daily_change_bps"] = (
                cleaned_prices[col].diff() * 100.0
            )  # 1% = 100 bps
        else:
            # For Prices/Indices: Calculate Simple & Log Returns
            returns_df[f"{col}_simple_return"] = cleaned_prices[col].pct_change()
            returns_df[f"{col}_log_return"] = np.log(
                cleaned_prices[col] / cleaned_prices[col].shift(1)
            )

    # Drop the first row which will contain NaN from return calculations
    returns_df = returns_df.dropna().reset_index(drop=True)

    # Save returns dataset
    returns_path = PROCESSED_DATA_DIR / "market_returns.csv"
    returns_df.to_csv(returns_path, index=False)
    logger.info(f"Saved market returns to {returns_path}")

    # 3. Generate Transformation Manifest
    manifest = {
        "processed_at_utc": process_time,
        "input_files": [str(p) for p in raw_files],
        "output_files": [str(cleaned_prices_path), str(returns_path)],
        "aligned_date_range": {
            "start_date": cleaned_prices["DATE"].min().strftime("%Y-%m-%d"),
            "end_date": cleaned_prices["DATE"].max().strftime("%Y-%m-%d"),
            "total_aligned_days": len(cleaned_prices),
        },
        "series_lineage": lineage_metrics,
    }

    manifest_path = PROCESSED_DATA_DIR / "process_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Transformation manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()