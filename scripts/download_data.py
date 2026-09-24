"""
scripts/download_market_data.py

Responsibility:
    Retrieve raw market observations via the official FRED API or generate 
    synthetic fallbacks for restricted series, documenting all metadata.

Outputs:
    - data/raw/{series_id}.csv (One raw file per series)
    - data/raw/download_manifest.json (Metadata, checksums, coverage)
    - data/raw/quality_summary.json (Row counts, nulls, duplicates)

Note:
    - Reads the API key directly from the FRED_API_KEY environment variable.
    - Uses synthetic generation for restricted series to allow open-source publishing.
    - Cleaning and return calculations belong in the next script.
    - Exploratory plots belong in notebooks.
"""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Directory Configurations
RAW_DATA_DIR = Path("../data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# FRED API Configuration
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Target Series Configuration
# Public domain series download live via FRED; restricted redistribution series use synthetic generation.
SERIES_CONFIG = [
    {
        "series_id": "DGS10",
        "source": "FRED",
        "description": "10-Year Treasury Constant Maturity Rate",
        "synthetic": False,
    },
    {
        "series_id": "SP500",
        "source": "FRED",
        "description": "S&P 500 Index (Restricted Redistribution)",
        "synthetic": True,  # Enables synthetic data generation for open-source safety
    },
]


def get_fred_api_key() -> str:
    """Retrieves FRED API Key from environment variable or exits gracefully if required."""
    api_key = os.getenv("FRED_API_KEY")
    # Only enforce key presence if at least one non-synthetic series is configured
    if not api_key and any(not s["synthetic"] for s in SERIES_CONFIG):
        logger.error(
            "Environment variable 'FRED_API_KEY' is not set. "
            "Please export your API key: export FRED_API_KEY='your_key_here'"
        )
        sys.exit(1)
    return api_key or ""


def calculate_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash for raw file checksum verification."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def fetch_fred_api_series(
    series_id: str,
    api_key: str,
    observation_start: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Retrieves raw observations directly from FRED REST API."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if observation_start:
        params["observation_start"] = observation_start

    try:
        response = requests.get(FRED_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        observations = data.get("observations", [])
        if not observations:
            return None, "No observations returned from FRED API"

        # Format to mirror raw FRED CSV structure: DATE, <SERIES_ID>
        df = pd.DataFrame(observations)[["date", "value"]]
        df.rename(columns={"date": "DATE", "value": series_id}, inplace=True)
        return df, None

    except requests.exceptions.RequestException as e:
        error_msg = f"FRED API Request failed: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def generate_synthetic_series(
    series_id: str,
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates synthetic price data mirroring market dynamics via Geometric Brownian Motion.
    Ensures repository can be published publicly without bundling restricted raw data.
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    np.random.seed(42)  # Fixed seed for deterministic, reproducible repository runs

    # Geometric Brownian Motion simulation
    returns = np.random.normal(0.0004, 0.012, size=len(dates))
    price_paths = 2000.0 * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "DATE": dates.strftime("%Y-%m-%d"),
            series_id: np.round(price_paths, 2).astype(str),
        }
    )

    # Insert raw missing flag '.' to mimic FRED raw output behavior
    if len(df) > 10:
        df.loc[10, series_id] = "."

    return df


def audit_quality(df: pd.DataFrame, series_col: str) -> Dict[str, Any]:
    """Calculates basic quality summary metrics on raw observations."""
    total_rows = len(df)

    # Missing observations: Nulls, empty strings, or standard FRED missing flags '.'
    raw_vals = df[series_col].astype(str).str.strip()
    missing_count = int(
        df[series_col].isnull().sum() + (raw_vals.isin([".", "", "nan", "None"])).sum()
    )

    duplicate_dates = int(df["DATE"].duplicated().sum()) if "DATE" in df.columns else 0

    return {
        "total_row_count": total_rows,
        "missing_observations_count": missing_count,
        "duplicate_dates_count": duplicate_dates,
    }


def main(requested_start: str = "2010-01-01") -> None:
    api_key = get_fred_api_key()
    manifest_records: List[Dict[str, Any]] = []
    quality_records: Dict[str, Any] = {}

    retrieval_time = datetime.now(timezone.utc).isoformat()

    for config in SERIES_CONFIG:
        series_id = config["series_id"]
        is_synthetic = config["synthetic"]
        failure_reason = None
        df_raw = None

        logger.info(f"Processing series: {series_id} (Synthetic={is_synthetic})")

        # 1. Acquire Data (API vs. Synthetic Generation)
        if is_synthetic:
            df_raw = generate_synthetic_series(series_id, start_date=requested_start)
        else:
            df_raw, failure_reason = fetch_fred_api_series(
                series_id, api_key, requested_start
            )

        raw_filepath = RAW_DATA_DIR / f"{series_id}.csv"

        if df_raw is not None and not df_raw.empty:
            # Save exact raw file without index or transformations
            df_raw.to_csv(raw_filepath, index=False)

            # Extract coverage and verify file integrity
            actual_start = df_raw["DATE"].min() if "DATE" in df_raw.columns else "N/A"
            actual_end = df_raw["DATE"].max() if "DATE" in df_raw.columns else "N/A"
            checksum = calculate_sha256(raw_filepath)

            # Audit quality
            series_col = (
                series_id if series_id in df_raw.columns else df_raw.columns[1]
            )
            quality_summary = audit_quality(df_raw, series_col)
            quality_summary["download_status"] = "SUCCESS"
            quality_summary["failure_reason"] = None

        else:
            checksum = None
            actual_start = None
            actual_end = None
            quality_summary = {
                "total_row_count": 0,
                "missing_observations_count": 0,
                "duplicate_dates_count": 0,
                "download_status": "FAILED",
                "failure_reason": failure_reason,
            }

        # 2. Record Manifest Metadata
        manifest_records.append(
            {
                "series_id": series_id,
                "source": config["source"],
                "is_synthetic": is_synthetic,
                "retrieval_time_utc": retrieval_time,
                "requested_coverage": {"start_date": requested_start},
                "actual_coverage": {
                    "start_date": actual_start,
                    "end_date": actual_end,
                },
                "file_path": str(raw_filepath) if df_raw is not None else None,
                "sha256_checksum": checksum,
            }
        )

        quality_records[series_id] = quality_summary

    # Save Output Manifest
    manifest_path = RAW_DATA_DIR / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "generated_at": retrieval_time,
                "series_count": len(manifest_records),
                "series": manifest_records,
            },
            f,
            indent=2,
        )

    # Save Quality Summary
    quality_path = RAW_DATA_DIR / "quality_summary.json"
    with open(quality_path, "w") as f:
        json.dump(
            {"evaluated_at": retrieval_time, "metrics": quality_records}, f, indent=2
        )

    logger.info(f"Done. Manifest saved to {manifest_path}")
    logger.info(f"Quality Summary saved to {quality_path}")


if __name__ == "__main__":
    main()