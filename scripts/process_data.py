"""
scripts/download_market_data.py

Responsibility:
    Retrieve and document raw market observations via the official FRED API.

Outputs:
    - data/raw/{series_id}.csv (One raw file per series)
    - data/raw/download_manifest.json (Source, series ID, retrieval time, coverage, file checksum)
    - data/raw/quality_summary.json (Row counts, missing observations, duplicate dates, download failures)

Note:
    - Reads the API key directly from the FRED_API_KEY environment variable.
    - No data synthetic generation, cleaning, or return calculations are performed here.
"""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# FRED API Endpoint Configuration
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Target FRED Series
SERIES_CONFIG = [
    {
        "series_id": "DGS10",
        "source": "FRED",
        "description": "10-Year Treasury Constant Maturity Rate",
    },
    {
        "series_id": "SP500",
        "source": "FRED",
        "description": "S&P 500 Index",
    },
]


def get_fred_api_key() -> str:
    """Retrieves FRED API Key from environment variable or exits gracefully."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.error(
            "Environment variable 'FRED_API_KEY' is not set. "
            "Please export your API key: export FRED_API_KEY='your_key_here'"
        )
        sys.exit(1)
    return api_key


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

        # Preserve raw payload structure: DATE, <SERIES_ID>
        df = pd.DataFrame(observations)[["date", "value"]]
        df.rename(columns={"date": "DATE", "value": series_id}, inplace=True)
        return df, None

    except requests.exceptions.RequestException as e:
        error_msg = f"FRED API Request failed: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


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
        logger.info(f"Retrieving raw series: {series_id}")

        df_raw, failure_reason = fetch_fred_api_series(
            series_id, api_key, requested_start
        )
        raw_filepath = RAW_DATA_DIR / f"{series_id}.csv"

        if df_raw is not None and not df_raw.empty:
            # Save exact raw file without indices or transformations
            df_raw.to_csv(raw_filepath, index=False)

            actual_start = df_raw["DATE"].min() if "DATE" in df_raw.columns else "N/A"
            actual_end = df_raw["DATE"].max() if "DATE" in df_raw.columns else "N/A"
            checksum = calculate_sha256(raw_filepath)

            quality_summary = audit_quality(df_raw, series_id)
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

        manifest_records.append(
            {
                "series_id": series_id,
                "source": config["source"],
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