"""
loader.py — Dataset loading and validation.
Computes all statistics from actual data; never hardcodes them.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from app.config import DATA_PATH, TARGET_COLUMN, ID_COLUMN

logger = logging.getLogger(__name__)

# Module-level cache so we load once
_df_cache: Optional[pd.DataFrame] = None


def load_dataset(path: Optional[Path] = None, force_reload: bool = False) -> pd.DataFrame:
    """
    Load, clean, and cache the churn dataset.
    
    Cleaning steps (documented):
    1. Strip leading/trailing whitespace from string columns.
    2. Convert TotalCharges to numeric — blank strings become NaN.
    3. For customers with tenure=0 and blank TotalCharges, impute TotalCharges = 0.
    4. Drop remaining rows where TotalCharges is NaN (very few, if any).
    5. Convert SeniorCitizen from 0/1 to 'No'/'Yes' for consistent encoding.
    6. Convert Churn to binary int (0/1) in a separate column for modeling.
    """
    global _df_cache
    if _df_cache is not None and not force_reload:
        return _df_cache.copy()

    data_path = path or DATA_PATH
    logger.info(f"Loading dataset from {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    # ── Strip whitespace ──────────────────────────────────────────────────────
    try:
        str_cols = df.select_dtypes(include="str").columns
    except Exception:
        str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # ── TotalCharges: safe numeric conversion ─────────────────────────────────
    # Blank strings → NaN first
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute: if tenure == 0 and TotalCharges is NaN → TotalCharges = 0
    mask_zero_tenure = (df["tenure"] == 0) & (df["TotalCharges"].isna())
    df.loc[mask_zero_tenure, "TotalCharges"] = 0.0
    logger.info(f"Imputed TotalCharges=0 for {mask_zero_tenure.sum()} zero-tenure records")

    # Drop any remaining NaN TotalCharges
    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with non-numeric TotalCharges")

    # ── SeniorCitizen: keep as numeric but document the encoding ─────────────
    # Some pipelines prefer it as object; we keep as int (0/1) since OrdinalEncoder
    # handles it fine. Just ensure it's integer.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # ── Binary target column ──────────────────────────────────────────────────
    df["Churn_Binary"] = (df[TARGET_COLUMN] == "Yes").astype(int)

    # ── Drop duplicate rows ───────────────────────────────────────────────────
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        logger.warning(f"Found {n_dups} duplicate rows — dropping them")
        df = df.drop_duplicates()

    logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    _df_cache = df.copy()
    return df.copy()


def get_dataset_info(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute and return a structured summary of the dataset.
    All numbers come from the actual data.
    """
    if df is None:
        df = load_dataset()

    n_rows, n_cols = df.shape
    churn_counts = df[TARGET_COLUMN].value_counts()
    churn_rate = (df[TARGET_COLUMN] == "Yes").mean() * 100

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].to_dict()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    try:
        cat_cols = df.select_dtypes(include="str").columns.tolist()
    except Exception:
        cat_cols = df.select_dtypes(include="object").columns.tolist()

    is_id_unique = df[ID_COLUMN].nunique() == len(df)

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "churn_yes": int(churn_counts.get("Yes", 0)),
        "churn_no": int(churn_counts.get("No", 0)),
        "churn_rate_pct": round(float(churn_rate), 2),
        "missing_columns": missing_cols,
        "n_missing_total": int(missing.sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "is_customer_id_unique": bool(is_id_unique),
        "n_duplicate_rows": int(df.duplicated().sum()),
    }


def get_customer_by_id(customer_id: str, df: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
    """Return the raw row for a customer ID, or None if not found."""
    if df is None:
        df = load_dataset()
    matches = df[df[ID_COLUMN].str.upper() == customer_id.strip().upper()]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


def invalidate_cache() -> None:
    """Force reload on next call to load_dataset()."""
    global _df_cache
    _df_cache = None
