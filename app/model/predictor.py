"""
predictor.py — Clean, callable prediction interface.

Two entry points:
  predict_churn_risk(customer_id)   → prediction for known customer
  predict_customer(features_dict)   → prediction for hypothetical customer

Both return structured dicts — no raw model output exposed.
"""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import (
    MODEL_PATH, METADATA_PATH, ALL_FEATURES,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    ID_COLUMN, get_risk_level, DEFAULT_THRESHOLD,
)
from app.data.loader import load_dataset, get_customer_by_id

logger = logging.getLogger(__name__)


# ── Model cache ────────────────────────────────────────────────────────────────

_pipeline_cache = None
_metadata_cache: Optional[Dict] = None


def load_pipeline(model_path: Path = None):
    global _pipeline_cache
    if _pipeline_cache is None:
        p = model_path or MODEL_PATH
        if Path(p).exists():
            try:
                loaded = joblib.load(p)
                # Validation check: test predict_proba to catch sklearn version attribute mismatches (e.g. SimpleImputer)
                from app.data.loader import load_dataset
                from app.config import ALL_FEATURES
                sample_df = load_dataset().head(1)[ALL_FEATURES]
                loaded.predict_proba(sample_df)
                _pipeline_cache = loaded
            except Exception as e:
                logger.warning(f"Model at {p} failed prediction validation test ({e}). Auto-retraining pipeline...")
                from app.model.train import train_and_save
                train_and_save()
                _pipeline_cache = joblib.load(p)
        else:
            logger.info(f"Model file not found at {p}. Auto-training model pipeline...")
            from app.model.train import train_and_save
            train_and_save()
            _pipeline_cache = joblib.load(p)
        logger.info(f"Model loaded successfully from {p}")
    return _pipeline_cache


def load_metadata(metadata_path: Path = None) -> Dict:
    global _metadata_cache
    if _metadata_cache is None:
        p = metadata_path or METADATA_PATH
        if not Path(p).exists():
            load_pipeline()  # Auto-train pipeline and generate metadata
        if Path(p).exists():
            with open(p) as f:
                _metadata_cache = json.load(f)
        else:
            _metadata_cache = {}
    return _metadata_cache


def _get_threshold() -> float:
    meta = load_metadata()
    return float(meta.get("optimal_threshold", DEFAULT_THRESHOLD))


# ── Core prediction logic ──────────────────────────────────────────────────────

def _predict_from_df(feature_df: pd.DataFrame) -> Dict:
    """
    Core prediction: takes a single-row DataFrame of raw features,
    returns structured prediction dict.
    """
    pipeline = load_pipeline()
    threshold = _get_threshold()

    prob = float(pipeline.predict_proba(feature_df)[:, 1][0])
    prediction = "Likely to churn" if prob >= threshold else "Unlikely to churn"
    risk_level = get_risk_level(prob)

    # ── Top factors via feature importance / SHAP ─────────────────────────────
    try:
        from app.model.explainability import get_top_factors
        top_factors = get_top_factors(feature_df, pipeline)
    except Exception as e:
        logger.warning(f"Could not compute top factors: {e}")
        top_factors = []

    return {
        "risk_score": round(prob, 4),
        "risk_level": risk_level,
        "prediction": prediction,
        "threshold_used": round(threshold, 4),
        "top_factors": top_factors,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def predict_churn_risk(customer_id: str) -> Dict[str, Any]:
    """
    Predict churn risk for an existing customer by ID.
    
    Returns:
        {
          "customer_id": "...",
          "risk_score": 0.78,
          "risk_level": "High",
          "prediction": "Likely to churn",
          "threshold_used": 0.47,
          "top_factors": [...],
          "customer_features": {...}
        }
    Or:
        {"error": "Customer not found", "customer_id": "..."}
    """
    df = load_dataset()
    row = get_customer_by_id(customer_id, df)

    if row is None:
        return {
            "error": f"Customer '{customer_id}' not found in dataset",
            "customer_id": customer_id,
        }

    feature_df = pd.DataFrame([row[ALL_FEATURES]])
    result = _predict_from_df(feature_df)
    result["customer_id"] = str(row[ID_COLUMN])
    result["customer_features"] = {k: _safe_val(row[k]) for k in ALL_FEATURES}
    return result


def predict_customer(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict churn risk for a hypothetical / new customer.
    
    Args:
        features: Dict with keys matching ALL_FEATURES.
                  Missing values are filled with sensible defaults.
    
    Returns: Same structure as predict_churn_risk (without customer_id).
    """
    df = load_dataset()
    # Fill defaults from dataset medians/modes
    defaults = _compute_defaults(df)
    merged = {**defaults, **features}

    # Validate that all required features are present
    missing_feats = [f for f in ALL_FEATURES if f not in merged]
    if missing_feats:
        return {"error": f"Missing required features: {missing_feats}"}

    feature_df = pd.DataFrame([{f: merged[f] for f in ALL_FEATURES}])
    result = _predict_from_df(feature_df)
    result["customer_features"] = {k: _safe_val(merged[k]) for k in ALL_FEATURES}
    return result


def predict_batch(customer_ids: list) -> list:
    """Batch predict for a list of customer IDs."""
    df = load_dataset()
    pipeline = load_pipeline()
    threshold = _get_threshold()

    results = []
    id_to_row = {row[ID_COLUMN]: row for _, row in df.iterrows()}

    for cid in customer_ids:
        row = id_to_row.get(cid)
        if row is None:
            results.append({"customer_id": cid, "error": "Not found"})
            continue
        feature_df = pd.DataFrame([row[ALL_FEATURES]])
        prob = float(pipeline.predict_proba(feature_df)[:, 1][0])
        results.append({
            "customer_id": cid,
            "risk_score": round(prob, 4),
            "risk_level": get_risk_level(prob),
            "prediction": "Likely to churn" if prob >= threshold else "Unlikely to churn",
        })
    return results


def predict_all_customers() -> pd.DataFrame:
    """
    Run batch predictions for all customers.
    Returns a DataFrame with customerID + risk_score + risk_level + prediction.
    """
    df = load_dataset()
    pipeline = load_pipeline()
    threshold = _get_threshold()

    X = df[ALL_FEATURES].copy()
    probs = pipeline.predict_proba(X)[:, 1]

    result_df = df[[ID_COLUMN]].copy()
    result_df["risk_score"] = np.round(probs, 4)
    result_df["risk_level"] = result_df["risk_score"].apply(get_risk_level)
    result_df["prediction"] = np.where(
        probs >= threshold, "Likely to churn", "Unlikely to churn"
    )
    return result_df


# ── Hypothetical analysis ──────────────────────────────────────────────────────

def predict_hypothetical(customer_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare current churn risk for a customer vs. hypothetical scenario.
    
    Args:
        customer_id: Existing customer ID.
        changes: Dict of feature overrides for the hypothetical scenario.
    
    Returns:
        {
          "customer_id": ...,
          "current": {...prediction...},
          "hypothetical": {...prediction...},
          "changes_applied": {...},
          "risk_delta": 0.12,  # positive = higher risk in hypothetical
        }
    """
    df = load_dataset()
    row = get_customer_by_id(customer_id, df)

    if row is None:
        return {"error": f"Customer '{customer_id}' not found", "customer_id": customer_id}

    # Current prediction
    current_features = {f: row[f] for f in ALL_FEATURES}
    current_df = pd.DataFrame([current_features])
    current_result = _predict_from_df(current_df)

    # Hypothetical prediction
    hypo_features = {**current_features, **changes}
    hypo_df = pd.DataFrame([hypo_features])
    hypo_result = _predict_from_df(hypo_df)

    return {
        "customer_id": customer_id,
        "current": current_result,
        "hypothetical": hypo_result,
        "changes_applied": {k: _safe_val(v) for k, v in changes.items()},
        "risk_delta": round(hypo_result["risk_score"] - current_result["risk_score"], 4),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_defaults(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute sensible defaults from the training dataset."""
    defaults = {}
    for col in NUMERICAL_FEATURES:
        defaults[col] = float(df[col].median())
    for col in CATEGORICAL_FEATURES:
        defaults[col] = df[col].mode().iloc[0]
    return defaults


def _safe_val(v):
    """Convert numpy scalars to Python natives."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def invalidate_model_cache():
    global _pipeline_cache, _metadata_cache
    _pipeline_cache = None
    _metadata_cache = None
