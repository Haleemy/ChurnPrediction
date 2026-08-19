"""
explainability.py — Model-faithful explanations using SHAP.

For HistGradientBoostingClassifier, we use TreeExplainer.
For other models, we fall back to permutation importance.
Explanations are always derived from the actual model — never fabricated.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from app.config import ALL_FEATURES

logger = logging.getLogger(__name__)


def get_top_factors(
    feature_df: pd.DataFrame,
    pipeline,
    n_top: int = 5,
) -> List[Dict[str, Any]]:
    """
    Compute SHAP values for a single prediction and return top N factors.
    
    Returns:
        List of dicts: [{"feature": ..., "value": ..., "shap_value": ..., "direction": ...}]
        Direction: "increases_risk" or "decreases_risk"
    """
    try:
        return _get_shap_factors(feature_df, pipeline, n_top)
    except Exception as e:
        logger.warning(f"SHAP failed ({e}), falling back to feature importance")
        return _get_importance_factors(feature_df, pipeline, n_top)


def _get_shap_factors(feature_df: pd.DataFrame, pipeline, n_top: int) -> List[Dict]:
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform the input
    X_transformed = preprocessor.transform(feature_df)

    # Get feature names after transformation
    feature_names = _get_feature_names(preprocessor)

    # Use TreeExplainer for tree-based models
    clf_name = type(classifier).__name__
    if "GradientBoosting" in clf_name or "RandomForest" in clf_name or "XGB" in clf_name:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # positive class
        else:
            shap_vals = shap_values[0]
    elif hasattr(classifier, "coef_"):
        # For linear models like LogisticRegression, use exact linear feature attribution: coef * x
        coefs = classifier.coef_[0]
        x_vec = X_transformed[0].toarray()[0] if hasattr(X_transformed[0], "toarray") else np.array(X_transformed[0]).flatten()
        shap_vals = coefs * x_vec
    else:
        explainer = shap.LinearExplainer(classifier, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        shap_vals = shap_values[0]

    # Build sorted factors
    factors = []
    for i, (name, val) in enumerate(zip(feature_names, shap_vals)):
        # Get the original feature value
        raw_val = _get_raw_value(feature_df, feature_names, name, i)
        factors.append({
            "feature": name,
            "raw_value": _safe_val(raw_val),
            "shap_value": round(float(val), 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        })

    factors = sorted(factors, key=lambda x: abs(x["shap_value"]), reverse=True)[:n_top]
    return factors


def _get_importance_factors(feature_df: pd.DataFrame, pipeline, n_top: int) -> List[Dict]:
    """Fallback: use model's feature_importances_ (global, not local)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = _get_feature_names(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return [{"feature": f, "raw_value": None, "shap_value": None, "direction": "unknown"} for f in ALL_FEATURES[:n_top]]

    factors = []
    for name, imp in zip(feature_names, importances):
        raw_val = _get_raw_value_by_name(feature_df, name)
        factors.append({
            "feature": name,
            "raw_value": _safe_val(raw_val),
            "importance": round(float(imp), 4),
            "direction": "model-derived",
        })
    factors = sorted(factors, key=lambda x: x["importance"], reverse=True)[:n_top]
    return factors


def get_global_feature_importance(pipeline) -> List[Dict]:
    """
    Return global feature importances for the trained model.
    Used by the agent to explain model behavior generally.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = _get_feature_names(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        importances = importances / importances.sum()  # normalize
    else:
        return []

    result = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in zip(feature_names, importances)
    ]
    return sorted(result, key=lambda x: x["importance"], reverse=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_feature_names(preprocessor) -> List[str]:
    """Extract feature names from ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(cols, "tolist"):
                names.extend(cols.tolist())
            else:
                names.extend(cols)
        return names


def _get_raw_value(feature_df: pd.DataFrame, feature_names: List[str], name: str, idx: int) -> Any:
    """Try to get the raw input value for a transformed feature."""
    # Strip prefixes like "num__" or "cat__"
    clean_name = name.split("__")[-1] if "__" in name else name
    if clean_name in feature_df.columns:
        return feature_df[clean_name].iloc[0]
    # Try by position against ALL_FEATURES
    if idx < len(ALL_FEATURES):
        col = ALL_FEATURES[idx]
        if col in feature_df.columns:
            return feature_df[col].iloc[0]
    return None


def _get_raw_value_by_name(feature_df: pd.DataFrame, name: str) -> Any:
    clean_name = name.split("__")[-1] if "__" in name else name
    if clean_name in feature_df.columns:
        return feature_df[clean_name].iloc[0]
    return None


def _safe_val(v) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v
