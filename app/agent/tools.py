"""
tools.py — Tool executor.

Maps tool names to actual Python functions.
All tools return structured dicts: {"success": bool, "data": ..., "error": ...}
The agent calls these; results go to the verifier.
"""
import json
import logging
from typing import Any, Dict, Optional

from app.data.loader import load_dataset, get_dataset_info as _get_dataset_info
from app.data.analyzer import run_analysis
from app.model.predictor import (
    predict_churn_risk,
    predict_customer,
    predict_hypothetical,
    predict_all_customers,
    load_metadata,
)
from app.model.explainability import get_global_feature_importance, get_top_factors
from app.visualization.charts import generate_chart as _generate_chart

logger = logging.getLogger(__name__)


def _ok(data: Any, description: str = "") -> Dict:
    return {"success": True, "description": description, "data": data}


def _err(message: str) -> Dict:
    return {"success": False, "error": message, "data": None}


# ── Tool implementations ──────────────────────────────────────────────────────

def tool_analyze_data(operation: str, **kwargs) -> Dict:
    """Dispatch to the dataframe analyzer."""
    try:
        df = load_dataset()
        result = run_analysis(df, operation, **kwargs)
        return result
    except Exception as e:
        logger.exception(f"tool_analyze_data error: {e}")
        return _err(f"Data analysis error: {str(e)}")


def tool_predict_customer_risk(customer_id: str) -> Dict:
    """Predict churn risk for existing customer."""
    try:
        result = predict_churn_risk(customer_id)
        if "error" in result:
            return _err(result["error"])
        return _ok(result, f"Churn risk for customer {customer_id}")
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        logger.exception(f"tool_predict_customer_risk error: {e}")
        return _err(f"Prediction error: {str(e)}")


def tool_predict_hypothetical(customer_id: str, changes: Dict[str, Any]) -> Dict:
    """Compare current vs. hypothetical risk."""
    try:
        result = predict_hypothetical(customer_id, changes)
        if "error" in result:
            return _err(result["error"])
        return _ok(result, f"Hypothetical analysis for {customer_id}")
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        logger.exception(f"tool_predict_hypothetical error: {e}")
        return _err(f"Hypothetical prediction error: {str(e)}")


def tool_predict_new_customer(features: Dict[str, Any]) -> Dict:
    """Predict for a new/hypothetical customer."""
    try:
        result = predict_customer(features)
        if "error" in result:
            return _err(result["error"])
        return _ok(result, "New customer churn prediction")
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        logger.exception(f"tool_predict_new_customer error: {e}")
        return _err(f"New customer prediction error: {str(e)}")


def tool_get_top_risk_customers(n: int = 10) -> Dict:
    """Return top N highest-risk customers + overall risk segment counts."""
    try:
        n = max(1, min(int(n), 100))  # Safety cap
        df = predict_all_customers()
        risk_counts = {k: int(v) for k, v in df["risk_level"].value_counts().to_dict().items()}
        top = df.nlargest(n, "risk_score")
        # Enrich with a few original features
        raw_df = load_dataset()
        enriched = top.merge(
            raw_df[["customerID", "Contract", "MonthlyCharges", "tenure", "InternetService"]],
            on="customerID", how="left",
        )
        return _ok({
            "risk_distribution": risk_counts,
            "total_customers": len(df),
            "top_customers": enriched.to_dict(orient="records"),
        }, f"Top {n} highest-risk customers and overall risk segment counts")
    except FileNotFoundError as e:
        return _err(str(e))
    except Exception as e:
        logger.exception(f"tool_get_top_risk_customers error: {e}")
        return _err(f"Error getting top risk customers: {str(e)}")


def tool_generate_chart(chart_type: str, column: Optional[str] = None, title: Optional[str] = None, **kwargs) -> Dict:
    """Generate a chart from actual data."""
    try:
        chart_data = _generate_chart(chart_type, column=column, title=title, **kwargs)
        if chart_data is None:
            return _err(f"Chart generation failed for type '{chart_type}'")
        return _ok(chart_data, f"Chart: {chart_type}")
    except Exception as e:
        logger.exception(f"tool_generate_chart error: {e}")
        return _err(f"Chart error: {str(e)}")


def tool_get_model_info() -> Dict:
    """Return model metadata and feature importance."""
    try:
        metadata = load_metadata()
        if not metadata:
            return _err("Model metadata not found. Run training first.")
        # Add global feature importance
        from app.model.predictor import load_pipeline
        try:
            pipeline = load_pipeline()
            importance = get_global_feature_importance(pipeline)
            metadata["global_feature_importance"] = importance[:10]
        except Exception:
            pass
        return _ok(metadata, "Model information")
    except Exception as e:
        logger.exception(f"tool_get_model_info error: {e}")
        return _err(f"Model info error: {str(e)}")


def tool_get_dataset_info() -> Dict:
    """Return dataset summary statistics."""
    try:
        df = load_dataset()
        info = _get_dataset_info(df)
        return _ok(info, "Dataset summary")
    except Exception as e:
        logger.exception(f"tool_get_dataset_info error: {e}")
        return _err(f"Dataset info error: {str(e)}")


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "analyze_data": tool_analyze_data,
    "predict_customer_risk": tool_predict_customer_risk,
    "predict_hypothetical": tool_predict_hypothetical,
    "predict_new_customer": tool_predict_new_customer,
    "get_top_risk_customers": tool_get_top_risk_customers,
    "generate_chart": tool_generate_chart,
    "get_model_info": tool_get_model_info,
    "get_dataset_info": tool_get_dataset_info,
}


def execute_tool(tool_name: str, **kwargs) -> Dict:
    """
    Central tool executor with error handling.
    Returns structured result dict.
    """
    if tool_name not in TOOL_REGISTRY:
        available = ", ".join(sorted(TOOL_REGISTRY.keys()))
        return _err(f"Unknown tool '{tool_name}'. Available: {available}")
    try:
        func = TOOL_REGISTRY[tool_name]
        return func(**kwargs)
    except Exception as e:
        logger.exception(f"Tool '{tool_name}' failed: {e}")
        return _err(f"Tool execution failed: {str(e)}")
