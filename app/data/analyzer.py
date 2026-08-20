"""
analyzer.py — Controlled dataframe analysis tool.

Exposes a set of safe, named operations that the agent can call.
The LLM never does the math; all computations happen here.
No arbitrary code execution.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import TARGET_COLUMN, ID_COLUMN

logger = logging.getLogger(__name__)


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to Python native types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, pd.Series):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def _ok(data: Any, description: str = "") -> Dict:
    return {"success": True, "description": description, "data": _to_serializable(data)}


def _err(message: str) -> Dict:
    logger.warning(f"Analyzer error: {message}")
    return {"success": False, "error": message, "data": None}


# ── Column validation ─────────────────────────────────────────────────────────

def validate_column(df: pd.DataFrame, col: str) -> Optional[str]:
    """Return None if column exists, else an error string."""
    if col not in df.columns:
        available = ", ".join(sorted(df.columns.tolist()))
        return f"Column '{col}' not found. Available columns: {available}"
    return None


# ── Individual operations ─────────────────────────────────────────────────────

def get_shape(df: pd.DataFrame) -> Dict:
    return _ok({"rows": df.shape[0], "columns": df.shape[1]}, "Dataset shape")


def get_columns(df: pd.DataFrame) -> Dict:
    cols = []
    for c in df.columns:
        cols.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "n_unique": int(df[c].nunique()),
            "n_missing": int(df[c].isna().sum()),
        })
    return _ok(cols, "Column metadata")


def get_churn_rate(df: pd.DataFrame) -> Dict:
    vc = df[TARGET_COLUMN].value_counts()
    total = len(df)
    yes = int(vc.get("Yes", 0))
    no = int(vc.get("No", 0))
    rate = round(yes / total * 100, 2) if total > 0 else 0.0
    return _ok(
        {"total_customers": total, "churned": yes, "not_churned": no, "churn_rate_pct": rate},
        "Churn rate",
    )


def get_missing_values(df: pd.DataFrame) -> Dict:
    missing = df.isnull().sum()
    missing_cols = {col: int(v) for col, v in missing.items() if v > 0}
    return _ok(
        {"n_columns_with_missing": len(missing_cols), "details": missing_cols},
        "Missing values summary",
    )


def get_average(df: pd.DataFrame, column: str) -> Dict:
    err = validate_column(df, column)
    if err:
        return _err(err)
    if not pd.api.types.is_numeric_dtype(df[column]):
        return _err(f"Column '{column}' is not numeric (dtype={df[column].dtype})")
    val = round(float(df[column].mean()), 4)
    return _ok({"column": column, "mean": val, "n": int(df[column].notna().sum())}, f"Mean of {column}")


def get_median(df: pd.DataFrame, column: str) -> Dict:
    err = validate_column(df, column)
    if err:
        return _err(err)
    if not pd.api.types.is_numeric_dtype(df[column]):
        return _err(f"Column '{column}' is not numeric")
    val = round(float(df[column].median()), 4)
    return _ok({"column": column, "median": val}, f"Median of {column}")


def get_distribution(df: pd.DataFrame, column: str) -> Dict:
    err = validate_column(df, column)
    if err:
        return _err(err)
    if pd.api.types.is_numeric_dtype(df[column]):
        desc = df[column].describe()
        return _ok(
            {
                "column": column,
                "type": "numeric",
                "count": int(desc["count"]),
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4),
                "min": round(float(desc["min"]), 4),
                "q25": round(float(desc["25%"]), 4),
                "median": round(float(desc["50%"]), 4),
                "q75": round(float(desc["75%"]), 4),
                "max": round(float(desc["max"]), 4),
            },
            f"Distribution of {column}",
        )
    else:
        vc = df[column].value_counts(dropna=False)
        total = len(df)
        data = [
            {"value": str(k), "count": int(v), "pct": round(v / total * 100, 2)}
            for k, v in vc.items()
        ]
        return _ok({"column": column, "type": "categorical", "values": data}, f"Distribution of {column}")


def group_by_churn(df: pd.DataFrame, column: str) -> Dict:
    """Compute churn rate for each category of `column`."""
    err = validate_column(df, column)
    if err:
        return _err(err)
    grouped = (
        df.groupby(column)[TARGET_COLUMN]
        .apply(lambda s: round((s == "Yes").mean() * 100, 2))
        .reset_index()
    )
    grouped.columns = [column, "churn_rate_pct"]
    # Also add count per group
    counts = df.groupby(column).size().reset_index(name="count")
    result = grouped.merge(counts, on=column)
    result = result.sort_values("churn_rate_pct", ascending=False)
    return _ok(result.to_dict(orient="records"), f"Churn rate by {column}")


def group_aggregate(
    df: pd.DataFrame,
    group_col: str,
    agg_col: str,
    agg_func: str = "mean",
    filter_col: Optional[str] = None,
    filter_val: Optional[Any] = None,
) -> Dict:
    """Flexible group-by aggregation with optional filter."""
    for col in [group_col, agg_col]:
        err = validate_column(df, col)
        if err:
            return _err(err)

    if filter_col and filter_val is not None:
        err = validate_column(df, filter_col)
        if err:
            return _err(err)
        df = df[df[filter_col] == filter_val]
        if len(df) == 0:
            return _err(f"No rows match filter {filter_col}={filter_val!r}")

    valid_funcs = {"mean", "median", "sum", "count", "min", "max", "std"}
    if agg_func not in valid_funcs:
        return _err(f"agg_func must be one of: {valid_funcs}")

    if not pd.api.types.is_numeric_dtype(df[agg_col]) and agg_func != "count":
        return _err(f"Column '{agg_col}' is not numeric for aggregation '{agg_func}'")

    grouped = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    grouped.columns = [group_col, f"{agg_func}_{agg_col}"]
    grouped[f"{agg_func}_{agg_col}"] = grouped[f"{agg_func}_{agg_col}"].round(4)
    grouped = grouped.sort_values(f"{agg_func}_{agg_col}", ascending=False)
    return _ok(grouped.to_dict(orient="records"), f"{agg_func} of {agg_col} by {group_col}")


def filter_and_count(
    df: pd.DataFrame,
    column: str,
    value: Any,
    operator: str = "eq",
) -> Dict:
    """Count rows matching a condition."""
    err = validate_column(df, column)
    if err:
        return _err(err)

    ops = {
        "eq": df[column] == value,
        "ne": df[column] != value,
        "gt": df[column] > value,
        "gte": df[column] >= value,
        "lt": df[column] < value,
        "lte": df[column] <= value,
        "contains": df[column].astype(str).str.contains(str(value), case=False, na=False),
    }
    if operator not in ops:
        return _err(f"operator must be one of: {list(ops.keys())}")

    mask = ops[operator]
    count = int(mask.sum())
    total = len(df)
    pct = round(count / total * 100, 2) if total > 0 else 0.0
    return _ok(
        {"column": column, "value": value, "operator": operator, "count": count, "total": total, "pct": pct},
        f"Count where {column} {operator} {value!r}",
    )


def top_n_records(
    df: pd.DataFrame,
    sort_col: str,
    n: int = 10,
    ascending: bool = False,
    columns: Optional[List[str]] = None,
) -> Dict:
    """Return top N records sorted by a column."""
    err = validate_column(df, sort_col)
    if err:
        return _err(err)
    cols = columns or list(df.columns)
    for c in cols:
        err = validate_column(df, c)
        if err:
            return _err(err)
    result = df[cols].sort_values(sort_col, ascending=ascending).head(n)
    return _ok(result.to_dict(orient="records"), f"Top {n} records by {sort_col}")


def correlation(df: pd.DataFrame, col1: Optional[str] = None, col2: Optional[str] = None) -> Dict:
    """Pearson correlation between two numeric columns or overall correlation matrix if columns omitted."""
    from app.config import NUMERICAL_FEATURES, TARGET_COLUMN

    if col1 and col2:
        for col in [col1, col2]:
            err = validate_column(df, col)
            if err:
                return _err(err)
            if not pd.api.types.is_numeric_dtype(df[col]):
                return _err(f"Column '{col}' must be numeric for correlation")
        r = round(float(df[col1].corr(df[col2])), 4)
        return _ok({"col1": col1, "col2": col2, "pearson_r": r}, f"Correlation: {col1} vs {col2}")

    # Overall numeric correlations
    df_calc = df.copy()
    if "Churn_Binary" not in df_calc.columns and TARGET_COLUMN in df_calc.columns:
        df_calc["Churn_Binary"] = (df_calc[TARGET_COLUMN] == "Yes").astype(int)

    num_cols = [c for c in NUMERICAL_FEATURES + ["Churn_Binary"] if c in df_calc.columns]
    if not num_cols:
        num_cols = df_calc.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df_calc[num_cols].corr().round(4).to_dict()
    return _ok({
        "correlation_matrix": corr_matrix,
        "features": num_cols,
    }, "Numeric feature correlation matrix")


def segment_comparison(
    df: pd.DataFrame,
    segment_col: str,
    metric_col: str,
    metric_func: str = "mean",
) -> Dict:
    """Compare a metric across segments — wrapper around group_aggregate."""
    return group_aggregate(df, segment_col, metric_col, metric_func)


def count_by_column(df: pd.DataFrame, column: str) -> Dict:
    """Value counts for any column."""
    err = validate_column(df, column)
    if err:
        return _err(err)
    vc = df[column].value_counts(dropna=False)
    total = len(df)
    data = [
        {"value": str(k), "count": int(v), "pct": round(v / total * 100, 2)}
        for k, v in vc.items()
    ]
    return _ok({"column": column, "counts": data}, f"Value counts for {column}")


def average_by_churn(df: pd.DataFrame, column: str) -> Dict:
    """Average of numeric column split by churn status."""
    err = validate_column(df, column)
    if err:
        return _err(err)
    if not pd.api.types.is_numeric_dtype(df[column]):
        return _err(f"Column '{column}' is not numeric")
    grouped = df.groupby(TARGET_COLUMN)[column].mean().round(4).to_dict()
    return _ok({"column": column, "average_by_churn": grouped}, f"Average {column} by churn")


def tenure_trend(df: pd.DataFrame) -> Dict:
    """Churn rate per tenure bucket (0-12, 13-24, 25-36, 37-48, 49-60, 61+)."""
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df2 = df.copy()
    df2["tenure_bucket"] = pd.cut(df2["tenure"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        df2.groupby("tenure_bucket", observed=True)
        .apply(lambda g: pd.Series({
            "count": len(g),
            "churn_rate_pct": round((g[TARGET_COLUMN] == "Yes").mean() * 100, 2),
            "mean_monthly_charges": round(g["MonthlyCharges"].mean(), 2),
        }))
        .reset_index()
    )
    return _ok(grouped.to_dict(orient="records"), "Churn rate by tenure bucket")


# ── Dispatcher ────────────────────────────────────────────────────────────────

OPERATION_MAP = {
    "get_shape": get_shape,
    "get_columns": get_columns,
    "get_churn_rate": get_churn_rate,
    "get_missing_values": get_missing_values,
    "get_average": get_average,
    "get_median": get_median,
    "get_distribution": get_distribution,
    "group_by_churn": group_by_churn,
    "group_aggregate": group_aggregate,
    "filter_and_count": filter_and_count,
    "top_n_records": top_n_records,
    "correlation": correlation,
    "segment_comparison": segment_comparison,
    "count_by_column": count_by_column,
    "average_by_churn": average_by_churn,
    "tenure_trend": tenure_trend,
}


def run_analysis(df: pd.DataFrame, operation: str, **kwargs) -> Dict:
    """
    Central dispatch for all dataframe operations.
    The agent calls this with operation name + parameters.
    """
    if operation not in OPERATION_MAP:
        available = ", ".join(sorted(OPERATION_MAP.keys()))
        return _err(f"Unknown operation '{operation}'. Available: {available}")
    try:
        func = OPERATION_MAP[operation]
        # Only pass kwargs that the function accepts
        import inspect
        sig = inspect.signature(func)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(df, **valid_kwargs)
    except Exception as e:
        logger.exception(f"Error in operation '{operation}': {e}")
        return _err(f"Analysis failed: {str(e)}")
