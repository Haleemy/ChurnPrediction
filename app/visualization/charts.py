"""
charts.py — Chart generation from actual data.

All charts are generated from real dataframe/model computations.
Returns Plotly figure dicts (JSON-serializable) for Streamlit rendering.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import TARGET_COLUMN, NUMERICAL_FEATURES

logger = logging.getLogger(__name__)


def _fig_to_dict(fig) -> Dict:
    """Convert Plotly figure to JSON-serializable dict."""
    return fig.to_dict()


def chart_churn_distribution(df: pd.DataFrame, title: str = None) -> Dict:
    """Pie chart of churn vs. no churn."""
    vc = df[TARGET_COLUMN].value_counts()
    fig = px.pie(
        names=vc.index,
        values=vc.values,
        title=title or "Customer Churn Distribution",
        color=vc.index,
        color_discrete_map={"Yes": "#E74C3C", "No": "#2ECC71"},
        hole=0.4,
    )
    fig.update_layout(
        font_family="Inter, sans-serif",
        title_font_size=16,
        legend_title_text="Churn",
    )
    return _fig_to_dict(fig)


def chart_churn_by_column(df: pd.DataFrame, column: str, title: str = None) -> Dict:
    """Bar chart: churn rate by category."""
    if column not in df.columns:
        return None
    grouped = (
        df.groupby(column)[TARGET_COLUMN]
        .apply(lambda s: round((s == "Yes").mean() * 100, 2))
        .reset_index()
    )
    grouped.columns = [column, "churn_rate_pct"]
    counts = df.groupby(column).size().reset_index(name="count")
    grouped = grouped.merge(counts, on=column).sort_values("churn_rate_pct", ascending=False)

    fig = px.bar(
        grouped,
        x=column,
        y="churn_rate_pct",
        color="churn_rate_pct",
        color_continuous_scale="RdYlGn_r",
        text="churn_rate_pct",
        title=title or f"Churn Rate by {column}",
        labels={"churn_rate_pct": "Churn Rate (%)"},
        hover_data={"count": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        font_family="Inter, sans-serif",
        xaxis_tickangle=-30,
        coloraxis_showscale=False,
        title_font_size=16,
    )
    return _fig_to_dict(fig)


def chart_distribution(df: pd.DataFrame, column: str, title: str = None) -> Dict:
    """Histogram of a numeric column, split by churn."""
    if column not in df.columns:
        return None

    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(
            df,
            x=column,
            color=TARGET_COLUMN,
            barmode="overlay",
            color_discrete_map={"Yes": "#E74C3C", "No": "#2ECC71"},
            title=title or f"Distribution of {column}",
            opacity=0.75,
            nbins=30,
        )
        fig.update_layout(font_family="Inter, sans-serif", title_font_size=16)
    else:
        return chart_churn_by_column(df, column, title)

    return _fig_to_dict(fig)


def chart_tenure_trend(df: pd.DataFrame, title: str = None) -> Dict:
    """Line chart: churn rate by tenure bucket."""
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    df2 = df.copy()
    df2["tenure_bucket"] = pd.cut(df2["tenure"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        df2.groupby("tenure_bucket", observed=True)
        .apply(lambda g: pd.Series({
            "count": len(g),
            "churn_rate_pct": round((g[TARGET_COLUMN] == "Yes").mean() * 100, 2),
        }))
        .reset_index()
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=grouped["tenure_bucket"].astype(str), y=grouped["count"], name="# Customers", marker_color="#3498DB", opacity=0.6),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=grouped["tenure_bucket"].astype(str), y=grouped["churn_rate_pct"], name="Churn Rate %", line=dict(color="#E74C3C", width=3), mode="lines+markers"),
        secondary_y=True,
    )
    fig.update_layout(
        title=title or "Churn Rate and Customer Count by Tenure",
        font_family="Inter, sans-serif",
        title_font_size=16,
        xaxis_title="Tenure (months)",
    )
    fig.update_yaxes(title_text="# Customers", secondary_y=False)
    fig.update_yaxes(title_text="Churn Rate (%)", secondary_y=True)
    return _fig_to_dict(fig)


def chart_risk_distribution(risk_df: pd.DataFrame, title: str = None) -> Dict:
    """Histogram of predicted risk scores."""
    fig = px.histogram(
        risk_df,
        x="risk_score",
        nbins=30,
        color="risk_level",
        color_discrete_map={"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"},
        title=title or "Predicted Churn Risk Score Distribution",
        labels={"risk_score": "Risk Score", "risk_level": "Risk Level"},
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Threshold")
    fig.update_layout(font_family="Inter, sans-serif", title_font_size=16)
    return _fig_to_dict(fig)


def chart_monthly_charges_by_churn(df: pd.DataFrame, title: str = None) -> Dict:
    """Box plot: monthly charges by churn status."""
    fig = px.box(
        df,
        x=TARGET_COLUMN,
        y="MonthlyCharges",
        color=TARGET_COLUMN,
        color_discrete_map={"Yes": "#E74C3C", "No": "#2ECC71"},
        title=title or "Monthly Charges by Churn Status",
        labels={TARGET_COLUMN: "Churned", "MonthlyCharges": "Monthly Charges ($)"},
    )
    fig.update_layout(font_family="Inter, sans-serif", title_font_size=16, showlegend=False)
    return _fig_to_dict(fig)


def chart_top_risk_customers(risk_df: pd.DataFrame, n: int = 15, title: str = None) -> Dict:
    """Horizontal bar: top N risk customers."""
    top = risk_df.nlargest(n, "risk_score")
    fig = px.bar(
        top,
        x="risk_score",
        y="customerID",
        orientation="h",
        color="risk_score",
        color_continuous_scale="RdYlGn_r",
        title=title or f"Top {n} Highest Churn Risk Customers",
        labels={"risk_score": "Risk Score", "customerID": "Customer ID"},
        text="risk_score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        font_family="Inter, sans-serif",
        title_font_size=16,
        yaxis={"autorange": "reversed"},
        coloraxis_showscale=False,
    )
    return _fig_to_dict(fig)


def chart_feature_importance(importance_list: List[Dict], title: str = None) -> Dict:
    """Horizontal bar chart of feature importances."""
    if not importance_list:
        return None
    df_imp = pd.DataFrame(importance_list).head(15)
    df_imp = df_imp.sort_values("importance", ascending=True)
    fig = px.bar(
        df_imp,
        x="importance",
        y="feature",
        orientation="h",
        title=title or "Top Feature Importances",
        labels={"importance": "Importance", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        font_family="Inter, sans-serif",
        title_font_size=16,
        coloraxis_showscale=False,
    )
    return _fig_to_dict(fig)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def generate_chart(chart_type: str, column: Optional[str] = None, title: Optional[str] = None, **kwargs) -> Optional[Dict]:
    """
    Central chart dispatcher.
    All charts use actual data — no invented numbers.
    """
    try:
        from app.data.loader import load_dataset
        df = load_dataset()

        if chart_type == "churn_distribution":
            return chart_churn_distribution(df, title)

        elif chart_type == "churn_by_column":
            if not column:
                return None
            return chart_churn_by_column(df, column, title)

        elif chart_type == "distribution":
            if not column:
                return None
            return chart_distribution(df, column, title)

        elif chart_type == "tenure_trend":
            return chart_tenure_trend(df, title)

        elif chart_type == "monthly_charges_by_churn":
            return chart_monthly_charges_by_churn(df, title)

        elif chart_type == "risk_distribution":
            from app.model.predictor import predict_all_customers
            risk_df = predict_all_customers()
            return chart_risk_distribution(risk_df, title)

        elif chart_type == "top_risk_customers":
            from app.model.predictor import predict_all_customers
            risk_df = predict_all_customers()
            n = kwargs.get("n", 15)
            return chart_top_risk_customers(risk_df, n, title)

        elif chart_type == "correlation_heatmap":
            numeric_df = df[NUMERICAL_FEATURES + ["Churn_Binary"]].corr()
            fig = px.imshow(
                numeric_df,
                title=title or "Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=True,
            )
            fig.update_layout(font_family="Inter, sans-serif", title_font_size=16)
            return _fig_to_dict(fig)

        else:
            logger.warning(f"Unknown chart type: {chart_type}")
            return None

    except Exception as e:
        logger.exception(f"Chart generation error ({chart_type}): {e}")
        return None
