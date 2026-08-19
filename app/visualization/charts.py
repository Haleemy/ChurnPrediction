"""
charts.py — Modern, vibrant Plotly chart generation for Churn Analyst UI.

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

# Ember dark theme palette & default layout properties
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(35, 22, 20, 0.45)",
    font=dict(family="Inter, sans-serif", color="#f0e2d8", size=12),
    title_font=dict(family="Space Grotesk, sans-serif", color="#f5ede4", size=16),
    margin=dict(l=40, r=40, t=50, b=40),
    colorway=["#FF6B35", "#FFB84D", "#B24BF3", "#2DD4BF", "#F72585"],
    xaxis=dict(
        gridcolor="#2a1d1a",
        zerolinecolor="#4a332c",
        showgrid=True,
        tickfont=dict(color="#c9a898"),
    ),
    yaxis=dict(
        gridcolor="#2a1d1a",
        zerolinecolor="#4a332c",
        showgrid=True,
        tickfont=dict(color="#c9a898"),
    ),
)

# Light theme layout properties
LIGHT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248, 250, 252, 0.8)",
    font=dict(family="Inter, sans-serif", color="#1e293b", size=12),
    title_font=dict(family="Space Grotesk, sans-serif", color="#0f172a", size=16),
    margin=dict(l=40, r=40, t=50, b=40),
    colorway=["#e11d48", "#d97706", "#7c3aed", "#059669", "#2563eb"],
    xaxis=dict(
        gridcolor="#e2e8f0",
        zerolinecolor="#cbd5e1",
        showgrid=True,
        tickfont=dict(color="#64748b"),
    ),
    yaxis=dict(
        gridcolor="#e2e8f0",
        zerolinecolor="#cbd5e1",
        showgrid=True,
        tickfont=dict(color="#64748b"),
    ),
)


def _apply_theme(fig: go.Figure, title: str = None, theme: str = "Dark") -> Dict:
    """Apply unified design system to a Plotly figure (Dark or Light)."""
    base_layout = LIGHT_LAYOUT if theme == "Light" else DARK_LAYOUT
    layout_update = base_layout.copy()
    if title:
        title_color = "#0f172a" if theme == "Light" else "#f5ede4"
        layout_update["title"] = dict(text=title, font=dict(size=16, color=title_color))
    fig.update_layout(**layout_update)
    return fig.to_dict()


def chart_risk_gauge(score: float, customer_id: str = "", theme: str = "Dark") -> Dict:
    """Sleek gauge chart for individual customer churn risk score."""
    pct = score * 100

    if score >= 0.65:
        gauge_color = "#dc2626" if theme == "Light" else "#FF6B6B"
        risk_label = "HIGH RISK"
    elif score >= 0.35:
        gauge_color = "#d97706" if theme == "Light" else "#FFB84D"
        risk_label = "MEDIUM RISK"
    else:
        gauge_color = "#059669" if theme == "Light" else "#2DD4BF"
        risk_label = "LOW RISK"

    bg_color = "rgba(241, 245, 249, 0.8)" if theme == "Light" else "rgba(35, 22, 20, 0.6)"
    border_color = "#cbd5e1" if theme == "Light" else "#4a332c"
    tick_color = "#94a3b8" if theme == "Light" else "#8a6a5c"
    title_sub_color = "#64748b" if theme == "Light" else "#c9a898"
    font_color = "#0f172a" if theme == "Light" else "#f5ede4"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "color": gauge_color, "family": "Space Grotesk, sans-serif"}},
            title={"text": f"<b>{risk_label}</b><br><span style='font-size:0.8em;color:{title_sub_color};'>{customer_id}</span>", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": tick_color},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": bg_color,
                "borderwidth": 1,
                "bordercolor": border_color,
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)" if theme == "Light" else "rgba(45, 212, 191, 0.15)"},
                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.15)" if theme == "Light" else "rgba(255, 184, 77, 0.15)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.15)" if theme == "Light" else "rgba(255, 107, 107, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": font_color, "width": 3},
                    "thickness": 0.8,
                    "value": 50.0,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=font_color),
        margin=dict(l=30, r=30, t=40, b=30),
        height=240,
    )
    return fig.to_dict()


def chart_churn_distribution(df: pd.DataFrame, title: str = None, theme: str = "Dark") -> Dict:
    """Pie/Donut chart of overall customer churn distribution."""
    vc = df[TARGET_COLUMN].value_counts()
    yes_color = "#dc2626" if theme == "Light" else "#FF6B6B"
    no_color = "#059669" if theme == "Light" else "#2DD4BF"
    border_color = "#ffffff" if theme == "Light" else "#120d0d"
    fig = px.pie(
        names=vc.index,
        values=vc.values,
        title=title or "Customer Churn Ratio",
        color=vc.index,
        color_discrete_map={"Yes": yes_color, "No": no_color},
        hole=0.55,
    )
    fig.update_traces(
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color=border_color, width=2)),
    )
    return _apply_theme(fig, title or "Customer Churn Ratio", theme=theme)


def chart_churn_by_column(df: pd.DataFrame, column: str, title: str = None, theme: str = "Dark") -> Dict:
    """Bar chart: churn rate by categorical column."""
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

    scale = ["#10b981", "#f59e0b", "#ef4444"] if theme == "Light" else ["#2DD4BF", "#FFB84D", "#FF6B6B"]
    line_color = "#cbd5e1" if theme == "Light" else "#2a1d1a"
    fig = px.bar(
        grouped,
        x=column,
        y="churn_rate_pct",
        color="churn_rate_pct",
        color_continuous_scale=scale,
        text="churn_rate_pct",
        title=title or f"Churn Rate by {column}",
        labels={"churn_rate_pct": "Churn Rate (%)", column: column},
        hover_data={"count": True},
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_color=line_color,
        marker_line_width=1,
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-25)
    return _apply_theme(fig, title or f"Churn Rate by {column}", theme=theme)


def chart_distribution(df: pd.DataFrame, column: str, title: str = None, theme: str = "Dark") -> Dict:
    """Histogram of a numeric or categorical column split by churn."""
    if column not in df.columns:
        return None

    if pd.api.types.is_numeric_dtype(df[column]):
        yes_color = "#dc2626" if theme == "Light" else "#FF6B6B"
        no_color = "#059669" if theme == "Light" else "#2DD4BF"
        fig = px.histogram(
            df,
            x=column,
            color=TARGET_COLUMN,
            barmode="overlay",
            color_discrete_map={"Yes": yes_color, "No": no_color},
            title=title or f"Distribution of {column}",
            opacity=0.7,
            nbins=30,
        )
        return _apply_theme(fig, title or f"Distribution of {column}", theme=theme)
    else:
        return chart_churn_by_column(df, column, title, theme=theme)


def chart_tenure_trend(df: pd.DataFrame, title: str = None, theme: str = "Dark") -> Dict:
    """Combo line/bar chart: customer volume & churn rate by tenure range."""
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"]
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

    bar_color = "rgba(124, 58, 237, 0.45)" if theme == "Light" else "rgba(178, 75, 243, 0.55)"
    bar_line = "#7c3aed" if theme == "Light" else "#B24BF3"
    line_color = "#dc2626" if theme == "Light" else "#FF6B6B"
    grid_color = "#e2e8f0" if theme == "Light" else "#2a1d1a"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=grouped["tenure_bucket"].astype(str),
            y=grouped["count"],
            name="Active Customers",
            marker_color=bar_color,
            marker_line_color=bar_line,
            marker_line_width=1,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["tenure_bucket"].astype(str),
            y=grouped["churn_rate_pct"],
            name="Churn Rate (%)",
            line=dict(color=line_color, width=3, shape="spline"),
            mode="lines+markers",
            marker=dict(size=8, color=line_color),
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Customer Count", secondary_y=False, gridcolor=grid_color)
    fig.update_yaxes(title_text="Churn Rate (%)", secondary_y=True, gridcolor=grid_color)
    fig.update_xaxes(title_text="Tenure (Months)")
    return _apply_theme(fig, title or "Churn Rate & Customer Count by Tenure", theme=theme)


def chart_risk_distribution(risk_df: pd.DataFrame, title: str = None, theme: str = "Dark") -> Dict:
    """Distribution of predicted churn risk scores."""
    cmap = {"High": "#dc2626", "Medium": "#d97706", "Low": "#059669"} if theme == "Light" else {"High": "#FF6B6B", "Medium": "#FFB84D", "Low": "#2DD4BF"}
    vline_color = "#64748b" if theme == "Light" else "#c9a898"
    fig = px.histogram(
        risk_df,
        x="risk_score",
        nbins=30,
        color="risk_level",
        color_discrete_map=cmap,
        title=title or "Predicted Churn Risk Distribution",
        labels={"risk_score": "Churn Risk Score", "risk_level": "Risk Tier"},
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color=vline_color, annotation_text="Decision Threshold (0.50)")
    return _apply_theme(fig, title or "Predicted Churn Risk Distribution", theme=theme)


def chart_monthly_charges_by_churn(df: pd.DataFrame, title: str = None, theme: str = "Dark") -> Dict:
    """Box plot of monthly charges split by churn status."""
    yes_color = "#dc2626" if theme == "Light" else "#FF6B6B"
    no_color = "#059669" if theme == "Light" else "#2DD4BF"
    fig = px.box(
        df,
        x=TARGET_COLUMN,
        y="MonthlyCharges",
        color=TARGET_COLUMN,
        color_discrete_map={"Yes": yes_color, "No": no_color},
        title=title or "Monthly Charges by Churn Status",
        labels={TARGET_COLUMN: "Churned", "MonthlyCharges": "Monthly Charges ($)"},
    )
    fig.update_layout(showlegend=False)
    return _apply_theme(fig, title or "Monthly Charges by Churn Status", theme=theme)


def chart_top_risk_customers(risk_df: pd.DataFrame, n: int = 15, title: str = None, theme: str = "Dark") -> Dict:
    """Horizontal bar chart of top N highest risk customers."""
    top = risk_df.nlargest(n, "risk_score")
    scale = ["#f59e0b", "#ef4444"] if theme == "Light" else ["#FFB84D", "#FF6B6B"]
    fig = px.bar(
        top,
        x="risk_score",
        y="customerID",
        orientation="h",
        color="risk_score",
        color_continuous_scale=scale,
        title=title or f"Top {n} Highest Risk Customers",
        labels={"risk_score": "Risk Score", "customerID": "Customer ID"},
        text="risk_score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False)
    return _apply_theme(fig, title or f"Top {n} Highest Risk Customers", theme=theme)


def chart_feature_importance(importance_list: List[Dict], title: str = None, theme: str = "Dark") -> Dict:
    """Horizontal bar chart of top feature importances."""
    if not importance_list:
        return None
    df_imp = pd.DataFrame(importance_list).head(15)
    df_imp = df_imp.sort_values("importance", ascending=True)
    scale = ["#f59e0b", "#8b5cf6"] if theme == "Light" else ["#FFB84D", "#B24BF3"]
    fig = px.bar(
        df_imp,
        x="importance",
        y="feature",
        orientation="h",
        title=title or "Top Feature Importances (ML Model)",
        labels={"importance": "Importance", "feature": "Feature"},
        color="importance",
        color_continuous_scale=scale,
    )
    fig.update_layout(coloraxis_showscale=False)
    return _apply_theme(fig, title or "Top Feature Importances (ML Model)", theme=theme)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def generate_chart(chart_type: str, column: Optional[str] = None, title: Optional[str] = None, theme: str = "Dark", **kwargs) -> Optional[Dict]:
    """Central chart dispatcher."""
    try:
        from app.data.loader import load_dataset
        df = load_dataset()

        if chart_type == "churn_distribution":
            return chart_churn_distribution(df, title, theme=theme)

        elif chart_type == "churn_by_column":
            if not column:
                return None
            return chart_churn_by_column(df, column, title, theme=theme)

        elif chart_type == "distribution":
            if not column:
                return None
            return chart_distribution(df, column, title, theme=theme)

        elif chart_type == "tenure_trend":
            return chart_tenure_trend(df, title, theme=theme)

        elif chart_type == "monthly_charges_by_churn":
            return chart_monthly_charges_by_churn(df, title, theme=theme)

        elif chart_type == "risk_distribution":
            from app.model.predictor import predict_all_customers
            risk_df = predict_all_customers()
            return chart_risk_distribution(risk_df, title, theme=theme)

        elif chart_type == "top_risk_customers":
            from app.model.predictor import predict_all_customers
            risk_df = predict_all_customers()
            n = kwargs.get("n", 15)
            return chart_top_risk_customers(risk_df, n, title, theme=theme)

        elif chart_type == "correlation_heatmap":
            numeric_df = df[NUMERICAL_FEATURES + ["Churn_Binary"]].corr()
            diverging = [
                [0.0, "#7c3aed" if theme == "Light" else "#B24BF3"],
                [0.5, "#f1f5f9" if theme == "Light" else "#1a1210"],
                [1.0, "#dc2626" if theme == "Light" else "#FF6B6B"],
            ]
            fig = px.imshow(
                numeric_df,
                title=title or "Feature Correlation Heatmap",
                color_continuous_scale=diverging,
                zmin=-1, zmax=1,
                text_auto=".2f",
            )
            return _apply_theme(fig, title or "Feature Correlation Heatmap", theme=theme)

        else:
            logger.warning(f"Unknown chart type: {chart_type}")
            return None

    except Exception as e:
        logger.exception(f"Chart generation error ({chart_type}): {e}")
        return None