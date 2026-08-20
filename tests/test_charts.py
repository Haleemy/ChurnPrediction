import pytest
from app.visualization.charts import generate_chart


def test_generate_chart_churn_distribution():
    res = generate_chart("churn_distribution")
    assert res is not None
    assert "data" in res
    assert "layout" in res


def test_generate_chart_correlation_heatmap():
    res = generate_chart("correlation_heatmap")
    assert res is not None
    assert "data" in res


def test_generate_chart_top_risk_customers():
    res = generate_chart("top_risk_customers", n=5)
    assert res is not None
    assert "data" in res


def test_generate_chart_unknown():
    res = generate_chart("unknown_chart_type")
    assert res is None
