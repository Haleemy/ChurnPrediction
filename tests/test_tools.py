import pytest
from app.agent.tools import execute_tool, TOOL_REGISTRY


def test_tool_registry_exists():
    assert isinstance(TOOL_REGISTRY, dict)
    assert "analyze_data" in TOOL_REGISTRY
    assert "predict_customer_risk" in TOOL_REGISTRY
    assert "generate_chart" in TOOL_REGISTRY


def test_execute_tool_analyze_data():
    res = execute_tool("analyze_data", operation="get_churn_rate")
    assert res["success"] is True
    assert "data" in res


def test_execute_tool_predict_customer_risk():
    res = execute_tool("predict_customer_risk", customer_id="7590-VHVEG")
    assert res["success"] is True
    assert res["data"]["customer_id"] == "7590-VHVEG"


def test_execute_tool_get_model_info():
    res = execute_tool("get_model_info")
    assert res["success"] is True
    assert "model_name" in res["data"]


def test_execute_tool_unknown():
    res = execute_tool("non_existent_tool")
    assert res["success"] is False
