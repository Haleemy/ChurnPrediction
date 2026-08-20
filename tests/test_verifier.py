import pytest
from app.agent.verifier import verify_tool_result, verify_plan, verify_answer_grounding


def test_verify_tool_result_success():
    tool_res = {"success": True, "data": {"n_rows": 7043, "churn_rate_pct": 26.54}}
    vr = verify_tool_result("analyze_data", tool_res)
    assert bool(vr) is True


def test_verify_tool_result_failure():
    tool_res = {"success": False, "error": "Something went wrong"}
    vr = verify_tool_result("analyze_data", tool_res)
    assert bool(vr) is False


def test_verify_plan_valid():
    plan = {
        "intent": "aggregate",
        "steps": [{"tool": "analyze_data", "params": {"operation": "get_churn_rate"}}]
    }
    vr = verify_plan(plan)
    assert bool(vr) is True


def test_verify_answer_grounding():
    answer = "The overall churn rate is 26.54% across 7043 customers."
    tool_results = [{"tool": "analyze_data", "result": {"success": True, "data": {"n_rows": 7043, "churn_rate_pct": 26.54}}}]
    passed, warnings = verify_answer_grounding(answer, tool_results)
    assert passed is True
