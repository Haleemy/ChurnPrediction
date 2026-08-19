"""
planner.py — Intent classification and plan generation.

The planner converts a user question + context into a structured execution plan
using a single LLM call. This minimizes LLM calls for the free-tier limit.
"""
import json
import logging
import re
from typing import Dict, List, Optional

from app.agent.prompts import PLANNING_PROMPT
from app.agent.verifier import verify_plan

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[Dict]:
    """Robustly extract JSON from LLM response (handles markdown code blocks)."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract from ```json ... ``` block
    pattern = r"```(?:json)?\s*([\s\S]+?)\s*```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    brace_match = re.search(r'\{[\s\S]+\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not extract JSON from planner response: {text[:200]}")
    return None


def _extract_hypothetical_changes(question: str) -> Dict:
    q = question.lower()
    changes = {}

    # Contract changes
    if any(phrase in q for phrase in ["two year", "two-year", "twoyear", "2 year", "2-year"]):
        changes["Contract"] = "Two year"
    elif any(phrase in q for phrase in ["one year", "one-year", "oneyear", "1 year", "1-year"]):
        changes["Contract"] = "One year"
    elif any(phrase in q for phrase in ["month-to-month", "month to month", "monthtomonth"]):
        changes["Contract"] = "Month-to-month"

    # Payment method
    if "bank transfer" in q:
        changes["PaymentMethod"] = "Bank transfer (automatic)"
    elif "credit card" in q:
        changes["PaymentMethod"] = "Credit card (automatic)"
    elif "electronic check" in q:
        changes["PaymentMethod"] = "Electronic check"
    elif "mailed check" in q:
        changes["PaymentMethod"] = "Mailed check"

    # Internet service
    if "fiber" in q:
        changes["InternetService"] = "Fiber optic"
    elif "dsl" in q:
        changes["InternetService"] = "DSL"

    # Add-on services
    action_words = ["add", "added", "yes", "get", "got", "switch", "with", "enable", "buy", "include", "included"]
    if ("tech support" in q or "techsupport" in q) and any(w in q for w in action_words):
        changes["TechSupport"] = "Yes"
    if ("online security" in q or "onlinesecurity" in q) and any(w in q for w in action_words):
        changes["OnlineSecurity"] = "Yes"
    if ("online backup" in q or "onlinebackup" in q) and any(w in q for w in action_words):
        changes["OnlineBackup"] = "Yes"
    if ("device protection" in q or "deviceprotection" in q) and any(w in q for w in action_words):
        changes["DeviceProtection"] = "Yes"

    return changes


def create_fallback_plan(question: str) -> Dict:
    """
    Create a rule-based fallback plan when LLM planning fails.
    Uses keyword matching to determine intent.
    """
    q = question.lower()

    # Customer ID pattern
    customer_id_match = re.search(r'\b([A-Z0-9]{4}-[A-Z0-9]{5})\b', question.upper())

    if customer_id_match:
        cid = customer_id_match.group(1)
        if any(word in q for word in ["hypothetical", "what if", "changes", "switch", "change"]):
            changes = _extract_hypothetical_changes(question)
            return {
                "intent": "hypothetical",
                "reasoning": f"Customer hypothetical analysis with changes {changes}",
                "steps": [{"tool": "predict_hypothetical", "params": {"customer_id": cid, "changes": changes}, "purpose": "Compare current vs hypothetical"}],
                "requires_unavailable_data": False,
            }
        return {
            "intent": "prediction",
            "reasoning": "Single customer risk prediction",
            "steps": [{"tool": "predict_customer_risk", "params": {"customer_id": cid}, "purpose": "Get customer risk"}],
            "requires_unavailable_data": False,
        }

    if any(word in q for word in ["how many", "count", "total customers", "number of"]):
        return {
            "intent": "dataset_info",
            "reasoning": "Dataset statistics requested",
            "steps": [{"tool": "get_dataset_info", "params": {}, "purpose": "Get dataset info"}],
            "requires_unavailable_data": False,
        }

    # Column / Segment breakdowns MUST be checked before general churn rate queries
    col_map = {
        "contract": "Contract",
        "payment": "PaymentMethod",
        "internet": "InternetService",
        "gender": "gender",
        "senior": "SeniorCitizen",
        "partner": "Partner",
        "dependents": "Dependents",
        "tech support": "TechSupport",
        "online security": "OnlineSecurity",
        "paperless": "PaperlessBilling",
    }
    for key, col in col_map.items():
        if key in q:
            return {
                "intent": "eda",
                "reasoning": f"Churn by {col}",
                "steps": [
                    {"tool": "analyze_data", "params": {"operation": "group_by_churn", "column": col}, "purpose": f"Churn rate by {col}"},
                    {"tool": "generate_chart", "params": {"chart_type": "churn_by_column", "column": col}, "purpose": f"Visualize churn rate by {col}"},
                ],
                "requires_unavailable_data": False,
            }

    if any(word in q for word in ["churn rate", "churn %", "percentage churn", "churn percentage"]):
        return {
            "intent": "eda",
            "reasoning": "Churn rate query",
            "steps": [
                {"tool": "analyze_data", "params": {"operation": "get_churn_rate"}, "purpose": "Compute churn rate"},
                {"tool": "generate_chart", "params": {"chart_type": "churn_distribution"}, "purpose": "Visualize overall churn distribution"},
            ],
            "requires_unavailable_data": False,
        }

    if any(word in q for word in ["high risk", "highest risk", "most likely to churn", "top risk"]):
        return {
            "intent": "aggregate",
            "reasoning": "Top risk customers query",
            "steps": [
                {"tool": "get_top_risk_customers", "params": {"n": 10}, "purpose": "Get highest risk customers"},
                {"tool": "generate_chart", "params": {"chart_type": "top_risk_customers", "n": 10}, "purpose": "Visualize top risk customers"},
            ],
            "requires_unavailable_data": False,
        }

    if any(word in q for word in ["average", "mean", "monthly charges"]):
        if any(word in q for word in ["churn", "churned", "vs", "difference"]):
            return {
                "intent": "eda",
                "reasoning": "Average monthly charges by churn status",
                "steps": [
                    {"tool": "analyze_data", "params": {"operation": "average_by_churn", "column": "MonthlyCharges"}, "purpose": "Average monthly charges for churned vs non-churned"},
                    {"tool": "generate_chart", "params": {"chart_type": "monthly_charges_by_churn"}, "purpose": "Visualize monthly charges by churn"},
                ],
                "requires_unavailable_data": False,
            }
        return {
            "intent": "eda",
            "reasoning": "Average monthly charges",
            "steps": [
                {"tool": "analyze_data", "params": {"operation": "get_average", "column": "MonthlyCharges"}, "purpose": "Average monthly charges"},
                {"tool": "generate_chart", "params": {"chart_type": "monthly_charges_by_churn"}, "purpose": "Visualize monthly charges by churn"},
            ],
            "requires_unavailable_data": False,
        }

    # Default: dataset info
    return {
        "intent": "dataset_info",
        "reasoning": "General dataset question — returning overview",
        "steps": [{"tool": "get_dataset_info", "params": {}, "purpose": "Dataset overview"}],
        "requires_unavailable_data": False,
    }


def generate_plan(question: str, context: str, llm_provider) -> Dict:
    """
    Generate an execution plan for the user's question.
    
    Strategy:
    1. Call LLM with planning prompt.
    2. Parse JSON response.
    3. Verify plan structure.
    4. If parsing fails, use rule-based fallback.
    
    Returns a plan dict.
    """
    prompt = PLANNING_PROMPT.format(question=question, context=context)

    try:
        response = llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        raw = response.get("content", "")
        plan = _extract_json(raw)

        if plan is None:
            logger.warning("Planner failed to return valid JSON — using fallback")
            return create_fallback_plan(question)

        # Verify the plan structure
        vr = verify_plan(plan)
        if not vr:
            logger.warning(f"Plan failed verification: {vr.issues} — using fallback")
            return create_fallback_plan(question)

        return plan

    except Exception as e:
        logger.warning(f"Planning LLM call failed: {e} — using fallback plan")
        return create_fallback_plan(question)
