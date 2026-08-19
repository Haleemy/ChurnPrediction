"""
verifier.py — Tool result verification before answering.

Every tool result must pass verification before the agent uses it.
If verification fails, the agent retries or gracefully degrades.
This is the anti-hallucination layer.
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Verification result ───────────────────────────────────────────────────────

class VerificationResult:
    def __init__(self, passed: bool, issues: List[str], warnings: List[str] = None):
        self.passed = passed
        self.issues = issues or []
        self.warnings = warnings or []

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return f"VerificationResult(passed={self.passed}, issues={self.issues})"


# ── Core verifiers ────────────────────────────────────────────────────────────

def verify_tool_result(tool_name: str, result: Dict, question: str = "") -> VerificationResult:
    """
    Verify a tool result before passing it to the answer generator.
    
    Checks:
    1. Tool executed successfully.
    2. Data is not None.
    3. Data is not empty.
    4. Structurally valid for this tool type.
    5. No suspicious values (NaN, Inf, obviously wrong).
    """
    issues = []
    warnings = []

    # ── Check 1: Tool success ────────────────────────────────────────────────
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        issues.append(f"Tool '{tool_name}' reported failure: {error_msg}")
        return VerificationResult(False, issues, warnings)

    # ── Check 2: Data presence ───────────────────────────────────────────────
    data = result.get("data")
    if data is None:
        issues.append(f"Tool '{tool_name}' returned success=True but data=None")
        return VerificationResult(False, issues, warnings)

    # ── Check 3: Non-empty data ───────────────────────────────────────────────
    if isinstance(data, (list, dict)) and len(data) == 0:
        warnings.append(f"Tool '{tool_name}' returned empty result")
        # Empty is OK for some tools — return pass with warning
        return VerificationResult(True, [], warnings)

    # ── Check 4: Type-specific structural validation ──────────────────────────
    tool_issues = _validate_by_tool(tool_name, data)
    issues.extend(tool_issues)

    # ── Check 5: Numeric sanity ───────────────────────────────────────────────
    numeric_issues = _check_numeric_sanity(data)
    warnings.extend(numeric_issues)

    passed = len(issues) == 0
    return VerificationResult(passed, issues, warnings)


def _validate_by_tool(tool_name: str, data: Any) -> List[str]:
    """Tool-specific structural validation."""
    issues = []

    if tool_name in ("predict_customer_risk", "predict_new_customer"):
        if isinstance(data, dict):
            for field in ["risk_score", "risk_level", "prediction"]:
                if field not in data:
                    issues.append(f"Prediction result missing field: '{field}'")
            if "risk_score" in data:
                score = data["risk_score"]
                if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                    issues.append(f"risk_score {score!r} is not in [0, 1]")

    elif tool_name == "predict_hypothetical":
        if isinstance(data, dict):
            for field in ["current", "hypothetical", "risk_delta"]:
                if field not in data:
                    issues.append(f"Hypothetical result missing field: '{field}'")

    elif tool_name == "analyze_data":
        if isinstance(data, dict):
            # Check for obvious NaN in numeric results
            for k, v in data.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    issues.append(f"Numeric result for '{k}' is NaN or Inf")

    elif tool_name == "get_top_risk_customers":
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if "risk_score" not in first:
                issues.append("Top risk customers result missing 'risk_score' field")

    elif tool_name == "get_model_info":
        if isinstance(data, dict):
            if "model_name" not in data and "test_metrics" not in data:
                issues.append("Model info appears incomplete")

    return issues


def _check_numeric_sanity(data: Any) -> List[str]:
    """Check for NaN/Inf in nested data — returns warnings, not errors."""
    warnings = []

    def _recurse(obj, path=""):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                warnings.append(f"NaN/Inf found at {path!r}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _recurse(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:20]):  # Only check first 20 elements
                _recurse(v, f"{path}[{i}]")

    _recurse(data)
    return warnings


# ── Plan verification ─────────────────────────────────────────────────────────

def verify_plan(plan: Dict) -> VerificationResult:
    """Verify the agent's execution plan before running it."""
    issues = []
    warnings = []

    if not isinstance(plan, dict):
        return VerificationResult(False, ["Plan is not a dict"])

    if "intent" not in plan:
        issues.append("Plan missing 'intent' field")

    if "steps" not in plan:
        issues.append("Plan missing 'steps' field")
    else:
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            issues.append("Plan 'steps' is not a list")
        elif len(steps) == 0 and plan.get("intent") != "unanswerable":
            warnings.append("Plan has 0 steps but intent is not 'unanswerable'")
        else:
            for i, step in enumerate(steps):
                if "tool" not in step:
                    issues.append(f"Step {i} missing 'tool' field")
                if "params" not in step:
                    warnings.append(f"Step {i} missing 'params' field")

    passed = len(issues) == 0
    return VerificationResult(passed, issues, warnings)


# ── Answer verification ───────────────────────────────────────────────────────

def verify_answer_grounding(answer: str, tool_results: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Lightweight check: scan the answer for numbers and verify they
    appear in the tool results. Handles formatted numbers (e.g. '7,043').
    """
    import re
    warnings = []

    # Clean commas in numbers like "7,043" -> "7043"
    cleaned_answer = re.sub(r'(\d+),(\d+)', r'\1\2', answer)
    cleaned_results = re.sub(r'(\d+),(\d+)', r'\1\2', str(tool_results))

    raw_ans_nums = re.findall(r'\b\d+\.?\d*\b', cleaned_answer)
    raw_res_nums = re.findall(r'\b\d+\.?\d*\b', cleaned_results)

    def parse_num(val):
        try:
            return round(float(val), 2)
        except ValueError:
            return None

    ans_nums = {parse_num(n) for n in raw_ans_nums if parse_num(n) is not None}
    res_nums = {parse_num(n) for n in raw_res_nums if parse_num(n) is not None}

    # Only check numbers > 10 (small numbers like 0, 1, 2 are common text indices)
    large_ans = {n for n in ans_nums if n > 10}
    ungrounded = large_ans - res_nums

    if ungrounded:
        # Display nicely as int if whole float
        display_nums = {int(n) if n.is_integer() else n for n in ungrounded}
        warnings.append(
            f"Answer contains numbers not found in tool results: {display_nums}. "
            "Possible hallucination risk — review answer carefully."
        )

    return len(ungrounded) == 0, warnings
