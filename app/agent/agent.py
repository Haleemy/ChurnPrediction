"""
agent.py — Main agentic orchestration loop.

Architecture:
  User question
    ↓
  1. Intent / Planning (LLM call #1)
    ↓
  2. Tool execution (deterministic Python)
    ↓
  3. Verification (verifier.py)
    ↓ [retry if failed, up to LLM_MAX_RETRIES]
  4. Answer generation (LLM call #2)
    ↓
  Answer with grounded evidence

Key principles:
- LLM calls are minimized (planning + answering = 2 calls max per question)
- All numbers come from tools, not the LLM
- Retry with modified plan if tool fails
- Never invent answers if tools return empty/failed
"""
import json
import logging
from typing import Any, Dict, Generator, List, Optional, Tuple

from app.agent.planner import generate_plan, create_fallback_plan
from app.agent.tools import execute_tool, TOOL_REGISTRY
from app.agent.verifier import verify_tool_result, verify_answer_grounding
from app.agent.prompts import SYSTEM_PROMPT, ANSWER_PROMPT
from app.config import LLM_MAX_RETRIES

logger = logging.getLogger(__name__)


# ── Conversation memory ────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Lightweight multi-turn conversation context.
    Stores structured turn history with tool results for reference.
    No vector DB — simple list, last N turns.
    """
    def __init__(self, max_turns: int = 10):
        self.turns: List[Dict] = []
        self.max_turns = max_turns

    def add_turn(self, question: str, answer: str, tool_results: List[Dict] = None):
        self.turns.append({
            "question": question,
            "answer": answer,
            "tool_results": tool_results or [],
        })
        # Keep only the last N turns
        self.turns = self.turns[-self.max_turns:]

    def get_context_string(self) -> str:
        """Compact context for planner/answer prompts."""
        if not self.turns:
            return "No prior conversation."
        parts = []
        for i, turn in enumerate(self.turns[-3:]):  # Last 3 turns
            parts.append(f"Q: {turn['question']}\nA: {turn['answer'][:200]}...")
        return "\n\n".join(parts)

    def clear(self):
        self.turns = []


# ── Agent ─────────────────────────────────────────────────────────────────────

class ChurnAnalystAgent:
    """
    The main agentic orchestrator.
    
    Each call to `answer(question)` goes through:
    1. Plan (1 LLM call)
    2. Execute tools (deterministic)
    3. Verify results
    4. Retry if needed (up to LLM_MAX_RETRIES)
    5. Generate answer (1 LLM call)
    """

    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.memory = ConversationMemory()

    def answer(
        self,
        question: str,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer a user question using the full agent pipeline.
        
        Returns:
            {
              "answer": str,
              "tool_results": [...],
              "plan": {...},
              "charts": [...],
              "warnings": [...],
            }
        """
        logger.info(f"Agent question: {question[:100]}")
        context = self.memory.get_context_string()

        # ── Step 1: Plan ──────────────────────────────────────────────────────
        plan = generate_plan(question, context, self.llm)
        logger.info(f"Plan intent: {plan.get('intent')} | steps: {len(plan.get('steps', []))}")

        # ── Handle unanswerable questions ─────────────────────────────────────
        if plan.get("intent") == "unanswerable" or plan.get("requires_unavailable_data"):
            reason = plan.get("unavailable_reason", "The required data is not available in the dataset.")
            answer = f"I cannot answer this question: {reason}"
            self.memory.add_turn(question, answer)
            return {
                "answer": answer,
                "tool_results": [],
                "plan": plan,
                "charts": [],
                "warnings": [],
            }

        # ── Step 2: Execute tools ─────────────────────────────────────────────
        steps = plan.get("steps", [])
        tool_results = []
        charts = []
        all_warnings = []

        attempt = 0
        while attempt <= LLM_MAX_RETRIES:
            tool_results = []
            failed_steps = []

            for step in steps:
                tool_name = step.get("tool", "")
                params = step.get("params", {})
                purpose = step.get("purpose", "")

                logger.info(f"  Executing tool: {tool_name} | {purpose}")
                result = execute_tool(tool_name, **params)

                # Collect charts separately
                if tool_name == "generate_chart" and result.get("success"):
                    charts.append(result["data"])

                # ── Step 3: Verify ────────────────────────────────────────────
                vr = verify_tool_result(tool_name, result, question)
                all_warnings.extend(vr.warnings)

                if not vr:
                    logger.warning(f"  Tool '{tool_name}' failed verification: {vr.issues}")
                    failed_steps.append({"tool": tool_name, "issues": vr.issues, "result": result})
                else:
                    tool_results.append({
                        "tool": tool_name,
                        "purpose": purpose,
                        "result": result,
                        "warnings": vr.warnings,
                    })

            # If no failures, break out of retry loop
            if not failed_steps:
                break

            # ── Retry with fallback plan ──────────────────────────────────────
            attempt += 1
            if attempt <= LLM_MAX_RETRIES:
                logger.info(f"  Retry {attempt}/{LLM_MAX_RETRIES} with fallback plan")
                # Use simpler fallback plan on retry
                plan = create_fallback_plan(question)
                steps = plan.get("steps", [])
            else:
                # Max retries reached — report what failed
                error_summary = "; ".join(
                    f"{fs['tool']}: {fs['issues']}" for fs in failed_steps
                )
                tool_results.append({
                    "tool": "system",
                    "purpose": "error_report",
                    "result": {
                        "success": False,
                        "error": f"Some tools failed after {LLM_MAX_RETRIES} retries: {error_summary}",
                        "data": None,
                    },
                    "warnings": [],
                })

        # ── Step 4: Auto-generate relevant chart if no chart was planned ──────
        if not charts and tool_results:
            auto_chart = self._auto_generate_chart(question, tool_results)
            if auto_chart:
                charts.append(auto_chart)

        # ── Step 5: Generate answer ───────────────────────────────────────────
        answer = self._generate_answer(question, context, tool_results)

        # ── Lightweight grounding check ───────────────────────────────────────
        is_grounded, grounding_warnings = verify_answer_grounding(answer, tool_results)
        all_warnings.extend(grounding_warnings)

        # ── Update memory ─────────────────────────────────────────────────────
        self.memory.add_turn(question, answer, tool_results)

        return {
            "answer": answer,
            "tool_results": tool_results,
            "plan": plan,
            "charts": charts,
            "warnings": all_warnings,
        }

    def _auto_generate_chart(self, question: str, tool_results: List[Dict]) -> Optional[Dict]:
        """Auto-generate an interactive Plotly chart if appropriate for the query."""
        from app.agent.tools import execute_tool
        q_lower = question.lower()

        # Check tool results for top risk customers or dataframe operations
        for tr in tool_results:
            tool = tr.get("tool")
            res_data = tr.get("result", {}).get("data")
            if tool == "get_top_risk_customers":
                res = execute_tool("generate_chart", chart_type="top_risk_customers", n=10)
                if res.get("success"):
                    return res["data"]
            elif tool == "analyze_data":
                if "contract" in q_lower:
                    res = execute_tool("generate_chart", chart_type="churn_by_column", column="Contract")
                    if res.get("success"): return res["data"]
                elif "payment" in q_lower:
                    res = execute_tool("generate_chart", chart_type="churn_by_column", column="PaymentMethod")
                    if res.get("success"): return res["data"]
                elif "internet" in q_lower:
                    res = execute_tool("generate_chart", chart_type="churn_by_column", column="InternetService")
                    if res.get("success"): return res["data"]
                elif "tenure" in q_lower:
                    res = execute_tool("generate_chart", chart_type="tenure_trend")
                    if res.get("success"): return res["data"]
                elif "monthly" in q_lower or "charges" in q_lower:
                    res = execute_tool("generate_chart", chart_type="monthly_charges_by_churn")
                    if res.get("success"): return res["data"]

        if any(w in q_lower for w in ["churn rate", "distribution", "overall churn"]):
            res = execute_tool("generate_chart", chart_type="churn_distribution")
            if res.get("success"):
                return res["data"]

        return None

    def _generate_answer(
        self,
        question: str,
        context: str,
        tool_results: List[Dict],
    ) -> str:
        """
        Generate the final answer using LLM with tool results as evidence.
        The LLM cannot add numbers that aren't in tool_results.
        """
        # Format tool results compactly
        formatted_results = []
        for tr in tool_results:
            tool = tr["tool"]
            result = tr["result"]
            purpose = tr.get("purpose", "")
            if result.get("success"):
                data_str = json.dumps(result["data"], default=str)[:12000]
                formatted_results.append(f"[{tool} — {purpose}]\n{data_str}")
            else:
                error = result.get("error", "Unknown error")
                formatted_results.append(f"[{tool} — FAILED]\nError: {error}")

        tool_results_str = "\n\n".join(formatted_results) if formatted_results else "No tool results."

        prompt = ANSWER_PROMPT.format(
            question=question,
            context=context,
            tool_results=tool_results_str,
        )

        try:
            response = self.llm.complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw_content = response.get("content", "I encountered an error generating the answer.")
            return self._clean_llm_response(raw_content)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Fallback: format tool results directly
            return self._format_results_directly(tool_results)

    def _clean_llm_response(self, text: str) -> str:
        """Strip thinking tags, inner monologue, rule lists, or preamble meta-reasoning from answer."""
        import re
        if not text:
            return ""

        # 1. Remove <think>...</think> tags
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()

        # 2. Strip robotic meta-disclaimers
        text = re.sub(r'\(as returned by the model\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\(as returned by model\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Source:\s*`?[a-zA-Z_]+`?\s*result\.?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'These figures come directly from standard tool computations\.?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'These figures come directly from the `?[a-zA-Z_]+`? result\.?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'The tool returned only these [a-zA-Z0-9\s]+ entries[\s\S]*?\.', '', text, flags=re.IGNORECASE)

        # 3. Check if text contains a "Final check of the text: \"...\"" pattern
        final_check_match = re.search(r'Final check of the text:\s*["“]([\s\S]+?)["”]', text, flags=re.IGNORECASE)
        if final_check_match:
            text = final_check_match.group(1).strip()

        # 4. Cut off at "Check against rules:" or "Self-Correction:" if preceded by text
        cutoff_patterns = [
            r"\n\s*Check against rules\s*:",
            r"\n\s*Constraints check\s*:",
            r"\n\s*Self-correction\s*:",
            r"\n\s*Matches all constraints\s*\.",
        ]
        for pat in cutoff_patterns:
            parts = re.split(pat, text, flags=re.IGNORECASE)
            if len(parts) > 1 and len(parts[0].strip()) > 10:
                text = parts[0].strip()
                break

        # 5. Check for explicit final section markers (e.g. "Final Polish:", "Final Answer:", "Draft:")
        section_markers = [
            r"(?:Final Polish|Final Answer|Final Response|Refined Answer|Draft)\s*:\s*",
        ]
        for marker in section_markers:
            parts = re.split(marker, text, flags=re.IGNORECASE)
            if len(parts) > 1 and len(parts[-1].strip()) > 20:
                text = parts[-1].strip()
                break

        # 6. Filter out lines matching meta-reasoning, rule checks, and self-auditing
        thinking_starters = (
            "the user", "i need to", "i should", "tool 1", "tool 2", "tool 3",
            "wait,", "actually,", "let's check", "i'll keep it", "this follows all rules",
            "no inner monologue", "direct answer", "grounded in", "is there any other",
            "i will state", "or i can just", "the prompt says", "the tool results",
            "user question:", "context:", "tool results:", "analyze tool data:",
            "extract key information", "synthesize the answer:", "refine language:",
            "draft the response:", "refine the response", "opening:", "details:",
            "factors:", "profile:", "customer features:", "draft:", "let's craft",
            "count, but", "rule", "check against rules", "matches all constraints",
            "proceeds. i will output", "final check of the text", "respond immediately?",
            "acknowledge insufficient", "no extra numbers?", "source:"
        )

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            stripped_lower = stripped.lower()

            if not stripped_lower:
                cleaned_lines.append("")
                continue

            # Skip lines starting with monologue/drafting starters
            if any(stripped_lower.startswith(starter) for starter in thinking_starters):
                continue

            # Skip bullet points like "• Respond Immediately? Yes." or "- Acknowledge..."
            if (stripped_lower.startswith("•") or stripped_lower.startswith("-")) and ("?" in stripped_lower or "yes" in stripped_lower or "no" in stripped_lower):
                continue

            # Skip lines that are system rule bullets
            if (stripped_lower.startswith("- ") or stripped_lower.startswith("• ")) and any(rule_kw in stripped_lower for rule_kw in ["base answer", "quote specific", "no inner", "conversational", "tool results", "respond immediately", "extra numbers"]):
                continue

            # Skip meta commentary lines
            if any(meta in stripped_lower for meta in ["this follows all rules", "no inner monologue", "direct answer", "grounded in tool", "let's craft", "matches all constraints", "i will just output the answer"]):
                continue

            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines).strip()
        # Clean quotes if surrounding the entire text
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1].strip()

        return result if result else text

    def _format_results_directly(self, tool_results: List[Dict]) -> str:
        """Fallback answer when LLM call fails — format tool results as text."""
        if not tool_results:
            return "I was unable to retrieve any data for this question. Please check that the model is trained and the dataset is loaded."

        lines = ["Here are the computed results:\n"]
        for tr in tool_results:
            tool = tr["tool"]
            result = tr["result"]
            if result.get("success") and result.get("data"):
                data = result["data"]
                lines.append(f"**{tool}**: {str(data)[:500]}")
            elif not result.get("success"):
                lines.append(f"**{tool}**: Failed — {result.get('error', 'Unknown error')}")
        return "\n".join(lines)

    def clear_memory(self):
        self.memory.clear()
