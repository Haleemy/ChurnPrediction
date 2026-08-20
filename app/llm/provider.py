"""
provider.py — LLM provider abstraction.

LLMProvider is the base class.
GroqProvider wraps the Groq API with rate limit handling.
FallbackProvider uses simple rule-based responses for testing/no-API scenarios.

Design: switching providers requires changing only the instantiation.
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.config import GROQ_API_KEY, MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract interface for any LLM backend.
    All agents use this interface — swapping backends is trivial.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Send messages and get a response.
        
        Returns:
            {
              "content": str,           # text content of the response
              "tool_calls": [...],      # if tools were used
              "model": str,
              "usage": {...}            # token usage
            }
        """
        ...

    def is_available(self) -> bool:
        """Check if this provider is configured and accessible."""
        return True


# ── Groq provider ─────────────────────────────────────────────────────────────

class GroqProvider(LLMProvider):
    """
    Groq API provider.
    
    Free tier limitations:
    - Rate limits: ~30 req/min, ~6000 tokens/min for 70B model
    - We minimize calls and use small token budgets.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        max_retries: int = 1,  # Keep low for free tier
    ):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or MODEL_NAME
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. Add it to your .env file:\n"
                    "GROQ_API_KEY=your_key_here"
                )
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        client = self._get_client()
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        tokens = max_tokens or LLM_MAX_TOKENS

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                msg = choice.message

                result = {
                    "content": msg.content or "",
                    "tool_calls": [],
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }

                # Parse tool calls if present
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        except json.JSONDecodeError:
                            args = {}
                        result["tool_calls"].append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": args,
                        })

                return result

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Rate limit — wait and retry once
                if "rate_limit" in error_str or "429" in error_str:
                    if attempt < self.max_retries:
                        wait = 30  # seconds
                        logger.warning(f"Rate limit hit. Waiting {wait}s before retry.")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(
                        "Groq API rate limit reached. Please wait a minute and try again."
                    ) from e

                # Auth error — don't retry
                if "401" in error_str or "invalid" in error_str and "key" in error_str:
                    raise ValueError(
                        "Invalid GROQ_API_KEY. Check your .env file."
                    ) from e

                # Other error — log and raise
                logger.error(f"Groq API error (attempt {attempt+1}): {e}")
                if attempt == self.max_retries:
                    raise RuntimeError(f"LLM call failed after {self.max_retries+1} attempts: {e}") from e
                time.sleep(2)

        raise RuntimeError(f"LLM call failed: {last_error}")


# ── Fallback provider (no API key required) ───────────────────────────────────

class FallbackProvider(LLMProvider):
    """
    Rule-based fallback provider that works without an API key.
    Used for testing and demo mode.
    Returns canned responses that direct users to use tools.
    """

    def __init__(self):
        logger.warning("Using FallbackProvider — no LLM API key configured. Responses will be limited.")

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        # Extract the last user message
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # Detect if this is a planning request (JSON expected)
        if '"steps"' in user_msg or "execution plan" in user_msg.lower() or "PLANNING_PROMPT" in str(messages):
            content = self._generate_fallback_plan(user_msg)
        else:
            content = self._generate_fallback_answer(user_msg, messages)

        return {
            "content": content,
            "tool_calls": [],
            "model": "fallback",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _generate_fallback_plan(self, user_msg: str) -> str:
        """Return a rule-based plan using create_fallback_plan."""
        from app.agent.planner import create_fallback_plan
        plan = create_fallback_plan(user_msg)
        return json.dumps(plan)

    def _generate_fallback_answer(self, user_msg: str, messages: List[Dict]) -> str:
        # Try to extract tool results from the message context
        for m in reversed(messages):
            if "Tool results" in m.get("content", ""):
                return (
                    "⚠️ **No LLM API key configured.** "
                    "I can run tools and return raw results, but natural language interpretation requires a Groq API key. "
                    "Please add GROQ_API_KEY to your .env file.\n\n"
                    "Raw tool results are shown below in the details panel."
                )
        return (
            "⚠️ **Demo mode (no API key).** "
            "Configure GROQ_API_KEY in .env for full agent capabilities. "
            "You can still use the prediction tools and data explorer directly."
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def create_provider() -> LLMProvider:
    """
    Create the best available LLM provider.
    Falls back gracefully if no API key is configured.
    """
    if GROQ_API_KEY:
        logger.info(f"Using Groq provider with model: {MODEL_NAME}")
        return GroqProvider()
    else:
        logger.warning("No GROQ_API_KEY found — using FallbackProvider")
        return FallbackProvider()
