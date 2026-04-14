from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import re
from typing import Any, Dict, Optional

import litellm


CONFIDENCE_EXTRACTION_PROMPT = """Extract the confidence score from the AI response below.
Find the line starting with "Confidence:" and return its integer value (0-100).

AI Response:
{prediction}

Return ONLY a JSON object: {{"confidence": <integer>}}
If no confidence line is found, return {{"confidence": 0}}."""


def _build_litellm_body(llm: str, messages: list, max_tokens: int, **extra) -> Dict[str, Any]:
    """Build litellm body with provider routing."""
    body: Dict[str, Any] = {"max_tokens": max_tokens, "messages": messages}
    body.update(extra)
    if "gemini" in llm:
        body["model"] = f"gemini/{llm}"
        body["api_key"] = os.getenv("GEMINI_API_KEY")
    elif "gpt" in llm or "openai" in llm:
        body["model"] = f"openai/{llm}" if not llm.startswith("openai/") else llm
        body["api_key"] = os.getenv("OPENAI_API_KEY")
    elif "Qwen" in llm:
        body["model"] = f"hosted_vllm/{llm}"
        body["api_base"] = os.getenv("EVAL_API_BASE", "http://localhost:7000/v1")
        body.setdefault("temperature", 0.7)
        body.setdefault("top_p", 0.8)
        body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        body["model"] = llm
        body["api_key"] = "EMPTY"
    return body


class Evaluator(ABC):
    _completer: Optional[Any] = None  # TinkerMessageCompleter when set

    def set_completer(self, completer: Any) -> None:
        """Inject a TinkerMessageCompleter to use instead of litellm."""
        self._completer = completer

    async def async_complete(self, messages: list, llm: str, max_tokens: int = 1024, **extra) -> str:
        """Call either the tinker completer or litellm, returning the response text."""
        if self._completer is not None:
            result = await self._completer(messages)
            return result["content"]
        body = _build_litellm_body(llm, messages, max_tokens, **extra)
        resp = await litellm.acompletion(**body)
        return resp.choices[0].message.content

    async def extract_confidence(self, prediction: str, llm: str = "gpt-4.1-mini") -> int:
        """Short eval call to extract a structured 'Confidence: X%' from the prediction."""
        if not prediction or not prediction.strip():
            return 0
        prompt = CONFIDENCE_EXTRACTION_PROMPT.format(prediction=prediction)
        try:
            text = await self.async_complete(
                [{"role": "user", "content": prompt}], llm, max_tokens=32, temperature=0.0
            )
            json_cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
            parsed = json.loads(json_cleaned)
            confidence = parsed.get("confidence", 0)
            if isinstance(confidence, (int, float)):
                return max(0, min(100, int(confidence)))
        except Exception as e:
            print(f"Failed to extract confidence: {e}")
        return 0

    @abstractmethod
    def build_prompt(self, prediction: str, item: Dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, judge_text: str) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def default_response(self, err_msg: str = "") -> Dict:
        raise NotImplementedError

    @abstractmethod
    async def compute_score(self, prediction: str, item: Dict, error: bool = False, err_msg: str = "") -> Dict:
        """Compute the evaluation score for a prediction."""
        raise NotImplementedError
