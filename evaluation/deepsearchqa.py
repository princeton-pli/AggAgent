from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import textwrap
from typing import Dict, List, Tuple

import litellm
from dotenv import load_dotenv

from .base import Evaluator

load_dotenv()

DEEPSEARCHQA_INSTRUCTION = """Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

DEEPSEARCH_QA_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), `"Excessive Answers"` (a list of strings indicating the excessive answers), and `"confidence"` (an integer 0-100, extracted from the AI Response if provided, otherwise 0).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")

GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ],
    "confidence": 85
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


@dataclass
class DeepSearchQAJudgeResponse:
    all_correct: bool
    correct_with_excessive_answers: int
    fully_incorrect: int
    precision: float
    recall: float
    f1_score: float
    judge_text: str
    confidence: int = 0

    def to_dict(self) -> Dict:
        return {
            "all_correct": self.all_correct,
            "correct_with_excessive_answers": self.correct_with_excessive_answers,
            "fully_incorrect": self.fully_incorrect,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "judge_text": self.judge_text,
            "confidence": self.confidence,
        }


class DeepSearchQAEvaluator(Evaluator):
    def build_prompt(self, prediction: str, item: Dict) -> str:
        answer = item.get("answer", "")
        question = item.get("question", "")
        answer_type = item.get("answer_type", "")
        return DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
            prompt=question,
            prompt_type=answer_type,
            answer=answer,
            response=prediction,
        )

    def default_response(self, err_msg: str = "") -> Dict:
        return DeepSearchQAJudgeResponse(
            all_correct=False,
            correct_with_excessive_answers=0,
            fully_incorrect=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            judge_text=err_msg or "Error",
            confidence=0,
        ).to_dict()

    def _extract_json_blob(self, raw_text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1)
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw_text[start : end + 1]
        return ""

    def _normalize_correctness_details(self, details_raw) -> Tuple[List[bool], List[str]]:
        # Supports:
        # 1) {"A": true, "B": false}
        # 2) [{"key": "A", "value": true}, ...] (simple-evals style)
        ratings: List[bool] = []
        expected_answers: List[str] = []

        if isinstance(details_raw, dict):
            for k, v in details_raw.items():
                expected_answers.append(str(k))
                ratings.append(bool(v))
            return ratings, expected_answers

        if isinstance(details_raw, list):
            for item in details_raw:
                if isinstance(item, dict):
                    key = item.get("key", "")
                    value = item.get("value", False)
                    expected_answers.append(str(key))
                    ratings.append(bool(value))
            return ratings, expected_answers

        return ratings, expected_answers

    def _calculate_metrics(self, ratings: List[bool], excessive_answers_count: int) -> Dict:
        num_correct = sum(1 for r in ratings if r)
        true_positive = num_correct
        false_negative = len(ratings) - num_correct
        false_positive = excessive_answers_count

        has_expected_answers = bool(ratings)
        all_expected_answers_correct = has_expected_answers and (num_correct == len(ratings))
        fully_incorrect = 1 if (has_expected_answers and num_correct == 0) else 0
        correct_with_excessive_answers = 1 if (false_positive > 0 and (all_expected_answers_correct or not has_expected_answers)) else 0
        all_correct = (all_expected_answers_correct or not has_expected_answers) and false_positive == 0

        precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0.0
        recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0.0
        f1_score = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "all_correct": all_correct,
            "correct_with_excessive_answers": correct_with_excessive_answers,
            "fully_incorrect": fully_incorrect,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def parse_response(self, judge_text: str) -> Dict:
        json_blob = self._extract_json_blob(judge_text)
        if not json_blob:
            return self.default_response(err_msg=judge_text)

        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            return self.default_response(err_msg=judge_text)

        # Supports both prompt formats:
        # - nested {"Answer Correctness": {...}}
        # - flat {"correctness_details": ..., "excessive_answers": ...}
        correctness = parsed.get("Answer Correctness", parsed)
        details = correctness.get("Correctness Details", correctness.get("correctness_details", {}))
        excessive_answers = correctness.get("Excessive Answers", correctness.get("excessive_answers", []))
        confidence = correctness.get("confidence", parsed.get("confidence", 0))

        ratings, _ = self._normalize_correctness_details(details)
        if not isinstance(excessive_answers, list):
            excessive_answers = []

        metrics = self._calculate_metrics(ratings, excessive_answers_count=len(excessive_answers))

        return DeepSearchQAJudgeResponse(
            all_correct=metrics["all_correct"],
            correct_with_excessive_answers=metrics["correct_with_excessive_answers"],
            fully_incorrect=metrics["fully_incorrect"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            judge_text=judge_text,
            confidence=confidence,
        ).to_dict()

    async def compute_score(self, prediction: str, item: Dict, llm: str = "gpt-4.1", error: bool = False, err_msg: str = "") -> Dict:
        """Compute evaluation score using LLM judge."""
        if error:
            result = self.default_response(err_msg=err_msg)
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)
            return result

        judge_prompt = self.build_prompt(prediction, item)

        body = {
            "model": "gpt-4.1",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": judge_prompt}]
        }

        if "gemini" in llm:
            body["model"] = f"gemini/{llm}"
            body["api_key"] = os.getenv("GEMINI_API_KEY")
        elif "gpt" in llm:
            body["model"] = f"openai/{llm}"
            body["api_key"] = os.getenv("OPENAI_API_KEY")
        else:
            body["api_key"] = "EMPTY"

        try:
            judge_response = await litellm.acompletion(**body)
            judge_text = judge_response.choices[0].message.content

            result = self.parse_response(judge_text)
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)

            return result
        except Exception as e:
            result = self.default_response(err_msg=f"Error: {str(e)}")
            if "confidence" not in result:
                result["confidence"] = await self.extract_confidence(prediction, llm=llm)
            return result
