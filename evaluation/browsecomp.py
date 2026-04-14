from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Dict

import litellm
from dotenv import load_dotenv

from .base import Evaluator

load_dotenv()

BROWSECOMP_INSTRUCTION = """Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

GRADER_TEMPLATE = """
You are an evaluator. Based ONLY on the [correct_answer], judge whether the [response] to the [question] is correct.

=== INPUTS ===
[question]: {question}
[response]: {response}
[correct_answer]: {ground_truth}

=== TASK ===
1. Extract the single final answer from the [response]. If no clear final answer exists, write "None".
2. Give a concise explanation (reasoning) that ONLY compares the extracted answer with the [correct_answer]. Do not solve the problem again or add extra background.
3. Decide correctness: set correctness = correct if they are equivalent / within a tiny numeric tolerance and acceptable difference of expression style; otherwise incorrect. [correct_answer] may contain multiple answers separated by "OR", the response is correct if it matches any of the answers.
4. Extract a confidence score (0-100). If the [response] provides none, use 0.

=== OUTPUT FORMAT (STRICT) ===
Return a valid JSON object with exactly these keys:
{{
  "extracted final answer": <string>,
  "reasoning": <string>,
  "correctness": <string "correct" or "incorrect">,
  "confidence": <integer 0-100>
}}

Do NOT output anything else|no comments, no code fences.
""".strip()


@dataclass
class BrowseCompJudgeResponse:
    extracted_final_answer: str
    correctness: str
    judge_text: str
    reasoning: str
    confidence: int = 0

    def to_dict(self) -> Dict:
        return {
            "extracted_final_answer": self.extracted_final_answer,
            "correctness": self.correctness,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "judge_text": self.judge_text,
        }


class BrowseCompEvaluator(Evaluator):
    def build_prompt(self, prediction: str, item: Dict) -> str:
        answer = item.get("answer", "")
        question = item.get("question", "")
        return GRADER_TEMPLATE.format(question=question, response=prediction, ground_truth=answer)

    def default_response(self, err_msg: str = "") -> Dict:
        return BrowseCompJudgeResponse(
            extracted_final_answer="None",
            correctness="no",
            confidence=0,
            reasoning="None",
            judge_text=err_msg or "Error",
        ).to_dict()

    def parse_response(self, judge_text: str) -> Dict:
        parsed = json.loads(judge_text)
        extracted_final_answer = parsed.get("extracted final answer", "None")
        reasoning = parsed.get("reasoning", "None")
        correctness = parsed.get("correctness", "incorrect")
        confidence = parsed.get("confidence", 0)

        return BrowseCompJudgeResponse(
            extracted_final_answer=extracted_final_answer,
            correctness=correctness,
            confidence=confidence,
            reasoning=reasoning,
            judge_text=judge_text,
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
