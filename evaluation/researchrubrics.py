from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .base import Evaluator


load_dotenv()


# litellm._turn_on_debug()
# Ported from researchrubrics/src/prompts/system_prompt.txt
SYSTEM_PROMPT = """You are an expert evaluator tasked with assessing whether a document satisfies specific rubric criteria. Your evaluation must be precise, objective, and based solely on the evidence present in the document.

## Evaluation Framework

You will evaluate each rubric criterion using a binary satisfaction scale:

1. **Not Satisfied (Score: 0.0)**: The document fails to meet the criterion. Key elements are missing, incorrect, or inadequately addressed.

2. **Satisfied (Score: 1.0)**: The document fully meets the criterion. All required elements are present, well-developed, and appropriately detailed.

## Evaluation Process

1. **Understand the Criterion**: Carefully read and interpret what the rubric is asking for.

2. **Search for Evidence**: Systematically review the document for relevant content that addresses the criterion.

3. **Assess Completeness**: Evaluate whether the evidence satisfies or fails to satisfy the criterion.

4. **Provide Reasoning**: Explain your evaluation with specific references to the document content.

## Important Guidelines

- Base your evaluation ONLY on what is explicitly present in the document
- Do not make assumptions about implied or missing content
- Consider the quality, completeness, and relevance of the evidence
- Be consistent in your evaluation standards across all criteria
- Provide specific examples from the document to support your verdict

## Handling Negative-Weight Criteria

Some criteria describe **undesirable behaviors** and carry a **negative weight**. The Weight field in the rubric indicates this. For these criteria:

- **Satisfied (Score: 1.0)**: The undesirable behavior IS present in the document. This will reduce the overall score.
- **Not Satisfied (Score: 0.0)**: The undesirable behavior is ABSENT from the document. This is the desired outcome and does not penalize the score.

Your verdict must reflect whether the described behavior is literally present or absent in the document — not whether the document is generally good or bad.

Note: Example lists in these rubrics are intended to illustrate possible reasoning patterns or relevant topics. These example lists contain correct answers but are not exhaustive. Use them as guidance, but also make your own final judgment about what qualifies as correct when appropriate."""

# Ported from researchrubrics/src/prompts/user_prompt.txt
USER_PROMPT_TEMPLATE = """## Document Content
{document_content}

## Rubric Criterion to Evaluate

**Title**: {rubric_title}
**Category**: {rubric_category}
**Weight**: {rubric_weight}

## Your Task

Evaluate whether the above document satisfies this specific rubric criterion.

## Required Response Format

Provide your evaluation in the following JSON format:

```json
{{
  "verdict": "[Not Satisfied/Satisfied]",
  "score": [0.0/1.0],
  "confidence": [0.0-1.0],
  "reasoning": "Detailed explanation with specific evidence from the document",
  "evidence_quotes": ["Direct quote 1", "Direct quote 2", ...],
  "missing_elements": ["Element 1 that would improve satisfaction", ...]
}}
```

Ensure your response is ONLY the JSON object, with no additional text."""

# Ported from researchrubrics/src/prompts/chunk_prompt_template.txt
CHUNK_PROMPT_TEMPLATE = """You are evaluating a large document in chunks. This is chunk {chunk_num} of {total_chunks}.

## Previous Context Summary
{context_summary}

## Current Chunk Content
{chunk_content}

## Rubric Criterion
**Title**: {rubric_title}
**Category**: {rubric_category}

Please evaluate this chunk for evidence related to the rubric criterion. Your response should be in JSON format:

```json
{{
  "relevant_evidence": ["Evidence point 1", "Evidence point 2", ...],
  "satisfaction": true/false,
  "confidence_for_chunk": [0.0-1.0],
  "notes": "Any important observations"
}}
```"""

# Ported from researchrubrics/src/prompts/synthesis_prompt_template.txt
SYNTHESIS_PROMPT_TEMPLATE = """Based on the following evidence collected from the document:

Evidence points:
{all_evidence}

Evaluate whether the document satisfies the rubric criterion:
**Title**: {rubric_title}
**Category**: {rubric_category}

Provide your final evaluation in JSON format:
{{
  "verdict": "[Not Satisfied/Satisfied]",
  "score": [0.0/1.0],
  "confidence": [0.0-1.0],
  "reasoning": "Synthesis of evidence"
}}"""

# Tokens per chunk (rough estimate: 1 token ≈ 4 chars)
CHUNK_MAX_TOKENS = 100000
CONTEXT_LIMIT_RESERVE = 50000
MODEL_TOKEN_LIMIT = 200000


@dataclass
class RubricResult:
    criterion: str
    verdict: str
    score: float
    confidence: float
    reasoning: str
    weight: float
    axis: str
    success: bool
    error: Optional[str] = None


class ResearchRubricsEvaluator(Evaluator):

    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent

    # ------------------------------------------------------------------
    # Evaluator interface
    # ------------------------------------------------------------------

    def build_prompt(self, prediction: str, item: Dict, rubric_item: Dict) -> str:
        """Build user prompt for a single rubric item (no-chunk path)."""
        return USER_PROMPT_TEMPLATE.format(
            document_content=prediction,
            rubric_title=rubric_item.get("criterion", ""),
            rubric_category=rubric_item.get("axis", ""),
            rubric_weight=rubric_item.get("weight", 0),
        )

    def default_response(self, err_msg: str = "") -> Dict:
        return {
            "metrics": {},
            "readable_explanation_str": "",
            "rubric_items_with_grades": [],
            "judge_text": err_msg or "Error",
            "confidence": 0,
        }

    def parse_response(self, judge_text: str) -> Dict:
        return self._parse_json(judge_text)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> Dict:
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fix invalid backslash escapes that LLMs sometimes emit inside JSON strings
            # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}")
                return {}

    # ------------------------------------------------------------------
    # Chunking (mirrors researchrubrics workflow)
    # ------------------------------------------------------------------

    def _chunk_content(self, content: str) -> List[str]:
        max_chars = CHUNK_MAX_TOKENS * 4
        if len(content) <= max_chars:
            return [content]
        paragraphs = content.split("\n\n")
        chunks, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < max_chars:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"
        if current:
            chunks.append(current.strip())
        return chunks

    # ------------------------------------------------------------------
    # Single rubric evaluation
    # ------------------------------------------------------------------

    async def _evaluate_single(self, rubric: Dict, content: str, llm: str) -> RubricResult:
        """Evaluate one rubric item against content (no chunking)."""
        user_prompt = self.build_prompt(content, {}, rubric)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(3):
            try:
                text = await self.async_complete(
                    messages, llm, max_tokens=30000, response_format={"type": "json_object"}
                )
                data = self._parse_json(text)
                if not data:
                    raise ValueError("Empty JSON response")
                return RubricResult(
                    criterion=rubric.get("criterion", ""),
                    verdict=data.get("verdict", "Not Satisfied"),
                    score=float(data.get("score", 0.0)),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=data.get("reasoning", ""),
                    weight=float(rubric.get("weight", 0)),
                    axis=rubric.get("axis", ""),
                    success=True,
                )
            except Exception as e:
                if attempt == 2:
                    return RubricResult(
                        criterion=rubric.get("criterion", ""),
                        verdict="Error",
                        score=0.0,
                        confidence=0.0,
                        reasoning=str(e),
                        weight=float(rubric.get("weight", 0)),
                        axis=rubric.get("axis", ""),
                        success=False,
                        error=str(e),
                    )
                await asyncio.sleep(2 ** attempt)

    async def _evaluate_with_chunks(self, rubric: Dict, content: str, llm: str) -> RubricResult:
        """Evaluate one rubric item using chunked content + synthesis."""
        chunks = self._chunk_content(content)
        all_evidence = []

        for i, chunk in enumerate(chunks, 1):
            chunk_prompt = CHUNK_PROMPT_TEMPLATE.format(
                chunk_num=i,
                total_chunks=len(chunks),
                context_summary="Previous chunks evaluated" if i > 1 else "First chunk",
                chunk_content=chunk,
                rubric_title=rubric.get("criterion", ""),
                rubric_category=rubric.get("axis", ""),
            )
            text = await self.async_complete(
                [
                    {"role": "system", "content": "You are evaluating document chunks for rubric criteria."},
                    {"role": "user", "content": chunk_prompt},
                ],
                llm, max_tokens=30000, response_format={"type": "json_object"},
            )
            chunk_data = self._parse_json(text)
            all_evidence.extend(chunk_data.get("relevant_evidence", []))

        synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            all_evidence=json.dumps(all_evidence, indent=2),
            rubric_title=rubric.get("criterion", ""),
            rubric_category=rubric.get("axis", ""),
        )
        text = await self.async_complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": synthesis_prompt},
            ],
            llm, max_tokens=30000, response_format={"type": "json_object"},
        )
        data = self._parse_json(text)
        return RubricResult(
            criterion=rubric.get("criterion", ""),
            verdict=data.get("verdict", "Not Satisfied"),
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            weight=float(rubric.get("weight", 0)),
            axis=rubric.get("axis", ""),
            success=True,
        )

    async def _evaluate_rubric(self, rubric: Dict, content: str, llm: str, semaphore: asyncio.Semaphore) -> RubricResult:
        async with semaphore:
            estimated_tokens = len(content) // 4
            if estimated_tokens > MODEL_TOKEN_LIMIT - CONTEXT_LIMIT_RESERVE:
                return await self._evaluate_with_chunks(rubric, content, llm)
            return await self._evaluate_single(rubric, content, llm)

    # ------------------------------------------------------------------
    # Scoring (mirrors researchrubrics calculate_compliance_score)
    # ------------------------------------------------------------------

    def _calculate_score(self, rubric_items: List[Dict], results: List[RubricResult]) -> Optional[float]:
        denominator = sum(r.weight for r in results if r.weight > 0)
        if denominator == 0:
            return None
        numerator = sum(r.score * r.weight for r in results)
        return numerator / denominator

    def _axis_scores(self, results: List[RubricResult]) -> Dict[str, float]:
        axis_groups: Dict[str, List[RubricResult]] = defaultdict(list)
        for r in results:
            if r.axis:
                axis_groups[r.axis].append(r)
        scores = {}
        for axis, group in axis_groups.items():
            denom = sum(r.weight for r in group if r.weight > 0)
            if denom > 0:
                scores[axis] = sum(r.score * r.weight for r in group) / denom
        return scores

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def compute_score(
        self, prediction: str, item: Dict, llm: str = "gpt-4.1-mini", error: bool = False, err_msg: str = ""
    ) -> Dict:
        if error:
            return self.default_response(err_msg=err_msg)

        rubric_items = item.get("rubrics", [])
        if not rubric_items:
            combo_judgements = item.get("combo_judgements", [])
            if not combo_judgements:
                return self.default_response(err_msg="No rubric found in item")
            rubric_items = [
                {k: r[k] for k in ("criterion", "weight", "axis") if k in r}
                for r in combo_judgements[0].get("rubric_items_with_grades", [])
            ]
            if not rubric_items:
                return self.default_response(err_msg="No rubric found in item")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Evaluate all rubric items in parallel (mirrors evaluate_all_rubrics)
        tasks = [
            self._evaluate_rubric(rubric, prediction, llm, semaphore)
            for rubric in rubric_items
        ]
        results: List[RubricResult] = await asyncio.gather(*tasks)

        overall_score = self._calculate_score(rubric_items, results)
        assert overall_score is not None
        metrics = {"overall_score": overall_score}
        metrics.update(self._axis_scores(results))

        # Build readable explanations
        rubric_items_with_grades = []
        readable_explanation_list = []
        judge_text = ""
        for result in results:
            readable = f"[{result.verdict}] {result.criterion}\n\tReasoning: {result.reasoning}"
            readable_explanation_list.append(readable)
            rubric_items_with_grades.append({
                "criterion": result.criterion,
                "weight": result.weight,
                "axis": result.axis,
                "verdict": result.verdict,
                "score": result.score,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "success": result.success,
            })
            judge_text = result.reasoning  # last one; full text per-rubric is in rubric_items_with_grades

        readable_explanation_list.sort(
            key=lambda x: x.startswith("[Not Satisfied]"), reverse=True
        )
        readable_explanation_str = "\n\n".join(readable_explanation_list)
        readable_explanation_str = f"\n\n{readable_explanation_str}"

        confidence = await self.extract_confidence(prediction, llm=llm)

        return {
            "metrics": metrics,
            "readable_explanation_str": readable_explanation_str,
            "rubric_items_with_grades": rubric_items_with_grades,
            "judge_text": judge_text,
            "confidence": confidence,
        }
