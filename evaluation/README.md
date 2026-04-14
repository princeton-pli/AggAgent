# Evaluation

Evaluators are used automatically during rollout (inline auto-scoring) and aggregation (scoring aggregated predictions). You typically do not need to call them directly, but this document describes the API for custom use.

## Overview

Each evaluator inherits from `Evaluator` (in `base.py`) and implements:

| Method | Description |
|---|---|
| `compute_score(prediction, item, llm, error, err_msg)` | Async. Returns a structured score dict. |
| `build_prompt(prediction, item, ...)` | Builds the judge prompt for the given prediction and item. |
| `parse_response(judge_text)` | Parses the LLM judge's response into a structured dict. |
| `default_response(err_msg)` | Returns a zeroed-out score dict for error cases. |

## Evaluators

### `BrowseCompEvaluator` — `browsecomp`, `browsecomp-plus`, `hle`

LLM judge that extracts the final answer from the prediction and checks it against the ground truth. Returns:

```python
{
    "extracted_final_answer": str,
    "correctness": "correct" | "incorrect",
    "confidence": int,   # 0–100, extracted from prediction if present
    "reasoning": str,
    "judge_text": str,
}
```

Score used in aggregation: `correctness == "correct"` → 1.0, else 0.0.

### `DeepSearchQAEvaluator` — `deepsearchqa`

LLM judge that evaluates set-answer correctness (supports single-answer and multi-answer questions). Returns:

```python
{
    "all_correct": bool,
    "correct_with_excessive_answers": int,
    "fully_incorrect": int,
    "precision": float,
    "recall": float,
    "f1_score": float,
    "confidence": int,
    "judge_text": str,
}
```

Score used in aggregation: `all_correct == True` → 1.0, else 0.0.

### `HealthBenchEvaluator` — `healthbench`

Rubric-based evaluation. Each item contains a list of rubric criteria with integer point values and tag groupings; the judge evaluates each criterion independently. Returns:

```python
{
    "metrics": {
        "overall_score": float,   # weighted score across all rubric items
        "<tag>": float,           # per-tag subscores (e.g. "theme:complex_responses")
        ...
    },
    "rubric_items_with_grades": [
        {
            "criterion": str,
            "points": int,        # positive or negative
            "tags": list[str],
            "criteria_met": bool,
            "explanation": str,
        },
        ...
    ],
    "readable_explanation_str": str,
    "confidence": int,
    "judge_text": str,
}
```

Score used in aggregation: `metrics["overall_score"]` (0.0–1.0).

### `ResearchRubricsEvaluator` — `researchrubrics`

Rubric-based evaluation with continuous per-criterion scores. Rubric items use float weights and axis groupings instead of integer points and tags. Long documents are automatically chunked and evaluated in segments, then synthesized. Returns:

```python
{
    "metrics": {
        "overall_score": float,   # weighted score across all rubric items
        "<axis>": float,          # per-axis subscores (e.g. "Explicit Criteria")
        ...
    },
    "rubric_items_with_grades": [
        {
            "criterion": str,
            "weight": float,      # can be negative (undesirable behavior)
            "axis": str,
            "verdict": str,       # "Satisfied" | "Not Satisfied"
            "score": float,       # 0.0 or 1.0
            "confidence": float,  # 0.0–1.0
            "reasoning": str,
            "success": bool,      # False if judge call failed
        },
        ...
    ],
    "readable_explanation_str": str,
    "confidence": int,
    "judge_text": str,
}
```

Score used in aggregation: `metrics["overall_score"]` (0.0–1.0).

## Usage

```python
import asyncio
from evaluation import get_evaluator

evaluator = get_evaluator("browsecomp")

item = {
    "question": "What is the capital of France?",
    "answer": "Paris",
}
result = asyncio.run(
    evaluator.compute_score(prediction="The answer is Paris.", item=item)
)
print(result["correctness"])  # "correct"
```

## Judge Model

All evaluators default to `gpt-4.1` as the judge model. You can override via the `llm` argument to `compute_score`:

```python
result = asyncio.run(
    evaluator.compute_score(prediction="...", item=item, llm="gpt-4o-mini")
)
```

Supported judge model prefixes:
- `gpt-*` — OpenAI (reads `OPENAI_API_KEY`)
- `gemini-*` — Google Gemini (reads `GEMINI_API_KEY`)
- `Qwen*` — local vLLM (reads `EVAL_API_BASE`, default `http://localhost:7000/v1`)
