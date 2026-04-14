"""Base strategy class for aggregation evaluation."""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import threading
import time

# Suppress LiteLLM logging-worker noise that fires when litellm.acompletion()
# is called from worker threads (each thread has its own event loop, which
# triggers a queue-reinit race in LiteLLM's GLOBAL_LOGGING_WORKER).
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="coroutine.*was never awaited"
)


# Model costs per 1M tokens (input, output) in USD
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "GLM-4.7-Flash": (0.07, 0.40),
    "MiniMax-M2.5": (0.30, 1.20),
    "gpt-oss-120b": (0.15, 0.60),
    "gemini-3-flash-preview": (0.5, 3.0),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.1": (1.25, 10),
    "Qwen3.5-122B-A10B": (0.26, 2.08),
}


@dataclass
class MetricResult:
    """Result of a metric calculation with optional cost."""
    value: float
    cost: float | None = None


@dataclass
class CostBreakdown:
    """Breakdown of costs by component."""
    rollout: float = 0.0      # LLM token costs during rollout
    tool: float = 0.0         # Tool call costs during rollout (search, scrape)
    aggregation: float = 0.0  # Aggregation LLM costs (added by LLM strategies)

    @property
    def total(self) -> float:
        return self.rollout + self.tool + self.aggregation

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            rollout=self.rollout + other.rollout,
            tool=self.tool + other.tool,
            aggregation=self.aggregation + other.aggregation,
        )

    def __truediv__(self, n: float) -> "CostBreakdown":
        return CostBreakdown(
            rollout=self.rollout / n,
            tool=self.tool / n,
            aggregation=self.aggregation / n,
        )


class Strategy(ABC):
    """Abstract base class for aggregation strategies."""

    name: str = "base"

    def __init__(self, task: str = "browsecomp", seed: int = 42, **kwargs):
        """Initialize strategy.

        Args:
            task: Task type (browsecomp, browsecomp-plus, deepsearchqa, hle, researchrubrics, healthbench)
            seed: Random seed for tie-breaking
        """
        self.task = task
        self.rng = random.Random(seed)

    @abstractmethod
    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> float | tuple[float, float]:
        """
        Calculate the metric at k for all problems.

        Returns:
            If model costs are set: tuple of (metric_value, average_cost)
            Otherwise: metric_value only
        """
        pass

    def run(
        self,
        results: dict[str, list[dict]],
        n: int,
        k_values: list[int] | None = None,
    ) -> dict[int, MetricResult]:
        """
        Run strategy for k=1, 2, 4, ..., N (or specified k_values).

        Returns dict mapping k to MetricResult.
        """
        scores = {}
        if k_values is None:
            k_values = []
            k = 1
            while k <= n:
                k_values.append(k)
                k *= 2

        label_width = max(len(f"{self.name}@{k}") for k in k_values)

        for k in k_values:
            result = self.calculate_at_k(results, k)
            std_str = self._format_std()

            if isinstance(result, tuple):
                metric, cost = result
                scores[k] = MetricResult(value=metric, cost=cost)
            else:
                metric, cost = result, None
                scores[k] = MetricResult(value=metric)

            label = f"{self.name}@{k}"
            std_col = std_str.ljust(9)  # fixed width so cost column always aligns
            cost_str = f"  (${cost:.4f}/prob)" if cost is not None else ""
            print(f"  {label:<{label_width}}  {metric * 100:6.2f}%{std_col}{cost_str}")

        return scores

    def calculate_trajectory_cost_breakdown(self, data: dict) -> CostBreakdown:
        """Return pre-computed rollout cost from result data.

        Cost is computed at rollout time by rollout/utils.compute_rollout_cost
        and saved as data["cost"] = {"rollout": ..., "tool": ...}.  LLM-based
        strategies will add aggregation cost on top via CostBreakdown.aggregation.
        """
        cost = data.get("cost", {})
        return CostBreakdown(
            rollout=cost.get("rollout", 0.0),
            tool=cost.get("tool", 0.0),
        )

    def calculate_combination_cost_breakdown(self, runs: list[dict]) -> CostBreakdown:
        """Calculate cost breakdown for a combination of k trajectories."""
        breakdown = CostBreakdown()
        for r in runs:
            breakdown = breakdown + self.calculate_trajectory_cost_breakdown(r)
        return breakdown

    @staticmethod
    def get_n(results: dict[str, list[dict]]) -> int:
        """Get N (number of runs per problem) from results."""
        if not results:
            return 0
        return len(next(iter(results.values())))

    def is_correct(self, data: dict) -> bool | float:
        """
        Get correctness score for a result.

        Returns bool for binary tasks, float for continuous-score tasks.
        """
        if self.task == "deepsearchqa":
            return data.get("auto_judge", {}).get("all_correct") == True
        elif self.task in ("healthbench", "researchrubrics"):
            score = data.get("auto_judge", {}).get("metrics", {}).get("overall_score")
            if score is not None:
                return score
            rubric_items = (
                data.get("auto_judge", {}).get("rubric_items_with_grades")
                or data.get("rubrics", [])
            )
            if rubric_items:
                field = "points" if self.task == "healthbench" else "weight"
                total_possible = sum(r.get(field, 0) for r in rubric_items if r.get(field, 0) > 0)
                if total_possible > 0:
                    min_val = sum(r.get(field, 0) for r in rubric_items if r.get(field, 0) < 0)
                    return min_val / total_possible
            return 0.0
        return data.get("auto_judge", {}).get("correctness") == "correct"

    @staticmethod
    def extract_answer(data: dict) -> str | None:
        """Extract answer."""
        return data.get("auto_judge", {}).get("extracted_final_answer") or data.get("prediction", "")

    @staticmethod
    def extract_confidence(data: dict) -> float:
        """Extract confidence from auto_judge (0–100 scale), normalized to 0–1. Default 0.0."""
        confidence = data.get("auto_judge", {}).get("confidence")
        if confidence is not None:
            return float(confidence) / 100
        return 0.0

    @staticmethod
    def get_tool_count(data: dict) -> int:
        """Get total tool call count from a result dict."""
        try:
            tool_usage = data.get("debug_data", {}).get("tool_usage", {})
            if tool_usage:
                return sum(tool_usage.values())
        except (KeyError, TypeError):
            pass
        # Fallback: count tool-role messages
        return sum(1 for m in data.get("messages", []) if m.get("role") == "tool")

    def _format_std(self) -> str:
        """Return ' ± X.XX' std string or '' if not available."""
        scores = getattr(self, "_last_problem_scores", None)
        if not scores or len(scores) < 2:
            return ""
        n = len(scores)
        mean = sum(scores) / n
        std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
        return f" ± {std * 100:.2f}%"

    def select_with_tie_breaking(self, items: list, key_func=None, select_func=max):
        """
        Select item with best key value, breaking ties randomly.

        Args:
            items: List of items
            key_func: Function to extract comparison value (None → random selection)
            select_func: max or min

        Returns:
            Selected item, or None if items is empty.
        """
        if not items:
            return None
        if key_func is None:
            return self.rng.choice(items) if len(items) > 1 else items[0]

        values = [key_func(item) for item in items]
        best_value = select_func(values)
        tied_items = [item for item, v in zip(items, values) if v == best_value]
        return self.rng.choice(tied_items) if len(tied_items) > 1 else tied_items[0]


# ---------------------------------------------------------------------------
# Scoring helpers (shared by llm_based and aggagent strategies)
# ---------------------------------------------------------------------------


def _compute_score(prediction: str, item: dict, task: str, llm: str = "gpt-4.1") -> dict:
    """Synchronous wrapper for the async evaluator compute_score."""
    from evaluation import get_evaluator
    evaluator = get_evaluator(task)

    async def _run():
        return await evaluator.compute_score(prediction=prediction, item=item, llm=llm)

    return asyncio.run(_run())
