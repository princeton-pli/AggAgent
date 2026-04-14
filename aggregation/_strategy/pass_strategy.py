"""Pass@k strategy for aggregation evaluation."""

import math
from collections import defaultdict
from itertools import combinations
from .base import Strategy, CostBreakdown


class PassStrategy(Strategy):
    """
    Pass@k strategy using explicit enumeration.

    For each C(N,k) combination, checks if at least one trajectory is correct
    (taking max score across the k trajectories). Averages across all combinations.
    """

    name = "Pass"

    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        if not results:
            return (0.0, 0.0)

        n = self.get_n(results)
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        num_combinations = math.comb(n, k)
        combo_correct: dict[int, list] = defaultdict(list)
        problem_breakdowns = []

        for problem_id, run_results in results.items():
            if len(run_results) != n:
                continue

            total_breakdown = CostBreakdown()

            for combo_idx, combo in enumerate(combinations(range(n), k)):
                combo_runs = [run_results[i] for i in combo]
                max_score = max(self.is_correct(r) for r in combo_runs)
                combo_correct[combo_idx].append(max_score)
                total_breakdown = total_breakdown + self.calculate_combination_cost_breakdown(combo_runs)

            problem_breakdowns.append(total_breakdown / num_combinations)

        combo_accuracies = [sum(v) / len(v) for v in combo_correct.values()]
        metric = sum(combo_accuracies) / len(combo_accuracies) if combo_accuracies else 0.0
        self._last_problem_scores = combo_accuracies

        avg_breakdown = CostBreakdown()
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        self._last_cost_breakdown = avg_breakdown
        return (metric, avg_breakdown.total)
