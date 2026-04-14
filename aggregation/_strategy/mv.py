"""Majority Voting strategies for aggregation evaluation."""

import math
from itertools import combinations
from collections import defaultdict
from .base import Strategy, CostBreakdown


class MVStrategy(Strategy):
    """
    MV@k (Majority Voting) strategy.

    For each C(N,k) combination, select the run whose answer has the most votes,
    then check its correctness.
    """

    name = "MV"

    def majority_vote(self, runs: list[dict], weighted: bool = False) -> dict | None:
        """
        Return the run with the answer that has the highest vote count.

        If weighted=True, each vote is weighted by the run's confidence.
        Ties are broken randomly via self.rng.
        """
        votes: dict[str, float] = defaultdict(float)
        answer_to_runs: dict[str, list] = defaultdict(list)

        for r in runs:
            answer = Strategy.extract_answer(r)
            if answer is not None:
                weight = Strategy.extract_confidence(r) if weighted else 1.0
                votes[answer] += weight
                answer_to_runs[answer].append(r)

        if not votes:
            # No extractable answers: random (MV) or highest-confidence (WMV) fallback
            if weighted:
                return self.select_with_tie_breaking(runs, Strategy.extract_confidence, max)
            return self.select_with_tie_breaking(runs)

        best_answer = self.select_with_tie_breaking(
            list(votes.keys()),
            key_func=lambda ans: votes[ans],
            select_func=max,
        )
        return self.select_with_tie_breaking(answer_to_runs[best_answer])

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
                selected = self.majority_vote(combo_runs, weighted=False)
                combo_correct[combo_idx].append(self.is_correct(selected) if selected is not None else 0)
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


class WMVStrategy(MVStrategy):
    """
    WMV@k (Weighted Majority Voting) strategy.

    Same as MV but votes are weighted by each run's confidence score.
    """

    name = "WMV"

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
                selected = self.majority_vote(combo_runs, weighted=True)
                combo_correct[combo_idx].append(self.is_correct(selected) if selected is not None else 0)
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
