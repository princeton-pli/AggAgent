"""AggAgent strategy for aggregation evaluation."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from tqdm import tqdm

from .base import Strategy, MODEL_COSTS, CostBreakdown, _compute_score
from aggagent import AggAgent as _AggAgent


def _aggregate_stats(all_stats: list[dict], k: int, metric: float, metadata: dict | None = None) -> dict:
    """Summarize per-combo agent stats into a single dict saved alongside logs."""
    total = len(all_stats)
    iterations_list = [s.get("iterations", 0) for s in all_stats]
    server_errors_list = [s.get("server_errors", 0) for s in all_stats]
    tool_call_errors_list = [s.get("tool_call_errors", 0) for s in all_stats]

    aggregated = {
        "metadata": metadata or {},
        "k": k,
        "metric_at_k": metric,
        "total_runs": total,
        "iterations": {
            "total": sum(iterations_list),
            "avg": sum(iterations_list) / total if total else 0,
            "min": min(iterations_list) if iterations_list else 0,
            "max": max(iterations_list) if iterations_list else 0,
        },
        "server_errors": {
            "total": sum(server_errors_list),
            "avg": sum(server_errors_list) / total if total else 0,
        },
        "tool_call_errors": {
            "total": sum(tool_call_errors_list),
            "avg": sum(tool_call_errors_list) / total if total else 0,
        },
        "context_limit_reached_count": sum(1 for s in all_stats if s.get("context_limit_reached", False)),
        "tool_calls": {},
    }
    for s in all_stats:
        for tool_name, count in s.get("tool_calls", {}).items():
            aggregated["tool_calls"][tool_name] = aggregated["tool_calls"].get(tool_name, 0) + count
    return aggregated


class AggAgent(Strategy):
    """
    AggAgent@k (Agent-based Aggregation) strategy.

    Runs an AggregatorAgent over a sampled combination of k trajectories to
    synthesize a final answer, then checks correctness.
    """

    name = "AggAgent"

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        task: str = "browsecomp",
        max_workers: int = 10,
        output_dir: str | None = None,
        resume: bool = False,
        skip_score: bool = False,
        **kwargs,
    ):
        super().__init__(task=task)
        self.model = model
        self.api_base = api_base or ""
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.resume = resume
        self.skip_score = skip_score

        self._log_entries: list[dict] = []
        self._log_lock = threading.Lock()
        self._current_log_file: str | None = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _write_log(self, entry: dict):
        with self._log_lock:
            self._log_entries.append(entry)

    def _flush_logs(self):
        if not self.output_dir or not self._current_log_file:
            return
        with self._log_lock:
            if not self._log_entries:
                return
            with open(self._current_log_file, "a") as f:
                for entry in self._log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._log_entries.clear()

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _calculate_agg_cost(self, stats: dict) -> float:
        """Calculate aggregation LLM cost from agent token_usage_each_step."""
        if self.model not in MODEL_COSTS:
            return 0.0
        input_rate, output_rate = MODEL_COSTS[self.model]
        total_input_delta = 0
        total_output = 0
        prev_input = 0
        for step in stats.get("token_usage_each_step", []):
            curr_input = step.get("input_tokens")
            curr_output = step.get("output_tokens")
            if curr_input is not None:
                total_input_delta += max(curr_input - prev_input, 0)
                prev_input = curr_input
            if curr_output is not None:
                total_output += curr_output
        return (total_input_delta * input_rate + total_output * output_rate) / 1_000_000

    # ------------------------------------------------------------------
    # Per-combo processing
    # ------------------------------------------------------------------

    def _process_single_combo(
        self,
        question: str,
        item: dict,
        instance: dict,
        combo_runs: list[dict],
        combo: tuple,
        k: int,
    ) -> tuple[float, dict, float]:
        """Process one combo. Returns (is_correct, agent_stats, llm_cost)."""
        combo_correct = [self.is_correct(r) for r in combo_runs]
        combo_judgements = [r.get("auto_judge") for r in combo_runs]

        agent = _AggAgent(
            model=self.model,
            api_base=self.api_base,
            task=self.task,
        )

        max_retries = 3
        last_error = None
        agent_result = None
        agent_messages = None
        agent_stats = None
        elapsed = None
        attempts_made = 0

        for attempt in range(max_retries):
            attempts_made = attempt + 1
            try:
                t_start = time.time()
                agent_output = agent._run(question, combo_runs)
                elapsed = time.time() - t_start
                agent_result = agent_output["result"]
                agent_messages = agent_output["messages"]
                agent_stats = agent_output.get("stats", {})

                if agent_result is not None and isinstance(agent_result, dict) and "solution" in agent_result:
                    break
                else:
                    last_error = f"Invalid agent_result: {'None' if agent_result is None else 'missing solution field'}"
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        combo_llm_cost = self._calculate_agg_cost(agent_stats) if agent_stats else 0.0

        if agent_result is None or not isinstance(agent_result, dict) or "solution" not in agent_result:
            log_entry = {
                "metadata": self._current_metadata,
                "auto_judge": None,
                "question": question,
                "instance": instance,
                "prediction": None,
                "is_correct": False,
                "aggregation_cost": combo_llm_cost,
                "aggregation_time": elapsed,
                "aggagent_result": agent_result,
                "aggagent_stats": agent_stats,
                "aggagent_messages": agent_messages,
                "combo": list(combo),
                "combo_correct": combo_correct,
                "combo_judgements": combo_judgements,
                "error": last_error or "Invalid agent result",
                "retries": max_retries,
            }
            self._write_log(log_entry)
            return (0.0, agent_stats or {}, combo_llm_cost)

        prediction = agent_result["solution"]

        if self.skip_score:
            judgement = None
            is_correct = None
        else:
            judgement = _compute_score(prediction, item, self.task)
            is_correct = self.is_correct({"auto_judge": judgement})

        log_entry = {
            "metadata": self._current_metadata,
            "auto_judge": judgement,
            "question": question,
            "instance": instance,
            "prediction": prediction,
            "is_correct": is_correct,
            "aggregation_cost": combo_llm_cost,
            "aggregation_time": elapsed,
            "aggagent_result": agent_result,
            "aggagent_stats": agent_stats,
            "aggagent_messages": agent_messages,
            "combo": list(combo),
            "combo_correct": combo_correct,
            "combo_judgements": combo_judgements,
        }
        if attempts_made > 1:
            log_entry["attempts"] = attempts_made
        self._write_log(log_entry)

        return (float(is_correct) if is_correct is not None else 0.0, agent_stats, combo_llm_cost)

    def _process_single_problem(
        self,
        problem_id: str,
        run_results: list[dict],
        sampled_combos: list[tuple],
        k: int,
    ) -> tuple[float, float, list[dict]]:
        """Process all sampled combos for one problem. Returns (score, avg_agg_cost, stats_list)."""
        question = run_results[0].get("question", "")
        instance = run_results[0].get("instance", {})
        # item = {"question": question, **instance}
        item = {"question": question, **run_results[0]}

        if self.task == "healthbench":
            question = "\n".join(
                str({"role": r["role"], "content": r["content"]})
                for r in instance.get("prompt", [])
            )

        correct_sum = 0.0
        total_agg_cost = 0.0
        stats_list = []

        for combo in sampled_combos:
            combo_runs = [run_results[i] for i in combo]
            is_correct, stats, llm_cost = self._process_single_combo(
                question, item, instance, combo_runs, combo, k
            )
            correct_sum += is_correct
            total_agg_cost += llm_cost
            stats_list.append(stats)

        score = correct_sum / len(sampled_combos) if sampled_combos else 0.0
        avg_agg_cost = total_agg_cost / len(sampled_combos) if sampled_combos else 0.0
        return (score, avg_agg_cost, stats_list)

    # ------------------------------------------------------------------
    # calculate_at_k
    # ------------------------------------------------------------------

    def calculate_at_k(
        self, results: dict[str, list[dict]], k: int
    ) -> tuple[float, float]:
        if not results:
            return (0.0, 0.0)

        n = self.get_n(results)
        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than N ({n})")

        all_combos = list(combinations(range(n), k))
        max_combos = 3
        sampled_combos = self.rng.sample(all_combos, max_combos) if len(all_combos) > max_combos else all_combos

        valid_problems = [(pid, rr) for pid, rr in results.items() if len(rr) == n]

        rollout_metadata = next(iter(results.values()))[0].get("metadata", {})
        self._current_metadata = {
            **rollout_metadata,
            "strategy": self.name,
            "k": k,
            "aggregation_model": self.model,
        }

        # Setup log file; load existing entries when resuming
        existing: dict[str, list[dict]] = {}
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self._current_log_file = os.path.join(self.output_dir, f"aggagent_logs_k{k}.jsonl")
            if self.resume and os.path.exists(self._current_log_file):
                with open(self._current_log_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        existing.setdefault(entry["question"], []).append(entry)
            if not existing:
                open(self._current_log_file, "w").close()

        problem_scores: list[float] = []
        problem_breakdowns: list[CostBreakdown] = []
        all_stats: list[dict] = []

        # Recover from existing log entries
        new_valid_problems = []
        for pid, rr in valid_problems:
            question = rr[0].get("question", "") if rr else ""
            if question in existing:
                entries = existing[question]
                combo_scores = [float(e.get("is_correct") or 0) for e in entries]
                problem_scores.append(sum(combo_scores) / len(combo_scores))
                avg_cost = sum(e.get("aggregation_cost", 0.0) for e in entries) / len(entries)
                traj_breakdown = CostBreakdown()
                for combo in sampled_combos:
                    combo_runs = [rr[i] for i in combo]
                    traj_breakdown = traj_breakdown + self.calculate_combination_cost_breakdown(combo_runs)
                traj_breakdown = traj_breakdown / len(sampled_combos)
                problem_breakdowns.append(CostBreakdown(
                    rollout=traj_breakdown.rollout,
                    tool=traj_breakdown.tool,
                    aggregation=avg_cost,
                ))
                all_stats.extend(e["aggagent_stats"] for e in entries if e.get("aggagent_stats"))
            else:
                new_valid_problems.append((pid, rr))

        if existing:
            print(f"Resuming: {len(existing)} already done, {len(new_valid_problems)} remaining")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_problem, pid, rr, sampled_combos, k): (pid, rr)
                for pid, rr in new_valid_problems
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc=f"AggAgent@{k}")
            for future in pbar:
                pid, rr = futures[future]
                score, agg_cost, stats_list = future.result()
                problem_scores.append(score)
                all_stats.extend(stats_list)

                traj_breakdown = CostBreakdown()
                for combo in sampled_combos:
                    combo_runs = [rr[i] for i in combo]
                    traj_breakdown = traj_breakdown + self.calculate_combination_cost_breakdown(combo_runs)
                traj_breakdown = traj_breakdown / len(sampled_combos)
                problem_breakdowns.append(CostBreakdown(
                    rollout=traj_breakdown.rollout,
                    tool=traj_breakdown.tool,
                    aggregation=agg_cost,
                ))

                if len(problem_scores) % 15 == 0:
                    self._flush_logs()

                if problem_scores:
                    pbar.set_postfix_str(f"{sum(problem_scores)/len(problem_scores)*100:.2f}%")

        self._flush_logs()

        metric = sum(problem_scores) / len(problem_scores) if problem_scores else 0.0
        self._last_problem_scores = problem_scores

        # Write aggregated stats file
        if self.output_dir and all_stats:
            aggregated_stats = _aggregate_stats(all_stats, k, metric, metadata=self._current_metadata)
            stats_file = os.path.join(self.output_dir, f"aggagent_stats_k{k}.json")
            with open(stats_file, "w") as f:
                json.dump(aggregated_stats, f, indent=2)

        avg_breakdown = CostBreakdown()
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        self._last_cost_breakdown = avg_breakdown

        return (metric, avg_breakdown.total)
