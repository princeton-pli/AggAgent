"""LLM-based aggregation strategies (SolAgg, SummAgg)."""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import litellm
from tqdm import tqdm

from .base import Strategy, MODEL_COSTS, CostBreakdown, _compute_score


REPORT_PROMPT = """Given the following problem-solving trajectory:
{traj}

Your task is to distill this trajectory into a concise yet sufficiently detailed problem-solving report.
You must only use the information contained within the provided trajectory — no additional or external information is allowed.

The report must include:

1. **Solution Planning**: Identify how the main problem is decomposed into subproblems, and describe the sequence and dependency relationships among those subproblems.
2. **Solution Methods**: For each subproblem, indicate which tools were invoked to solve it, the parameters used in those tool calls, and any resulting partial answers that contributed directly or indirectly to progress toward the final answer.
   _Do not repeat the full output of the tools; only include the specific fragments of tool results that were essential in deriving the subanswers._
3. **Final Reasoning**: Clearly outline the reasoning process by which the subproblems and their associated subanswers led to the derivation of the final answer.

Additional requirements:
- The report must remain **concise** and **focused**.
- Remove any content unrelated to problem-solving or any ineffective tool calls.
- Ensure the final report has clear logical structure, with each step traceable and analyzable.

Finally, present the complete report in **Markdown format**, and wrap the entire report content within <report> </report> tags.
""".strip()

INTEGRATE_PROMPT = """You are tasked with solving the question: {question}.

Multiple independent teams have provided detailed process reports describing their approaches to solving this problem. As the final analyst, your role is to consolidate these reports, carefully examine the problem-solving methods they contain, and identify the key information obtained in each.

Your goal is to produce a final answer that perfectly resolves the question. Note that some of the reports may contain inconsistencies — you must critically evaluate which report(s) are reasonable and trustworthy.

If multiple reports reach the same conclusion, this increases the likelihood that the conclusion is correct; however, this is not guaranteed. You must still carefully verify and reflect to ensure that the final selected answer is truly the most accurate possible.

Wrap your final answer in <answer> </answer> tags and provide a brief explanation for your final answer in <explanation> </explanation> tags.

Important:
- Every question has a definitive, certain answer.
- You are not allowed to decline answering on the grounds of uncertainty.
- For any report that does not provide a clear and definite final answer, its confidence level should be significantly reduced.
- You must ultimately select one report as having the most correct answer.
- You are not allowed to call or use any external tools for verification. You must rely solely on the information already provided, conduct in-depth analysis, and then produce the final answer.
- You are not allowed to merge multiple different answers, nor are you allowed to produce an overly broad answer that attempts to encompass all candidate answers — such an answer should be eliminated first.
- You do not need to restate or summarize the reports; instead, provide a short-form answer that directly answers the question.

Below is the content of these reports:
""".strip()

INTEGRATE_PROMPT_REPORT = """You are tasked with solving the task: {question}.

Multiple independent teams have provided detailed process reports describing their approaches to solving this problem. As the final analyst, your role is to consolidate these reports, carefully examine the problem-solving methods they contain, and identify the key information obtained in each.

Your goal is to produce a final report that perfectly resolves the problem. Note that some of the reports may contain inconsistencies — you must critically evaluate which report(s) are reasonable and trustworthy.

If multiple reports reach the same conclusion, this increases the likelihood that the conclusion is correct; however, this is not guaranteed. You must still carefully verify and reflect to ensure that the final report is truly the most accurate possible.

Produce a comprehensive final report that directly addresses the task. Cite every nontrivial claim using <cite url="...">...</cite> tags drawn only from URLs mentioned in the provided reports — never fabricate URLs or content.

Important:
- You are not allowed to decline answering on the grounds of uncertainty.
- For any report that does not provide a clear and definite solution, its confidence level should be significantly reduced.
- You are not allowed to call or use any external tools for verification. You must rely solely on the information already provided, conduct in-depth analysis, and then produce the final report.

Below is the content of these reports:
""".strip()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_content(message: dict, key: str = "content") -> str:
    """Extract content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            text = value[0].get("text") or ""
            if key == "content":
                recipient = message.get("recipient")
                name = message.get("name")
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    return ""


def _construct_interaction(messages: list[dict]) -> str:
    """Convert a messages list into a readable trajectory string."""
    interaction = ""
    for r in messages:
        if r["role"] in ("developer", "system"):
            continue
        elif r["role"] == "user":
            interaction += f"**User:**\n{r['content']}\n\n"
        elif r["role"] == "tool":
            interaction += f"**Tool Response:**\n{r['content']}\n\n"
        elif r["role"] == "assistant":
            interaction += "**Assistant:**\n"
            if r.get("reasoning_content") is not None:
                interaction += f"*Thinking:* {r['reasoning_content']}\n"
            if r.get("reasoning") is not None:
                interaction += f"*Thinking:* {r['reasoning']}\n"
            if r.get("content") is not None:
                interaction += f"*Content:* {r['content']}\n"
            if r.get("tool_calls") is not None:
                func = r["tool_calls"][0]["function"]
                if not isinstance(func, dict):
                    func = func.to_dict()
                interaction += f"*Tool Call:* {json.dumps(func)}\n"
            interaction += "\n"
    return interaction.strip()


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _generate_compact_summary(
    messages: list[dict], model: str = "gpt-4o-mini", api_base: str | None = None
) -> tuple[str, dict]:
    """Summarize a trajectory using REPORT_PROMPT. Returns (summary, usage)."""
    trajectory = _construct_interaction(messages)
    prompt = REPORT_PROMPT.format(traj=trajectory)

    kwargs: dict = {
        "model": "hosted_vllm/" + model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if "gpt" in model and "oss" not in model:
        kwargs["model"] = "openai/" + model
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        kwargs["max_tokens"] = 10000
    if api_base:
        kwargs["api_base"] = api_base

    content = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }
            if content is not None:
                break
        except Exception as e:
            print(f"[generate_compact_summary] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
    if not content:
        return "", usage
    match = re.search(r"<report>(.*?)</report>", content, re.DOTALL)
    if match:
        return match.group(1).strip(), usage
    return content, usage


def _integrate_and_answer(
    question: str,
    reports: list[str],
    model: str = "gpt-4o-mini",
    api_base: str | None = None,
    use_report_prompt: bool = False,
) -> tuple[str, dict, str]:
    """Integrate reports and produce a final answer. Returns (content, usage, prompt)."""
    prompt_template = INTEGRATE_PROMPT_REPORT if use_report_prompt else INTEGRATE_PROMPT
    prompt = prompt_template.format(question=question)
    for i, report in enumerate(reports, 1):
        prompt += f"\n\n## Report {i}\n{report}"

    kwargs: dict = {
        "model": "hosted_vllm/" + model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if api_base:
        kwargs["api_base"] = api_base
    if "gpt" in model and "oss" not in model:
        kwargs["model"] = "openai/" + model
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        kwargs["max_tokens"] = 10000

    content = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }
            if content is not None:
                break
        except Exception as e:
            print(f"[integrate_and_answer] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))

    # Strip control characters that break JSON serialization
    if content:
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)
    return content or "", usage, prompt


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------

class SolAgg(Strategy):
    """
    Sol@k (Solution Integration) strategy.

    For each sampled C(N,k) combination, collects predictions from each run,
    integrates them using an LLM, and checks correctness.
    """

    name = "SolAgg"

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        task: str = "browsecomp",
        max_workers: int = 10,
        output_dir: str | None = None,
        skip_score: bool = False,
        resume: bool = False,
        **kwargs,
    ):
        super().__init__(task=task)
        self.model = model
        self.api_base = api_base
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.skip_score = skip_score
        self.resume = resume
        self.compact = False  # overridden by SummAgg

        self._log_entries: list[dict] = []
        self._log_lock = threading.Lock()
        self._current_log_file: str | None = None

    # ------------------------------------------------------------------
    # Logging helpers
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
    # Cost helper
    # ------------------------------------------------------------------

    def _calculate_llm_cost(self, usage: dict) -> float:
        """Calculate aggregation LLM cost from token usage."""
        if self.model not in MODEL_COSTS:
            return 0.0
        input_cost_per_m, output_cost_per_m = MODEL_COSTS[self.model]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return (prompt_tokens * input_cost_per_m + completion_tokens * output_cost_per_m) / 1_000_000

    # ------------------------------------------------------------------
    # Per-problem processing
    # ------------------------------------------------------------------

    def _process_single_problem(
        self,
        problem_id: str,
        run_results: list[dict],
        sampled_combos: list[tuple],
    ) -> tuple[float, float]:
        """Process one problem over all sampled combos. Returns (score, avg_llm_cost)."""
        question = run_results[0].get("question", "")
        instance = run_results[0].get("instance", {})
        # item = {"question": question, **instance}
        item = {"question": question, **run_results[0]}

        if self.task == "healthbench":
            question = "\n".join(
                str({"role": r["role"], "content": r["content"]})
                for r in run_results[0].get("instance", {}).get("prompt", [])
            )

        pass_count = 0
        total_llm_cost = 0.0

        for combo in sampled_combos:
            combo_runs = [run_results[i] for i in combo]
            combo_llm_cost = 0.0
            combo_correct = [self.is_correct(r) for r in combo_runs]
            combo_judgements = [r.get("auto_judge") for r in combo_runs]

            t_start = time.time()
            reports = []

            if self.compact:
                def _summarize(r):
                    messages = r.get("messages", [])
                    if messages:
                        return _generate_compact_summary(
                            messages, model=self.model, api_base=self.api_base
                        )
                    return None, {"prompt_tokens": 0, "completion_tokens": 0}

                with ThreadPoolExecutor(max_workers=len(combo_runs)) as pool:
                    futs = [pool.submit(_summarize, r) for r in combo_runs]
                    for fut in futs:
                        summary, usage = fut.result()
                        if summary:
                            reports.append(summary)
                            combo_llm_cost += self._calculate_llm_cost(usage)
            else:
                for r in combo_runs:
                    prediction = r.get("prediction", "")
                    if prediction:
                        reports.append(prediction)

            if reports:
                integrated_response, usage, integrate_prompt = _integrate_and_answer(
                    question,
                    reports,
                    model=self.model,
                    api_base=self.api_base,
                    use_report_prompt=self.task in ("healthbench", "researchrubrics"),
                )
                elapsed = time.time() - t_start
                combo_llm_cost += self._calculate_llm_cost(usage)

                if self.skip_score:
                    judge_result = None
                    is_correct = None
                else:
                    judge_result = _compute_score(integrated_response, item, self.task)
                    is_correct = self.is_correct({"auto_judge": judge_result})
                    pass_count += is_correct

                log_entry = {
                    "metadata": self._current_metadata,
                    "auto_judge": judge_result,
                    "question": question,
                    "instance": instance,
                    "prediction": integrated_response,
                    "is_correct": is_correct,
                    "aggregation_cost": combo_llm_cost,
                    "aggregation_time": elapsed,
                    "prompt": integrate_prompt,
                    "combo": list(combo),
                    "combo_correct": combo_correct,
                    "combo_judgements": combo_judgements,
                }
                self._write_log(log_entry)

            total_llm_cost += combo_llm_cost

        score = pass_count / len(sampled_combos) if sampled_combos else 0.0
        avg_llm_cost = total_llm_cost / len(sampled_combos) if sampled_combos else 0.0
        return score, avg_llm_cost

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
        if len(all_combos) > max_combos:
            sampled_combos = self.rng.sample(all_combos, max_combos)
        else:
            sampled_combos = all_combos

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
            label_lower = "summagg" if self.compact else "solagg"
            self._current_log_file = os.path.join(
                self.output_dir, f"{label_lower}_logs_k{k}.jsonl"
            )
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
        label = "SummAgg" if self.compact else "SolAgg"

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
            else:
                new_valid_problems.append((pid, rr))

        if existing:
            print(f"Resuming: {len(existing)} already done, {len(new_valid_problems)} remaining")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_problem, pid, rr, sampled_combos): (pid, rr)
                for pid, rr in new_valid_problems
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc=f"{label}@{k}")
            for future in pbar:
                pid, rr = futures[future]
                score, llm_cost = future.result()
                problem_scores.append(score)

                traj_breakdown = CostBreakdown()
                for combo in sampled_combos:
                    combo_runs = [rr[i] for i in combo]
                    traj_breakdown = traj_breakdown + self.calculate_combination_cost_breakdown(combo_runs)
                traj_breakdown = traj_breakdown / len(sampled_combos)

                problem_breakdowns.append(CostBreakdown(
                    rollout=traj_breakdown.rollout,
                    tool=traj_breakdown.tool,
                    aggregation=llm_cost,
                ))

                if len(problem_scores) % 20 == 0:
                    self._flush_logs()

                running_avg = sum(problem_scores) / len(problem_scores) * 100
                pbar.set_postfix_str(f"{running_avg:.2f}%")

        self._flush_logs()

        metric = sum(problem_scores) / len(problem_scores) if problem_scores else 0.0
        self._last_problem_scores = problem_scores

        avg_breakdown = CostBreakdown()
        for b in problem_breakdowns:
            avg_breakdown = avg_breakdown + b
        if problem_breakdowns:
            avg_breakdown = avg_breakdown / len(problem_breakdowns)
        self._last_cost_breakdown = avg_breakdown

        if self.output_dir and problem_scores:
            n_probs = len(problem_scores)
            n_combos = len(sampled_combos)
            total_agg_cost = avg_breakdown.aggregation * n_probs
            stats = {
                "metadata": self._current_metadata,
                "metric_at_k": metric,
                "n_problems": n_probs,
                "combos_per_problem": n_combos,
                "total_combos": n_probs * n_combos,
                "aggregation_cost": {
                    "total": total_agg_cost,
                    "avg_per_problem": avg_breakdown.aggregation,
                    "avg_per_combo": total_agg_cost / (n_probs * n_combos) if n_combos else 0.0,
                },
            }
            stats_file = os.path.join(self.output_dir, f"{label_lower}_stats_k{k}.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

        return (metric, avg_breakdown.total)


class SummAgg(SolAgg):
    """
    Summ@k (Compact Solution Integration) strategy.

    Same as Sol@k but first compresses each run's trajectory into a
    concise report using REPORT_PROMPT before integration.
    """

    name = "SummAgg"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compact = True
