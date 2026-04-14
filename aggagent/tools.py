import re
import json
from typing import Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool
from rouge_score import rouge_scorer as _rouge_scorer

_scorer = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def _get_content(message: dict, key: str = "content") -> str:
    """Extract text content from a message dict (handles string or list formats)."""
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


def _rouge_l_recall(query: str, text: str) -> float:
    """ROUGE-L recall of query against text."""
    if not query or not text:
        return 0.0
    return _scorer.score(query, text)['rougeL'].recall


def truncate_text(text: str, max_words: int = 150) -> str:
    """Truncate text to first n words."""
    if not text:
        return ""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == max_words:
            return text[:m.end()] + '\n[... truncated]'
    return text


def _count_tokens_approx(messages: list, chars_per_token: float = 4.0) -> int:
    """Approximate token count for a list of messages."""
    total_chars = 0
    for msg in messages:
        for key in ["role", "reasoning_content", "reasoning", "content"]:
            value = msg.get(key)
            if isinstance(value, str):
                total_chars += len(value)
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False))
    return int(total_chars / chars_per_token)


def format_metadata(trajectories: list) -> str:
    """Format trajectory metadata for inclusion in user prompt."""
    blocks = []
    for i, traj in enumerate(trajectories):
        num_steps = len(traj)
        approx_tokens = _count_tokens_approx(traj)

        tool_counts: dict[str, int] = {}
        for msg in traj:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                func = tc.get("function", {})
                if not isinstance(func, dict):
                    try:
                        func = func.to_dict()
                    except Exception:
                        func = {}
                name = func.get("name")
                if name:
                    tool_counts[name] = tool_counts.get(name, 0) + 1

        tool_str = ", ".join(f"{n}×{c}" for n, c in sorted(tool_counts.items())) if tool_counts else "none"
        blocks.append(f"Trajectory {i + 1}: {num_steps} steps, ~{approx_tokens:,} tokens | tools: {tool_str}")

    return "\n\n".join(blocks)


@register_tool("get_solution", allow_overwrite=True)
class GetSolutionTool(BaseTool):
    name = "get_solution"
    description = "Retrieves the final content from trajectories' last step. Returns a list of {trajectory_id, content} entries."
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index. Omit to retrieve all trajectories."}
        },
        "required": []
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> list:
        trajectories = kwargs.get("trajectories", [])
        trajectory_id = params.get("trajectory_id") if isinstance(params, dict) else None

        if trajectory_id is not None:
            n = len(trajectories)
            if trajectory_id < 1 or trajectory_id > n:
                return f"[get_solution] 'trajectory_id' must be 1-{n}"
            trajs = [(trajectory_id - 1, trajectories[trajectory_id - 1])]
        else:
            trajs = list(enumerate(trajectories))

        results = []
        for i, traj in trajs:
            content = (_get_content(traj[-1]) if traj else None) or ""
            results.append({"trajectory_id": i + 1, "content": content})
        return results

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("get_segment", allow_overwrite=True)
class GetSegmentTool(BaseTool):
    name = "get_segment"
    description = "Reads the full content of a contiguous range of steps from a trajectory (max 5). Use after search_trajectory to inspect a step in full with its surrounding context."
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index."},
            "start_step": {"type": "integer", "description": "Start step (inclusive)."},
            "end_step": {"type": "integer", "description": "End step (inclusive); end_step - start_step ≤ 4."}
        },
        "required": ["trajectory_id", "start_step", "end_step"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            trajectory_id = params["trajectory_id"]
            start_step = params["start_step"]
            end_step = params["end_step"]
        except Exception:
            return "[get_segment] Invalid request format: Input must be a JSON object containing 'trajectory_id', 'start_step', and 'end_step' field"

        trajectories = kwargs.get("trajectories", [])
        n_traj = len(trajectories)
        if trajectory_id < 1 or trajectory_id > n_traj:
            return f"[get_segment] 'trajectory_id' must be 1-{n_traj}"
        traj = trajectories[trajectory_id - 1]
        n = len(traj)
        start_step = max(1, min(start_step, n))
        end_step = max(1, min(end_step, n))
        if start_step > end_step:
            start_step = end_step
        if end_step - start_step > 4:
            end_step = start_step + 4
        start_0 = start_step - 1
        end_0 = end_step - 1

        result = []
        for step_idx in range(start_0, end_0 + 1):
            step = traj[step_idx]
            entry = {"step": step_idx + 1, "role": step.get("role", "")}
            content = _get_content(step)
            reasoning = _get_content(step, "reasoning_content") or _get_content(step, "reasoning")
            tool_calls = step.get("tool_calls")
            if content:
                entry["content"] = truncate_text(content, 600)
            if reasoning:
                entry["reasoning"] = truncate_text(reasoning, 600)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
        return result

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("search_trajectory", allow_overwrite=True)
class SearchTrajectoriesTool(BaseTool):
    name = "search_trajectory"
    description = "Searches for keywords or phrases within a single trajectory. Returns top matching steps ranked by relevance score."
    parameters = {
        "type": "object",
        "properties": {
            "trajectory_id": {"type": "integer", "description": "Trajectory index to search within."},
            "query": {"type": "string", "description": "Search term or phrase."},
            "role": {"type": "string", "enum": ["tool", "assistant"], "description": "Optional. Filter to 'tool' steps (actual environment observations) or 'assistant' steps only. Omit to search all steps."},
            "k": {"type": "integer", "description": "Max matches to return (default 5, max 10).", "default": 5}
        },
        "required": ["trajectory_id", "query"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def _score_traj(self, traj_idx, traj, query, role_filter=None):
        scored = []
        for step_idx, step in enumerate(traj):
            if role_filter is not None and step.get("role", "") != role_filter:
                continue
            content = _get_content(step) or ""
            reasoning_content = _get_content(step, "reasoning_content") or _get_content(step, "reasoning") or ""
            tool_calls_str = json.dumps(step.get("tool_calls"), ensure_ascii=False) if step.get("tool_calls") else ""

            score = max(
                _rouge_l_recall(query, content),
                _rouge_l_recall(query, reasoning_content),
                _rouge_l_recall(query, tool_calls_str),
            )
            if score > 0:
                scored.append((score, traj_idx, step_idx, step))
        return scored

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except Exception:
            return "[search_trajectory] Invalid request format: Input must be a JSON object containing 'query' field"

        trajectory_id = params.get("trajectory_id")
        if trajectory_id is None:
            return "[search_trajectory] 'trajectory_id' is required"
        max_results = min(params.get("k", 5), 10)
        role_filter = params.get("role", None)
        trajectories = kwargs.get("trajectories", [])
        n = len(trajectories)

        if trajectory_id < 1 or trajectory_id > n:
            return f"[search_trajectory] 'trajectory_id' must be 1-{n}"

        scored = self._score_traj(trajectory_id - 1, trajectories[trajectory_id - 1], query, role_filter=role_filter)
        scored.sort(key=lambda x: -x[0])

        matches = []
        for score, traj_idx, step_idx, step in scored[:max_results]:
            content = _get_content(step)
            reasoning_content = _get_content(step, "reasoning_content") or _get_content(step, "reasoning")
            tool_calls = step.get("tool_calls")
            match_entry = {
                "trajectory_id": traj_idx + 1,
                "step": step_idx + 1,
                "role": step.get("role", ""),
                "score": round(score, 3),
            }
            if content:
                match_entry["content"] = truncate_text(content)
            if reasoning_content:
                match_entry["reasoning"] = truncate_text(reasoning_content)
            if tool_calls:
                match_entry["tool_calls"] = tool_calls
            matches.append(match_entry)

        if not matches:
            role_msg = f" (role={role_filter})" if role_filter else ""
            return f"[search_trajectory] No matches found for '{query}'{role_msg} in trajectory {trajectory_id}"
        return matches

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("finish", allow_overwrite=True)
class FinishTool(BaseTool):
    name = "finish"

    _PROPERTIES = {
        "solution": {"type": "string", "description": "A comprehensive, standalone solution as a single string with two XML sections: <explanation>detailed reasoning leading to the answer</explanation><answer>the exact answer</answer>. The explanation must be self-contained and make sense without any reference to trajectories or aggregation."},
        "solution_report": {"type": "string", "description": "The synthesized long-form response. Write it as a complete, standalone report — do not reference trajectories, agents, or aggregation. Cite using <cite url=\"...\">...</cite> if necessary."},
        "reason": {"type": "string", "description": "Meta-reasoning explaining your decision: how you evaluated the trajectories, what evidence you relied on, and how you resolved any conflicts or inconsistencies."},
    }

    _QWEN_SOLUTION_DESCRIPTION = (
        "A comprehensive, standalone solution as a single string in the following format:\n"
        "Explanation: {detailed reasoning leading to the answer}\n"
        "Exact Answer: {the exact answer}\n"
        "The explanation must be self-contained and make sense without any reference to trajectories or aggregation."
    )

    def __init__(self, cfg: Optional[dict] = None, variant: str = "", model: str = ""):
        # variant: "" (default) or "long_form"
        super().__init__(cfg)
        self.variant = variant
        self.use_qwen_solution = "qwen" in model.lower()
        if variant == "long_form":
            required = ["solution_report", "reason"]
            self.description = "Submits the final synthesized long-form response."
        else:
            required = ["solution", "reason"]
            self.description = "Submits the final synthesized solution."
        properties = {k: v for k, v in self._PROPERTIES.items() if k in required}
        if self.use_qwen_solution and "solution" in required:
            properties = dict(properties)
            properties["solution"] = {"type": "string", "description": self._QWEN_SOLUTION_DESCRIPTION}
        self.parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        required = self.parameters["required"]
        missing = [f for f in required if f not in params]
        if missing:
            return f"[finish] Invalid request format: missing field(s): {', '.join(missing)}"
        if self.variant == "long_form":
            if not params.get("solution_report", "").strip():
                return "[finish] Invalid format: 'solution_report' must not be empty."
            return {"solution": params["solution_report"], "reason": params.get("reason", "")}
        # default: solution + reason
        solution = params.get("solution", "")
        if self.use_qwen_solution:
            explanation_match = re.search(r'Explanation:\s*(.*?)(?=\nExact Answer:|\Z)', solution, re.DOTALL)
            answer_match = re.search(r'Exact Answer:\s*(\S.*)', solution, re.DOTALL)
            if not explanation_match or not explanation_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty 'Explanation:' section."
            if not answer_match or not answer_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty 'Exact Answer:' section."
        else:
            explanation_match = re.search(r'<explanation>(.*?)</explanation>', solution, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
            if not explanation_match or not explanation_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty <explanation>...</explanation> section."
            if not answer_match or not answer_match.group(1).strip():
                return "[finish] Invalid solution format: missing or empty <answer>...</answer> section."
        return {f: params[f] for f in required}

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }
