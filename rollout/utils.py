"""Rollout utilities for AggAgent."""

import json

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

# Tool usage costs per 1K calls in USD
SEARCH_COST_PER_K = 0.5
SCRAPE_COST_PER_K = 0.83


def compute_rollout_cost(debug_data: dict, model_name: str, task: str | None = None) -> dict:
    """
    Compute cost breakdown for a single rollout.

    Args:
        debug_data: The debug_data dict (token_lengths_each_step, tool_usage)
        model_name: Model basename (e.g. "GLM-4.7-Flash") for cost lookup
        task: Task name; browsecomp-plus uses local FAISS tools (no tool cost)

    Returns:
        {"rollout": token_cost, "tool": tool_cost} in USD, or zeros if model unknown.
    """
    costs = MODEL_COSTS.get(model_name)
    if not costs:
        return {"rollout": 0.0, "tool": 0.0}

    input_cost_per_m, output_cost_per_m = costs

    # --- token cost (incremental input + output) ---
    total_input_delta = 0
    total_output = 0
    prev_input = 0
    try:
        for step in debug_data.get("token_lengths_each_step", []):
            curr_input = step.get("input_tokens", 0)
            total_input_delta += max(curr_input - prev_input, 0)
            prev_input = curr_input
            total_output += step.get("output_tokens", 0)
    except (KeyError, TypeError):
        pass

    token_cost = (
        total_input_delta * input_cost_per_m + total_output * output_cost_per_m
    ) / 1_000_000

    # --- tool call cost (external API tools only) ---
    # browsecomp-plus uses local FAISS-based tools, so no tool cost applies.
    tool_cost = 0.0
    if task != "browsecomp-plus":
        try:
            tool_usage = debug_data.get("tool_usage", {})
            search_count = tool_usage.get("search", 0)
            scrape_count = tool_usage.get("visit", 0)
            tool_cost = (
                search_count * SEARCH_COST_PER_K + scrape_count * SCRAPE_COST_PER_K
            ) / 1_000
        except (KeyError, TypeError):
            pass

    return {"rollout": token_cost, "tool": tool_cost}


def store_token_length(
    debug_data: dict,
    iteration: int,
    response=None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Append a token-length entry to debug_data["token_lengths_each_step"]."""
    if response is None or isinstance(response, str):
        token_entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
    elif input_tokens is not None or output_tokens is not None:
        token_entry = {"iteration": iteration, "input_tokens": input_tokens, "output_tokens": output_tokens}
    elif hasattr(response, "usage") and response.usage:
        token_entry = {
            "iteration": iteration,
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(response.usage, "completion_tokens", None),
        }
    else:
        token_entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
    debug_data["token_lengths_each_step"].append(token_entry)


def prepare_messages_for_tokenization(messages: list, model_type: str | None) -> list:
    """
    Normalize messages for tokenizer.apply_chat_template.

    Renames "reasoning" → "reasoning_content" and, for GLM/Qwen/MiniMax models,
    deserializes tool_call arguments from JSON strings to dicts.
    """
    normalized = []
    for msg in messages:
        prep = dict(msg)
        if "reasoning" in prep:
            prep["reasoning_content"] = prep.pop("reasoning")
        normalized.append(prep)

    if model_type in ("glm", "qwen", "minimax"):
        prepared = []
        for msg, prep in zip(messages, normalized):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls_list = [
                    {
                        **tc,
                        "function": {
                            **tc["function"],
                            "arguments": json.loads(tc["function"]["arguments"]),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
                prepared.append({**prep, "tool_calls": tool_calls_list})
            else:
                prepared.append(prep)
        return prepared

    return normalized
