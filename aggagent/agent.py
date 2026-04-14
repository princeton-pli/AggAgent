"""AggAgent: tool-calling agent for trajectory aggregation."""

import json
import os
import random
import re
import time

import litellm

from .tools import (
    GetSolutionTool, GetSegmentTool,
    SearchTrajectoriesTool, FinishTool,
    format_metadata, _count_tokens_approx,
)
from .prompts import (
    SYSTEM_PROMPT_AGGAGENT, USER_PROMPT_AGGAGENT,
    SYSTEM_PROMPT_AGGAGENT_QWEN,
    SYSTEM_PROMPT_AGGAGENT_REPORT, USER_PROMPT_AGGAGENT_REPORT,
    FINAL_MESSAGE,
)

LONG_FORM_TASKS = {"healthbench", "researchrubrics"}


class AggAgent:
    """
    AggAgent: agentic aggregation over a set of parallel trajectories.

    Usage::

        agent = AggAgent(model="GLM-4-Flash", api_base="http://...", task="browsecomp")
        result = agent.run(question="...", trajectories=[[msg, ...], [msg, ...]])
        # {"solution": "...", "reason": "..."}
        # or on failure: {"solution": None, "reason": None, "error": "..."}
    """

    def __init__(
        self,
        model: str = "",
        api_base: str | None = None,
        max_context_tokens: int = 100 * 1024,
        task: str = "",
        llm_kwargs: dict | None = None,
    ):
        self.model = model
        self.api_base = api_base or ""
        self.max_context_tokens = max_context_tokens
        self.task = task
        self.llm_kwargs = llm_kwargs or {}

        variant = "long_form" if task in LONG_FORM_TASKS else ""
        tools = [
            GetSolutionTool(),
            SearchTrajectoriesTool(),
            GetSegmentTool(),
            FinishTool(variant=variant, model=model),
        ]

        self.tool_map = {tool.name: tool for tool in tools}
        self.tool_description = [self.tool_map[t].get_tool_definitions() for t in self.tool_map]

    def run(self, question: str, trajectories: list[list[dict]]) -> dict:
        """
        Synthesize a final answer from N parallel trajectories.

        Args:
            question: The task/question string.
            trajectories: List of N trajectories, each a list of message dicts
                          with keys like "role", "content", "tool_calls".

        Returns:
            On success: {"solution": str, "reason": str}
            On failure: {"solution": None, "reason": None, "error": str}
        """
        run_results = [{"messages": traj} for traj in trajectories]
        output = self._run(question, run_results)
        result = output.get("result")
        if result is None or not isinstance(result, dict) or "solution" not in result:
            return {"solution": None, "reason": None, "error": "Agent did not produce a valid solution"}
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def count_tokens_approx(self, messages, chars_per_token=4.0):
        tool_chars = len(json.dumps(self.tool_description, ensure_ascii=False))
        return _count_tokens_approx(messages, chars_per_token) + int(tool_chars / chars_per_token)

    def sanitize_tool_name(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '', name)

    def _message_to_dict(self, assistant_message):
        content = assistant_message.get("content") or ""
        reasoning_content = assistant_message.get("reasoning_content") or ""
        tool_calls_list = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": self.sanitize_tool_name(tc.function.name),
                    "arguments": tc.function.arguments,
                }
            }
            for tc in assistant_message.tool_calls[:1]
        ] if assistant_message.tool_calls else []
        if "gemini" in self.model:
            thinking_blocks = assistant_message.get("thinking_blocks") or []
            return {"role": "assistant", "content": content, "reasoning_content": reasoning_content,
                    "thinking_blocks": thinking_blocks, "tool_calls": tool_calls_list}
        else:
            return {"role": "assistant", "content": content, "reasoning": reasoning_content,
                    "tool_calls": tool_calls_list}

    def call_server(self, messages, max_tries=5, only_finish=False):
        base_sleep_time = 1
        tools = self.tool_description if not only_finish else [self.tool_map["finish"].get_tool_definitions()]
        if self.llm_kwargs:
            litellm_kwargs = {**self.llm_kwargs, "messages": messages,"tools": tools}
        else:
            litellm_kwargs = {
                "model": "hosted_vllm/" + self.model,
                "api_key": "EMPTY",
                "api_base": self.api_base,
                "messages": messages,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 10000,
                "parallel_tool_calls": False,
                "tools": tools,
            }
            if "gemini" in self.model.lower():
                litellm_kwargs["model"] = "gemini/" + self.model
                litellm_kwargs["api_key"] = os.getenv("GEMINI_API_KEY")
                del litellm_kwargs["api_base"]
                litellm_kwargs["reasoning_effort"] = "low"
                del litellm_kwargs["parallel_tool_calls"]
            elif "oss" in self.model.lower():
                litellm_kwargs["reasoning_effort"] = "high"
            elif "gpt" in self.model.lower():
                litellm_kwargs["model"] = "openai/" + self.model
                litellm_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
                del litellm_kwargs["api_base"]
                del litellm_kwargs["top_p"]
            elif "minimax" in self.model.lower():
                litellm_kwargs["extra_body"] = {"reasoning_split": True}

        for attempt in range(max_tries):
            try:
                response = litellm.completion(**litellm_kwargs)
                if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    return response
            except litellm.BadRequestError as e:
                if "context length" in str(e).lower() or "input tokens" in str(e).lower():
                    print(f"Context length exceeded: {e}")
                    return "ContextLengthError"
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error: {e}")

            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                time.sleep(sleep_time)

        return "Server error"

    def _make_output(self, result, messages, stats):
        return {"result": result, "messages": messages, "stats": stats}

    def _run(self, question, run_results):
        trajectories = [r.get("messages", []) for r in run_results]
        metadata = format_metadata(trajectories)
        iteration = 0
        MAX_ITERATIONS = 100

        if self.task in LONG_FORM_TASKS:
            system_prompt = SYSTEM_PROMPT_AGGAGENT_REPORT
            user_prompt = USER_PROMPT_AGGAGENT_REPORT.format(question=question, metadata=metadata, traj_N=len(trajectories))
        elif "qwen" in self.model.lower():
            system_prompt = SYSTEM_PROMPT_AGGAGENT_QWEN
            user_prompt = USER_PROMPT_AGGAGENT.format(question=question, metadata=metadata, traj_N=len(trajectories))
        else:
            system_prompt = SYSTEM_PROMPT_AGGAGENT
            user_prompt = USER_PROMPT_AGGAGENT.format(question=question, metadata=metadata, traj_N=len(trajectories))

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        full_messages = list(messages)

        stats = {
            "iterations": 0,
            "server_errors": 0,
            "tool_call_errors": 0,
            "tool_calls": {},
            "context_limit_reached": False,
            "token_usage_each_step": [],
        }

        def store_token_usage(response, iteration):
            if response is None or isinstance(response, str):
                entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
            elif hasattr(response, "usage") and response.usage:
                entry = {
                    "iteration": iteration,
                    "input_tokens": getattr(response.usage, "prompt_tokens", None),
                    "output_tokens": getattr(response.usage, "completion_tokens", None),
                }
            else:
                entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
            stats["token_usage_each_step"].append(entry)

        while iteration < MAX_ITERATIONS:
            iteration += 1
            stats["iterations"] = iteration
            response = self.call_server(messages)
            store_token_usage(response, iteration)

            if response == "ContextLengthError":
                stats["context_limit_reached"] = True
                return self._make_output(None, full_messages, stats)

            if isinstance(response, str):
                stats["server_errors"] += 1
                error_message = {"role": "assistant", "content": response}
                messages.append(error_message)
                full_messages.append(error_message)
                continue

            assistant_message = response.choices[0].message
            assistant_dict = self._message_to_dict(assistant_message)
            messages.append(assistant_dict)
            full_messages.append(assistant_dict)

            for tool_call in assistant_message.tool_calls[:1] if assistant_message.tool_calls else []:
                tool_name = self.sanitize_tool_name(tool_call.function.name)
                stats["tool_calls"][tool_name] = stats["tool_calls"].get(tool_name, 0) + 1
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_result = self.custom_call_tool(tool_name, tool_args, trajectories=trajectories)
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                except Exception as e:
                    print(e)
                    stats["tool_call_errors"] += 1
                    tool_result = None
                    tool_result_str = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field'

                if tool_name == "finish" and isinstance(tool_result, dict):
                    return self._make_output(tool_result, full_messages, stats)

                tool_message = {"role": "tool", "tool_call_id": tool_call.id, "name": tool_name,
                                "content": tool_result_str}
                messages.append(tool_message)
                full_messages.append(tool_message)

            if self.count_tokens_approx(messages) > self.max_context_tokens:
                stats["context_limit_reached"] = True
                final_msg = {"role": "user", "content": FINAL_MESSAGE}
                messages.append(final_msg)
                full_messages.append(final_msg)
                response = self.call_server(messages, only_finish=True)
                store_token_usage(response, iteration)
                if isinstance(response, str):
                    stats["server_errors"] += 1
                    err_msg = {"role": "assistant", "content": response}
                    messages.append(err_msg)
                    full_messages.append(err_msg)
                    return self._make_output(None, full_messages, stats)

                assistant_message = response.choices[0].message
                assistant_dict2 = self._message_to_dict(assistant_message)
                messages.append(assistant_dict2)
                full_messages.append(assistant_dict2)
                for tc in assistant_message.tool_calls[:1] if assistant_message.tool_calls else []:
                    tn = self.sanitize_tool_name(tc.function.name)
                    stats["tool_calls"][tn] = stats["tool_calls"].get(tn, 0) + 1
                    try:
                        tool_args = json.loads(tc.function.arguments)
                        tool_result = self.custom_call_tool(tn, tool_args, trajectories=trajectories)
                        tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                    except Exception as e:
                        print(e)
                        stats["tool_call_errors"] += 1
                        tool_result = None
                        tool_result_str = "Error: Tool call is not a valid JSON."
                    if tn == "finish" and isinstance(tool_result, dict):
                        return self._make_output(tool_result, full_messages, stats)
                return self._make_output(None, full_messages, stats)

        return self._make_output(None, full_messages, stats)

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in self.tool_map:
            return self.tool_map[tool_name].call(tool_args, **kwargs)
        return f"Error: Tool {tool_name} not found"
