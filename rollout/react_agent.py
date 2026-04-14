import json
import json5
import os
import asyncio
import copy
import random
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.tools import BaseTool
import litellm
from transformers import AutoTokenizer
from prompts import *
from evaluation import get_task_instruction, get_evaluator

TRUNCATED_MESSAGE = """
--- Maximum Length Limit Reached ---
You have reached the maximum length limit.
The response is truncated."""
FINAL_MESSAGE = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer."

from tools.tool_search import *
from tools.tool_visit import *
from tools.tool_search_bcp import *
from tools.tool_get_document_bcp import *
from utils import compute_rollout_cost, store_token_length, prepare_messages_for_tokenization


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self, function_list: Optional[List[Union[str, Dict, BaseTool]]] = None, llm: Optional[Dict] = None, **kwargs):
        self.llm_generate_cfg = llm["generate_cfg"] if llm else {}
        self.model_type = llm.get("model_type") if llm else None
        self.model_name = llm.get("model_name") if llm else None
        self.tokenizer = AutoTokenizer.from_pretrained(llm["model"]) if llm else None
        self.task = kwargs.get("task", None)
        self.max_llm_call_per_run = kwargs.get("max_llm_call_per_run", 100)
        self.max_tokens = kwargs.get("max_tokens", 108000)
        self.system_prompt = SYSTEM_PROMPT

        if self.task == "browsecomp-plus":
            self.tool_class = [SearchBCP(), GetDocumentBCP()]
            self.function_list = ["search_bcp", "get_document_bcp"]
        elif self.task in ["browsecomp", "deepsearchqa", "hle", "healthbench", "researchrubrics"]:
            self.tool_class = [Search(), Visit()]
            self.function_list = ["visit", "search"]
        else:
            self.tool_class = []
            self.function_list = []

        self.tool_map = {tool.name: tool for tool in self.tool_class} if self.tool_class else {}
        self.tool_description = [self.tool_map[tool].get_tool_definitions() for tool in self.tool_map] if self.tool_class else []

    async def call_server(self, msgs: List[Dict[str, Any]], max_tries: int = 3, no_tool_calls: bool = False, tool_description: List[Dict[str, Any]] = None) -> Union[Any, str]:
        kwargs = self.llm_generate_cfg.copy()
        kwargs["messages"] = msgs
        if not no_tool_calls and tool_description:
            kwargs["tools"] = tool_description

        base_sleep_time = 1
        response = None
        for attempt in range(max_tries):
            try:
                response = await litellm.acompletion(**kwargs)
                content = response.choices[0].message.content
                reasoning_content = response.choices[0].message.get("reasoning_content") or ""
                if content and content.strip():
                    return response
                elif hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    return response
                elif reasoning_content and reasoning_content.strip():
                    return response
            except Exception as e:
                print(f"Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        print(f"Response dump: {e.response.text if hasattr(e.response, 'text') else e.response}")
                    except:
                        print(f"Response object: {e.response}")
            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                await asyncio.sleep(sleep_time)

        if response:
            print(f"Response dump: {response.model_dump() if hasattr(response, 'model_dump') else response}")
        return "Server error"

    def count_tokens(self, messages: List[Dict[str, Any]], tool_description: Optional[List[Dict[str, Any]]] = None, reasoning_effort: str = 'high', add_generation_prompt: bool = True) -> int:
        prepared_messages = prepare_messages_for_tokenization(messages, self.model_type)
        apply_template_kwargs = {
            "tokenize": True,
            "reasoning_effort": reasoning_effort,
            "add_generation_prompt": add_generation_prompt
        }
        if tool_description is not None: apply_template_kwargs["tools"] = tool_description
        if self.model_type == "glm": apply_template_kwargs["chat_template_kwargs"] = {"enable_thinking": True, "clear_thinking": False}
        tokens = self.tokenizer.apply_chat_template(prepared_messages, **apply_template_kwargs)
        if isinstance(tokens, list): return len(tokens)
        return len(tokens["input_ids"])

    def sanitize_tool_name(self, tool_name: str) -> str:
        return tool_name.split('<|')[0].split('|>')[0].strip()

    def answer_in_content(self, content):
        """Check if answer is in content. Just for statistics."""
        if not content: return False
        if self.task in ["healthbench", "researchrubrics"]: return True
        return "Answer:" in content or "Answer**" in content or "Exact Answer" in content

    async def add_auto_judge(self, result, item):
        prediction = result.get("prediction", "") or ""
        if self.task in ["healthbench"]: item["actual_queried_prompt_messages"] = self.actual_queried_prompt_messages
        evaluator = get_evaluator(self.task)

        try:
            if not prediction:
                judge_result = await evaluator.compute_score(
                    prediction="", item=item, error=True, err_msg="No prediction found"
                )
                result = {"auto_judge": judge_result, **result}
                return result
            judge_result = await evaluator.compute_score(prediction, item)
            result = {"auto_judge": judge_result, **result}
        except Exception as e:
            judge_result = await evaluator.compute_score(
                prediction="", item=item, error=True, err_msg=f"Error: {str(e)}"
            )
            result = {"auto_judge": judge_result, **result}
        return result

    async def _run(self, data: dict, model: str, context_manager=None, **kwargs) -> dict:
        self.model = model
        question = data['item']['question']
        task_instruction = get_task_instruction(self.task)
        TASK_INSTRUCTION = f"\n\n{task_instruction}" if task_instruction else ""
        if self.task == "healthbench":
            messages = [{"role": "system", "content": self.system_prompt + "\n\n" + SYSTEM_PROMPT_DR}] + data['item']["prompt"]
            self.actual_queried_prompt_messages = copy.deepcopy(messages)
        elif self.task == "researchrubrics":
            messages = [{"role": "system", "content": self.system_prompt + "\n\n" + SYSTEM_PROMPT_DR}, {"role": "user", "content": question}]
        else:
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": question+TASK_INSTRUCTION}]
        debug_data = {
            "token_lengths_each_step": [],
            "tool_usage": {},
            "full_messages": [],
        }
        if not debug_data["full_messages"]:
            for msg in messages:
                debug_data["full_messages"].append({**msg, "iteration": 0})

        iteration = 0
        termination = "unknown"
        t_start = asyncio.get_event_loop().time()
        while True:
            iteration += 1
            response = await self.call_server(messages, tool_description=self.tool_description)
            store_token_length(debug_data, iteration, response)

            ### Error
            if isinstance(response, str):
                error_message = {"role": "assistant", "content": response}
                messages.append(error_message)
                debug_data["full_messages"].append({**error_message, "iteration": iteration, "raw_response": response})
                termination = "error"
                break

            # add content, reasoning, tool_calls, and tool_responses
            assistant_message = response.choices[0].message
            content = (assistant_message.get("content") or "").strip()
            message_dict = {"role": "assistant", "content": content}
            reasoning_content = (assistant_message.get("reasoning_content") or "").strip() 
            if reasoning_content: message_dict["reasoning"] = reasoning_content
            # Store raw response for debugging
            raw_response = response.model_dump() if hasattr(response, 'model_dump') else str(response)

            ### Tool Call
            tool_calls = assistant_message.get("tool_calls") or []
            if tool_calls:
                tool_calls_list = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": self.sanitize_tool_name(tc.function.name),
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls[:1] ### Only call one tool at a time
                ]
                message_dict["tool_calls"] = tool_calls_list
                messages.append(message_dict)
                debug_data["full_messages"].append({**message_dict, "iteration": iteration, "raw_response": raw_response})

                for tool_call in tool_calls[:1]: ### Only call one tool at a time
                    tool_name = self.sanitize_tool_name(tool_call.function.name)
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        result = await self.custom_call_tool(tool_name, tool_args)
                        result = json.dumps(result, ensure_ascii=False)
                        debug_data["tool_usage"][tool_name] = debug_data["tool_usage"].get(tool_name, 0) + 1
                    except Exception as e:
                        print(e)
                        result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field'
                    tool_message = {"role": "tool", "tool_call_id": tool_call.id, "name": tool_name, "content": result}
                    messages.append(tool_message)
                    debug_data["full_messages"].append({**tool_message, "iteration": iteration})
            else:
                messages.append(message_dict)
                debug_data["full_messages"].append({**message_dict, "iteration": iteration, "raw_response": raw_response})
                if content or not reasoning_content:
                    termination = "no_tool_call"
                    break

            token_count = self.count_tokens(messages, tool_description=self.tool_description)

            ### Exceed Token Limit or max LLM calls
            if token_count >= self.max_tokens or iteration >= self.max_llm_call_per_run:
                messages[-1]['content'] = TRUNCATED_MESSAGE
                debug_data["full_messages"][-1]["content"] = TRUNCATED_MESSAGE
                messages.append({"role": "user", "content": FINAL_MESSAGE})
                debug_data["full_messages"].append({"role": "user", "content": FINAL_MESSAGE, "iteration": iteration})
                response = await self.call_server(messages, no_tool_calls=True, tool_description=self.tool_description)
                store_token_length(debug_data, iteration, response)
                raw_response = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                if isinstance(response, str): message_dict = {"role": "assistant", "content": response}
                else:
                    assistant_message = response.choices[0].message
                    content = (assistant_message.get("content") or "").strip()
                    message_dict = {"role": "assistant", "content": content}
                    reasoning_content = (assistant_message.get("reasoning_content") or "").strip()
                    if reasoning_content: message_dict["reasoning"] = reasoning_content

                messages.append(message_dict)
                debug_data["full_messages"].append({**message_dict, "iteration": iteration, "raw_response": raw_response})

                termination = "max_exceed"
                break

        time = asyncio.get_event_loop().time() - t_start
        prediction = messages[-1]['content']
        if self.answer_in_content(prediction): termination = "answer"
        result = {
            "question": data["item"].get("question", ""),
            "instance": data["item"],
            "prediction": prediction,
            "termination": termination,
            "time": time,
            "cost": compute_rollout_cost(debug_data, self.model_name, self.task),
            "messages": messages,
            "debug_data": debug_data,
        }
        result = await self.add_auto_judge(result, data["item"])
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in self.tool_map:
            tool_args["params"] = tool_args
            raw_result = await self.tool_map[tool_name].call(tool_args, **kwargs)
            result = raw_result
            return result
        else: return f"Error: Tool {tool_name} not found"
