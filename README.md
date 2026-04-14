# Agentic Aggregation for Parallel Scaling of Long-Horizon Agentic Tasks

<p align="center">
  <a href="https://arxiv.org/abs/2604.11753"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b" alt="arXiv"></a>
  <a href="https://pypi.org/project/aggagent/"><img src="https://img.shields.io/pypi/v/aggagent?color=blue" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>
  <img src="https://img.shields.io/badge/Claude_Code_Skill-beta-blueviolet" alt="Claude Code Skill">
</p>

**AggAgent** is a framework for *parallel test-time scaling* of long-horizon agentic tasks. Instead of running a single agent trajectory, you sample **K independent trajectories** and then use AggAgent to synthesize the best final solution. AggAgent inspects tool observations across trajectories, cross-checks reasoning, and resolves conflicts — producing answers that are more reliable than any single run.

> **TL;DR:** Sample K agent trajectories in parallel → aggregate with AggAgent → get a better solution.

<!-- ![Overview](assets/overview.png) -->

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Rollout](#rollout)
- [Aggregation](#aggregation)
- [AggAgent Package](#aggagent-package)
- [Claude Code Skill](#claude-code-skill-beta)
- [Citation](#citation)

---

## Quick Start

The `aggagent` package is available on PyPI. If you already have K agent trajectories and want to aggregate them, this is all you need:

```bash
pip install aggagent
```

```python
from aggagent import AggAgent

# Each trajectory is a list of message dicts (OpenAI message format)
traj_1 = [
    {"role": "user",      "content": "Who won the 1986 FIFA World Cup?"},
    {"role": "assistant", "content": "...", "reasoning": "...", "tool_calls": [...]},
    {"role": "tool",      "tool_call_id": "...", "name": "search", "content": "..."},
    {"role": "assistant", "content": "Argentina won the 1986 FIFA World Cup.", "reasoning": "..."},
]
# ... collect traj_2, traj_3, traj_4 from parallel runs

agent = AggAgent(
    model="gpt-4.1",   # aggregation model; use api_base for local vLLM
    task="browsecomp", # task type 
    # optionally override litellm kwargs (messages and tools are always injected)
    # llm_kwargs={"model": "gemini/gemini-2.0-flash", "api_key": "...", "temperature": 0.7},
)
result = agent.run(
    question="Who won the 1986 FIFA World Cup?",
    trajectories=[traj_1, traj_2, traj_3, traj_4],
)

print(result["solution"])  # self-contained answer string
print(result["reason"])    # meta-reasoning about how trajectories were evaluated
```

See the [AggAgent Package](#aggagent-package) section for the full API.

---

## Installation

### From source (for running experiments)

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/princeton-pli/AggAgent.git
cd AggAgent
uv sync --extra rollout
```

Or with pip:

```bash
git clone https://github.com/princeton-pli/AggAgent.git
cd AggAgent
pip install -e ".[rollout]"
```

### Environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...       # for GPT-based judge / aggregation
GEMINI_API_KEY=...          # for Gemini-based judge / aggregation
SERPER_KEY_ID=...           # for web search (Google Serper)
```

---

## Rollout

The rollout stage runs N independent ReAct agent trajectories over a benchmark dataset. Each trajectory uses web search and page-visit tools to gather evidence and arrive at an answer.

### Download datasets

```bash
# BrowseComp, DeepSearchQA, HealthBench, ResearchRubrics
uv run python scripts/download_dataset.py

# BrowseComp-Plus (also downloads FAISS indexes and corpus)
uv run python scripts/download_dataset.py --browsecomp-plus
```

### Run rollout

First, serve your model with vLLM:

```bash
uv run vllm serve <MODEL_PATH> \
  --served-model-name <MODEL_NAME> \
  --host 0.0.0.0 --port 6000 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47         # adjust per model
```

Then run rollouts:

```bash
# Run all datasets with the settings in scripts/rollout.sh
bash scripts/rollout.sh

# Or run a single dataset directly
uv run python rollout/run_multi_react.py \
  --model GLM-4.7-Flash \
  --dataset browsecomp \
  --roll_out_count 8 \
  --max_workers 3 \
  --api_base http://localhost:6000/v1 \
  --output_dir output/rollout/GLM-4.7-Flash/browsecomp
```

Results are written as individual JSON files under `output/rollout/<MODEL>/<DATASET>/iter{k}/`.

**Supported datasets:** `browsecomp`, `browsecomp-plus`, `hle`, `deepsearchqa`, `healthbench`, `researchrubrics`

For detailed instructions — model-specific flags, distributed (multi-worker) splits, BrowseComp+ local retrieval setup — see [rollout/README.md](rollout/README.md).

---

## Aggregation

Given a directory of rollout results, `aggregation/aggregate.py` computes aggregation metrics across a range of strategies and k values.

### Heuristic strategies (no LLM calls)

```bash
uv run python aggregation/aggregate.py \
  --strategy heuristic \
  --task browsecomp \
  output/rollout/GLM-4.7-Flash/browsecomp
```

### LLM-based strategies

```bash
# SolAgg: integrate raw predictions from k trajectories
uv run python aggregation/aggregate.py \
  --strategy solagg \
  --model GLM-4.7-Flash \
  --api_base http://localhost:6000/v1 \
  --task browsecomp \
  --k 4 \
  output/rollout/GLM-4.7-Flash/browsecomp

# SummAgg: summarize each trajectory first, then integrate
uv run python aggregation/aggregate.py \
  --strategy summagg \
  --model GLM-4.7-Flash \
  --api_base http://localhost:6000/v1 \
  --task browsecomp \
  --k 4 \
  output/rollout/GLM-4.7-Flash/browsecomp
```

### AggAgent

```bash
# Using the script
bash scripts/aggregation.sh

# Or directly
uv run python aggregation/aggregate.py \
  --strategy aggagent \
  --model GLM-4.7-Flash \
  --api_base http://localhost:6000/v1 \
  --task browsecomp \
  --k 4 \
  output/rollout/GLM-4.7-Flash/browsecomp
```

Logs are written to `output/aggregation/<MODEL>/<DIR_TAG>/aggagent_logs_k{k}.jsonl` and a summary to `aggagent_stats_k{k}.json`.

### Strategy reference

| Strategy | Type | Description |
|---|---|---|
| `pass` | Heuristic | Pass@k upper bound — correct if any trajectory is correct |
| `mv` | Heuristic | Majority voting over extracted answers |
| `wmv` | Heuristic | Confidence-weighted majority voting |
| `bon` | Heuristic | Best of N — pick the trajectory with highest confidence |
| `fewtool` | Heuristic | Pick the trajectory that used the fewest tool calls |
| `solagg` | LLM-based | Feed k raw predictions to an LLM to integrate |
| `summagg` | LLM-based | Summarize each trajectory into a report, then integrate |
| `aggagent` | LLM-based | Agentic aggregation — inspect tool evidence, cross-check, synthesize |

### All strategies at once

```bash
uv run python aggregation/aggregate.py \
  --strategy all \
  --model GLM-4.7-Flash \
  --api_base http://localhost:6000/v1 \
  --task browsecomp \
  output/rollout/GLM-4.7-Flash/browsecomp
```

---

## AggAgent Package

The `aggagent` Python package exposes a single class: `AggAgent`.

### Installation

```bash
pip install aggagent
```

### `AggAgent(model, api_base, task, max_context_tokens, llm_kwargs)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `""` | Model name. Use the model's served name for local vLLM, or `gpt-4.1` / `gemini-...` for API models. |
| `api_base` | `str \| None` | `None` | Base URL for a local vLLM server (e.g. `http://localhost:6000/v1`). Set `None` for OpenAI/Gemini. |
| `task` | `str` | `""` | Task type. Controls the output format of the `finish` tool. Use one of the supported task names or `""` for generic short-answer tasks. |
| `max_context_tokens` | `int` | `102400` | Approximate token budget. When exceeded, the agent is forced to call `finish` immediately. |
| `llm_kwargs` | `dict \| None` | `None` | If provided, passed directly to `litellm.completion` (bypassing built-in defaults). `messages` and `tools` are always injected. Useful for setting `model`, `api_key`, `api_base`, `temperature`, `top_p`, `max_tokens`, etc. |

**Supported task types:** `browsecomp`, `browsecomp-plus`, `hle`, `deepsearchqa`, `healthbench`, `researchrubrics`
- Short-answer tasks (`browsecomp`, `hle`, `deepsearchqa`) → solution format: `<explanation>...</explanation><answer>...</answer>`
- Long-form tasks (`healthbench`, `researchrubrics`) → solution format: a full synthesized report with inline citations

### `agent.run(question, trajectories) → dict`

```python
result = agent.run(
    question="...",
    trajectories=[traj_1, traj_2, ...],
)
```

| Parameter | Type | Description |
|---|---|---|
| `question` | `str` | The task or question being answered. |
| `trajectories` | `list[list[dict]]` | N trajectories. Each trajectory is a list of message dicts in OpenAI message format (`role`, `content`, optionally `tool_calls`, `reasoning_content`). |

**Returns** on success:
```python
{"solution": str, "reason": str}
```
- `solution`: a self-contained answer string (does not reference trajectories or agents).
- `reason`: the agent's meta-reasoning — how it evaluated and reconciled trajectories.

**Returns** on failure:
```python
{"solution": None, "reason": None, "error": str}
```

### Trajectory format

Each trajectory is a list of messages with standard OpenAI roles. Tool calls and tool responses are supported:

```python
trajectory = [
    {"role": "system",    "content": "You are a research assistant..."},
    {"role": "user",      "content": "What year did ..."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
            "name": "search", "arguments": '{"query": "..."}'
        }}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "name": "search",
     "content": "Search results: ..."},
    {"role": "assistant", "content": "Based on the search results, the answer is ..."},
]
```

Reasoning/thinking tokens can be included under the `reasoning_content` key in assistant messages.

### How AggAgent works

AggAgent operates as a tool-calling agent with four internal tools:

| Tool | Description |
|---|---|
| `get_solution` | Retrieve the final message from one or all trajectories |
| `search_trajectory` | Search for a keyword/phrase within a trajectory (ROUGE-L ranked) |
| `get_segment` | Read a contiguous range of steps from a trajectory in full |
| `finish` | Submit the final synthesized answer |

The agent is instructed to: survey trajectory metadata → retrieve final solutions → verify key claims against raw tool observations (`search_trajectory` + `get_segment`) → cross-check reasoning → call `finish`.

---

## Claude Code Skill (Beta)

AggAgent is also available as a [Claude Code](https://claude.ai/code) skill. The skill lives in [`.claude/skills/aggagent/`](.claude/skills/aggagent/).

The skill expects a single flat directory containing one JSON file per trajectory, all for the same question. Use `scripts/collect_trajs.py` to assemble this from a rollout output directory:

```bash
# Collect trajectories for a question matching "MMORPG"
python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa "MMORPG"

# Then aggregate
/aggagent trajs_mmorpg/
```

The skill surveys all final solutions, verifies key claims against raw tool observations, and synthesizes a final answer — the same aggregation logic as the Python package, but interactive inside Claude Code.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{lee2026agentic,
  title={Agentic Aggregation for Parallel Scaling of Long-Horizon Agentic Tasks},
  author={Yoonsang Lee and Howard Yen and Xi Ye and Danqi Chen},
  journal={arXiv preprint arXiv:2604.11753},
  year={2026}
}
```

---

<p align="center">
Princeton Language and Intelligence (PLI) · Apache 2.0 License
</p>
