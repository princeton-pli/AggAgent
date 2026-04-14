# Rollout

This directory implements the ReAct-style agent that generates N independent trajectories for each problem instance. Each trajectory runs a multi-turn reasoning loop with web search and page-visit tools until it produces a final answer.

> **Attribution:** The rollout implementation is adopted from [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch). We thank the authors for their open-source contribution.

## Overview

```
rollout/
├── run_multi_react.py      # Entry point: runs N rollouts over a dataset
├── react_agent.py          # ReAct agent for a single problem instance
├── prompts.py              # System prompts
├── utils.py                # Utility functions
├── tools/
│   ├── serve_search.py     # FastAPI server: search (Serper) + visit (crawl4ai)
│   ├── tool_search.py      # Search tool (client)
│   ├── tool_visit.py       # Visit tool (client)
│   ├── tool_search_bcp.py  # BrowseComp-Plus local FAISS search tool
│   └── tool_get_document_bcp.py  # BrowseComp-Plus local document retrieval tool
└── searchers/              # FAISS / BM25 searcher backends (BrowseComp-Plus)
```

## Prerequisites

### 1. Model server

The rollout agent calls a vLLM-served model via the OpenAI-compatible API. Start your model server before running rollout. A minimal example:

```bash
uv run vllm serve zai-org/GLM-4.7-Flash \
  --served-model-name GLM-4.7-Flash \
  --host 0.0.0.0 --port 6000 \
  --tensor-parallel-size 2 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --max-model-len 128000 \
  --trust-remote-code
```

Model-specific `--tool-call-parser` and `--reasoning-parser` flags:

| Model | `--tool-call-parser` | `--reasoning-parser` | 
|---|---|---|
| GLM-4.7-Flash | `glm47` | `glm45` |
| Qwen3.5-122B | `qwen3_coder` | `qwen3` |
| MiniMax-M2.5 | `minimax_m2` | `minimax_m2` |

### 2. Search server (non-BrowseComp-Plus tasks)

For all tasks except `browsecomp-plus`, rollout uses a local FastAPI server that wraps Google Serper (search) and crawl4ai (page visit). Set `SERPER_KEY_ID` in your `.env`, then start the server:

```bash
cd rollout/tools
uv run python serve_search.py --host 0.0.0.0 --port 8765 --workers 3
```

Or let `scripts/rollout.sh` start it automatically.

Set the server URL via environment variable (default `http://localhost:8765`):

```bash
export SEARCH_SERVER_URL=http://localhost:8765
```

## Running Rollout

### Quick run (all datasets)

```bash
bash scripts/rollout.sh
```

Edit `MODEL`, `DATASETS`, `ROLL_OUT_COUNT`, and `API_BASE` at the top of the script.

### Single dataset

```bash
uv run python rollout/run_multi_react.py \
  --model       GLM-4.7-Flash \
  --dataset     browsecomp \
  --roll_out_count 8 \
  --max_workers 3 \
  --api_base    http://localhost:6000/v1 \
  --output_dir  output/rollout/GLM-4.7-Flash/browsecomp
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `""` | Model name (must match the served model name) |
| `--dataset` | `browsecomp` | Dataset name |
| `--roll_out_count` | `4` | Number of independent rollouts (N) |
| `--max_workers` | `20` | Max concurrent async workers |
| `--api_base` | `http://localhost:8000/v1` | vLLM server base URL |
| `--output_dir` | auto | Output directory (`output/rollout/<model>/<dataset>`) |
| `--max_llm_call_per_run` | `100` | Max LLM calls before forcing a final answer |
| `--max_tokens` | `108000` | Context token limit (triggers forced answer when exceeded) |
| `--max_instances` | `None` | Cap on dataset instances (for debugging) |
| `--total_splits` / `--worker_split` | `1` / `1` | Distributed mode: split dataset across workers |

### Output format

Results are written as individual JSON files:

```
output/rollout/<MODEL>/<DATASET>/
├── iter1/
│   ├── run_20250101T120000000000Z_abc12345.json
│   └── ...
├── iter2/
│   └── ...
└── iter8/
    └── ...
```

Each file contains the full trajectory, prediction, auto-judge score, token usage, and cost:

```json
{
  "question": "...",
  "prediction": "...",
  "messages": [...],
  "auto_judge": {"correctness": "correct", "confidence": 85, ...},
  "cost": {"rollout": 0.0012, "tool": 0.0008},
  "debug_data": {"token_lengths_each_step": [...], "tool_usage": {"search": 4, "visit": 2}}
}
```

## BrowseComp-Plus (Local Retrieval)

`browsecomp-plus` uses a local FAISS index instead of live web search. After downloading the index with `scripts/download_dataset.py --browsecomp-plus`, configure via environment variables before running rollout:

```bash
export SEARCHER_TYPE=faiss
export INDEX_PATH="data/browsecomp-plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl"
export SEARCH_MODEL_NAME=Qwen/Qwen3-Embedding-8B
export SEARCH_NORMALIZE=true
export SEARCH_DATASET_NAME="data/browsecomp-plus/corpus"
export SEARCH_K=5
export SNIPPET_MAX_TOKENS=512
export SNIPPET_TOKENIZER_PATH=Qwen/Qwen3-0.6B
```

These are set automatically when using `scripts/rollout.sh`.

## Supported Datasets

`browsecomp`, `browsecomp-plus`, `hle`, `deepsearchqa`, `healthbench`, `researchrubrics`.