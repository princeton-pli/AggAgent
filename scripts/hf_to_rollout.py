#!/usr/bin/env python3
"""Convert published AggAgent rollout parquets into the on-disk layout that
``aggregation/aggregate.py`` and ``scripts/collect_trajs.py`` expect.

The parquets we publish on Hugging Face (and ship in the BrowseComp tar) are
flat: one row per rollout, with the structured per-trajectory fields
(``messages``, ``instance``, ``cost``, ...) JSON-encoded as strings. The
codebase, however, expects one JSON file per (question, iter) pair under

    output/rollout/<MODEL>/<DATASET>/
        iter1/  <question-id>.json
        iter2/  <question-id>.json
        ...
        iterN/  <question-id>.json

This script materializes that layout from a single parquet.

Usage (local parquet)::

    python scripts/hf_to_rollout.py path/to/Qwen3.5-122B-A10B.parquet \
        --out output/rollout/Qwen3.5-122B-A10B/deepsearchqa

Usage (download from Hugging Face)::

    python scripts/hf_to_rollout.py \
        --repo yoonsanglee/deepsearchqa-react \
        --model Qwen3.5-122B-A10B \
        --out output/rollout/Qwen3.5-122B-A10B/deepsearchqa
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

JSON_FIELDS = ("metadata", "instance", "cost", "messages", "debug_data", "auto_judge")


def load_rows(parquet_path: Path | None, repo: str | None, model: str | None):
    if parquet_path is not None:
        import pyarrow.parquet as pq
        return pq.read_table(parquet_path).to_pylist()
    from datasets import load_dataset
    ds = load_dataset(repo, name=model, split="train")
    return [dict(row) for row in ds]


def parse_record(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in JSON_FIELDS and isinstance(v, str):
            try:
                out[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        out[k] = v
    return out


def question_id(question: str, instance: object) -> str:
    """Stable filename id. Prefer instance.id / instance.prompt_id when present
    (HealthBench, etc.); otherwise hash the question text."""
    if isinstance(instance, dict):
        for key in ("id", "prompt_id", "qid"):
            v = instance.get(key)
            if isinstance(v, str) and v:
                return v
    return "q_" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("parquet", nargs="?", type=Path, help="Path to a local parquet file")
    src.add_argument("--repo", help="Hugging Face dataset repo (e.g. yoonsanglee/deepsearchqa-react)")
    ap.add_argument("--model", help="Config name when using --repo (e.g. Qwen3.5-122B-A10B)")
    ap.add_argument("--out", required=True, type=Path, help="Output rollout dir")
    args = ap.parse_args()

    if args.repo and not args.model:
        ap.error("--model is required with --repo")

    rows = load_rows(args.parquet, args.repo, args.model)
    print(f"Loaded {len(rows)} rows")

    # Group rollouts by question
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rec = parse_record(row)
        groups[rec["question"]].append(rec)

    n_iters = max(len(rs) for rs in groups.values())
    print(f"{len(groups)} unique questions, up to {n_iters} rollouts per question")

    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(n_iters):
        (args.out / f"iter{i + 1}").mkdir(exist_ok=True)

    written = 0
    for q, rollouts in groups.items():
        qid = question_id(q, rollouts[0].get("instance"))
        for i, rec in enumerate(rollouts):
            target = args.out / f"iter{i + 1}" / f"{qid}.json"
            with target.open("w") as f:
                json.dump(rec, f, ensure_ascii=False)
            written += 1

    print(f"Wrote {written} JSON files into {args.out}/iter1..iter{n_iters}")


if __name__ == "__main__":
    main()
