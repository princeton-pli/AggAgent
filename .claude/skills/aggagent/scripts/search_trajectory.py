#!/usr/bin/env python3
"""
search_trajectory: ROUGE-L search through a trajectory's steps.

Usage:
  python search_trajectory.py <trajectories_dir> <trajectory_id> "<query>" [--role tool|assistant] [--k N]

trajectory_id is 1-based.
--role tool      restricts search to actual tool responses (ground truth observations)
--role assistant restricts search to agent reasoning steps
--k N            number of results to return (default 5, max 10)
"""
import argparse
import json
import os
import re
import sys

try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _scorer = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    def rouge_l_recall(query, text):
        if not query or not text:
            return 0.0
        return _scorer.score(query, text)['rougeL'].recall

except ImportError:
    import difflib

    def rouge_l_recall(query, text):
        """Fallback when rouge_score is not installed."""
        if not query or not text:
            return 0.0
        return difflib.SequenceMatcher(None, query.lower(), text.lower()).ratio()


def get_content(message, key="content"):
    """Extract text content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            text = value[0].get("text") or ""
            if key == "content":
                name = message.get("name")
                recipient = message.get("recipient")
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    return ""


def truncate_text(text, max_words=150):
    """Truncate text to first N words — mirrors tools.truncate_text."""
    if not text:
        return ""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == max_words:
            return text[:m.end()] + '\n[... truncated]'
    return text


def load_trajectories(traj_dir):
    files = sorted([
        os.path.join(traj_dir, f)
        for f in os.listdir(traj_dir)
        if f.endswith(".json")
    ])
    if not files:
        print(f"No JSON files found in {traj_dir}", file=sys.stderr)
        sys.exit(1)
    trajectories = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        trajectories.append(data.get("messages", []))
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="ROUGE-L search through trajectory steps")
    parser.add_argument("traj_dir", help="Directory containing trajectory JSON files")
    parser.add_argument("trajectory_id", type=int, help="Trajectory index (1-based)")
    parser.add_argument("query", help="Search term or phrase")
    parser.add_argument("--role", choices=["tool", "assistant"], default=None,
                        help="Filter to 'tool' (environment observations) or 'assistant' steps only")
    parser.add_argument("--k", type=int, default=5, help="Max matches to return (default 5, max 10)")
    args = parser.parse_args()

    trajectories = load_trajectories(args.traj_dir)
    n = len(trajectories)

    if args.trajectory_id < 1 or args.trajectory_id > n:
        print(f"Error: trajectory_id must be 1-{n}")
        sys.exit(1)

    traj = trajectories[args.trajectory_id - 1]
    max_results = min(args.k, 10)

    # Score each step — mirrors SearchTrajectoriesTool._score_traj
    scored = []
    for step_idx, step in enumerate(traj):
        if args.role is not None and step.get("role", "") != args.role:
            continue
        content = get_content(step) or ""
        reasoning = get_content(step, "reasoning_content") or get_content(step, "reasoning") or ""
        tool_calls_str = json.dumps(step.get("tool_calls"), ensure_ascii=False) if step.get("tool_calls") else ""

        score = max(
            rouge_l_recall(args.query, content),
            rouge_l_recall(args.query, reasoning),
            rouge_l_recall(args.query, tool_calls_str),
        )
        if score > 0:
            scored.append((score, step_idx, step))

    scored.sort(key=lambda x: -x[0])

    if not scored:
        role_msg = f" (role={args.role})" if args.role else ""
        print(f"No matches found for '{args.query}'{role_msg} in trajectory {args.trajectory_id}")
        return

    n_shown = min(max_results, len(scored))
    print(f"Top {n_shown} matches for '{args.query}' in trajectory {args.trajectory_id}"
          + (f" [role={args.role}]" if args.role else "") + ":")
    print()

    for score, step_idx, step in scored[:max_results]:
        role = step.get("role", "")
        content = get_content(step)
        reasoning = get_content(step, "reasoning_content") or get_content(step, "reasoning")
        tool_calls = step.get("tool_calls")

        print(f"  Step {step_idx + 1} [{role}]  score={score:.3f}")
        if reasoning:
            print(f"  [reasoning] {truncate_text(reasoning)}")
        if tool_calls:
            print(f"  [tool_calls] {json.dumps(tool_calls[:1], ensure_ascii=False)}")
        if content:
            print(f"  {truncate_text(content)}")
        print()


if __name__ == "__main__":
    main()
