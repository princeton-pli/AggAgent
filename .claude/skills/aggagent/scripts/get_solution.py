#!/usr/bin/env python3
"""
get_solution: Retrieve final content from trajectory JSON files.

Usage:
  python get_solution.py <trajectories_dir>              # all trajectories + metadata
  python get_solution.py <trajectories_dir> <traj_id>   # single trajectory (1-based)
"""
import json
import os
import re
import sys


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


def count_tokens_approx(messages, chars_per_token=4.0):
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


def format_metadata(trajectories):
    """Format trajectory metadata — mirrors tools.format_metadata."""
    blocks = []
    for i, traj in enumerate(trajectories):
        num_steps = len(traj)
        approx_tokens = count_tokens_approx(traj)

        tool_counts: dict = {}
        for msg in traj:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                func = tc.get("function", {})
                if isinstance(func, dict):
                    name = func.get("name")
                    if name:
                        tool_counts[name] = tool_counts.get(name, 0) + 1

        tool_str = ", ".join(f"{n}×{c}" for n, c in sorted(tool_counts.items())) if tool_counts else "none"
        blocks.append(f"Trajectory {i + 1}: {num_steps} steps, ~{approx_tokens:,} tokens | tools: {tool_str}")

    return "\n".join(blocks)


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
    questions = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        trajectories.append(data.get("messages", []))
        questions.append(data.get("question", ""))
    return trajectories, questions


def main():
    if len(sys.argv) < 2:
        print("Usage: get_solution.py <trajectories_dir> [trajectory_id]")
        sys.exit(1)

    traj_dir = sys.argv[1]
    trajectory_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    trajectories, questions = load_trajectories(traj_dir)
    n = len(trajectories)

    if trajectory_id is None:
        # Print metadata header + all solutions
        question = next((q for q in questions if q), "")
        print(f"TRAJECTORY METADATA ({n} trajectories):")
        if question:
            print(f"Question: {question}")
        print()
        print(format_metadata(trajectories))
        print()
        print("=" * 60)
        print("SOLUTIONS")
        print("=" * 60)
        for i, traj in enumerate(trajectories):
            content = get_content(traj[-1]) if traj else ""
            print(f"\n--- Trajectory {i + 1} ---")
            if len(content) > 2000:
                print(content[:2000] + "\n[... truncated]")
            else:
                print(content)
    else:
        if trajectory_id < 1 or trajectory_id > n:
            print(f"Error: trajectory_id must be 1-{n}")
            sys.exit(1)
        traj = trajectories[trajectory_id - 1]
        content = get_content(traj[-1]) if traj else ""
        print(f"Trajectory {trajectory_id} (total steps: {len(traj)}):")
        print(content)


if __name__ == "__main__":
    main()
