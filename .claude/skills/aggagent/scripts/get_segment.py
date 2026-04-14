#!/usr/bin/env python3
"""
get_segment: Read a contiguous range of steps from a trajectory (max 5).

Usage:
  python get_segment.py <trajectories_dir> <trajectory_id> <start_step> <end_step>

trajectory_id is 1-based. start_step and end_step are 1-based step indices.
end_step - start_step must be <= 4 (enforced automatically).
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


def truncate_text(text, max_words=600):
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
    if len(sys.argv) < 5:
        print("Usage: get_segment.py <trajectories_dir> <trajectory_id> <start_step> <end_step>")
        sys.exit(1)

    traj_dir = sys.argv[1]
    trajectory_id = int(sys.argv[2])
    start_step = int(sys.argv[3])
    end_step = int(sys.argv[4])

    trajectories = load_trajectories(traj_dir)
    n = len(trajectories)

    if trajectory_id < 1 or trajectory_id > n:
        print(f"Error: trajectory_id must be 1-{n}")
        sys.exit(1)

    traj = trajectories[trajectory_id - 1]
    n_steps = len(traj)

    # Clamp and enforce max-5 window — mirrors GetSegmentTool.call
    start_step = max(1, min(start_step, n_steps))
    end_step = max(1, min(end_step, n_steps))
    if start_step > end_step:
        start_step = end_step
    if end_step - start_step > 4:
        end_step = start_step + 4

    print(f"Trajectory {trajectory_id}, steps {start_step}–{end_step} (of {n_steps} total):")
    print()

    for step_idx in range(start_step - 1, end_step):
        step = traj[step_idx]
        role = step.get("role", "")
        content = get_content(step)
        reasoning = get_content(step, "reasoning_content") or get_content(step, "reasoning")
        tool_calls = step.get("tool_calls")

        print(f"--- Step {step_idx + 1} [{role}] ---")
        if reasoning:
            print(f"[reasoning]\n{truncate_text(reasoning)}")
        if tool_calls:
            print(f"[tool_calls] {json.dumps(tool_calls, ensure_ascii=False)}")
        if content:
            print(truncate_text(content))
        print()


if __name__ == "__main__":
    main()
