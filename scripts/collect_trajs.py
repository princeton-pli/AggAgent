#!/usr/bin/env python3
"""
collect_trajs.py: Collect N trajectory files for a single problem (or all problems) into flat directories.

The rollout pipeline stores results as:
  <rollout_dir>/
    iter1/  run_<ts>_<hash>.json   (one file per problem)
    iter2/  run_<ts>_<hash>.json
    ...

Usage:
  # List all available questions
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa --list

  # Collect trajectories for a question matching "MMORPG"
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa "MMORPG"

  # Collect ALL problems (one subdirectory per problem)
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa --all

  # Specify output directory explicitly
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa "MMORPG" --out /tmp/mmorpg_trajs
  python scripts/collect_trajs.py output/rollout/GLM-4.7-Flash/deepsearchqa --all --out /tmp/all_trajs
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def find_iter_dirs(rollout_dir: Path) -> list[Path]:
    """Return sorted iter* leaf directories under rollout_dir."""
    dirs = []
    for entry in sorted(rollout_dir.iterdir()):
        if entry.is_dir():
            dirs.append(entry)
    if not dirs:
        print(f"No subdirectories found in {rollout_dir}", file=sys.stderr)
        sys.exit(1)
    return dirs


def load_question(filepath: Path) -> str:
    with open(filepath) as f:
        data = json.load(f)
    return data.get("question") or data.get("instance", {}).get("id") or ""


def index_dir(dir_path: Path) -> dict[str, Path]:
    """Return {question: filepath} for all JSON files in a directory."""
    index = {}
    for f in sorted(dir_path.glob("*.json")):
        q = load_question(f)
        if q:
            index[q] = f
    return index


def find_matching_question(index: dict[str, Path], query: str) -> str | None:
    """Find the question that contains query (case-insensitive). Returns the full question."""
    query_lower = query.lower()
    matches = [q for q in index if query_lower in q.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous query '{query}' matches {len(matches)} questions:", file=sys.stderr)
        for m in matches:
            print(f"  {m[:100]}", file=sys.stderr)
        print("Refine your query to match exactly one.", file=sys.stderr)
        sys.exit(1)
    return None


def collect_one(
    question: str,
    iter_dirs: list[Path],
    out_dir: Path,
) -> int:
    """Copy one file per iter for `question` into out_dir. Returns number of files copied."""
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    n_copied = 0
    for iter_dir in iter_dirs:
        index = index_dir(iter_dir)
        matched = find_matching_question(index, question)
        if matched is None:
            missing.append(iter_dir.name)
            continue
        src = index[matched]
        dst = out_dir / f"{iter_dir.name}_{src.name}"
        shutil.copy2(src, dst)
        n_copied += 1
    if missing:
        print(f"  Warning: not found in {missing}", file=sys.stderr)
    return n_copied


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-problem trajectories from iter* rollout directories."
    )
    parser.add_argument("rollout_dir", help="Parent rollout directory (contains iter1/, iter2/, ...)")
    parser.add_argument("query", nargs="?", help="Question substring to match (case-insensitive)")
    parser.add_argument("--out", help="Output directory (leaf dir for single query; parent dir for --all)")
    parser.add_argument("--list", action="store_true", help="List all available questions and exit")
    parser.add_argument("--all", action="store_true", help="Collect all problems (one subdir per problem)")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.is_dir():
        print(f"Error: {rollout_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    iter_dirs = find_iter_dirs(rollout_dir)
    print(f"Found {len(iter_dirs)} iter directories: {[d.name for d in iter_dirs]}")

    first_index = index_dir(iter_dirs[0])

    if args.list:
        print(f"\nAvailable questions ({len(first_index)} in {iter_dirs[0].name}):")
        for i, q in enumerate(sorted(first_index), 1):
            print(f"  [{i:3d}] {q[:120]}")
        return

    # --all: collect every problem into its own subdirectory
    if args.all:
        questions = sorted(first_index.keys())
        base_out = Path(args.out) if args.out else rollout_dir.parent / "trajs_all"
        print(f"\nCollecting {len(questions)} problems → {base_out}/")
        total = 0
        for i, question in enumerate(questions):
            slug = question.lower().replace(" ", "_")[:40]
            slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)
            prob_dir = base_out / f"problem_{i:03d}_{slug}"
            n = collect_one(question, iter_dirs, prob_dir)
            print(f"  [{i:3d}] {question[:80]} → {prob_dir.name}/ ({n} trajs)")
            total += n
        print(f"\nDone. {len(questions)} problems, {total} files → {base_out}")
        print(f"\nRun aggregation on one problem with:")
        print(f"  /aggagent {base_out}/problem_000_...")
        return

    # Single query mode
    if not args.query:
        print("Error: provide a query string, --list, or --all", file=sys.stderr)
        sys.exit(1)

    matched_question = find_matching_question(first_index, args.query)
    if matched_question is None:
        print(f"No question matching '{args.query}' found in {iter_dirs[0].name}", file=sys.stderr)
        print("Use --list to see all available questions.", file=sys.stderr)
        sys.exit(1)

    print(f"\nMatched: {matched_question[:120]}")

    slug = args.query.lower().replace(" ", "_")[:30]
    out_dir = Path(args.out) if args.out else rollout_dir.parent / f"trajs_{slug}"

    n = collect_one(matched_question, iter_dirs, out_dir)
    if n == 0:
        print("Error: no files collected.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCollected {n} trajectories → {out_dir}")
    print(f"\nRun aggregation with:")
    print(f"  /aggagent {out_dir}")


if __name__ == "__main__":
    main()
