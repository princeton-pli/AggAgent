import argparse
import json
import os
import asyncio
from tqdm import tqdm
from datetime import datetime
from react_agent import MultiTurnReactAgent
from evaluation import get_evaluator
import math
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Model path")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for results")
    parser.add_argument("--dataset", type=str, default="browsecomp", choices=["browsecomp", "browsecomp-plus", "hle", "deepsearchqa", "healthbench", "researchrubrics"], help="Dataset name")
    parser.add_argument("--roll_out_count", type=int, default=4, help="Number of rollouts per question")
    parser.add_argument("--max_workers", type=int, default=20, help="Maximum number of concurrent workers")
    parser.add_argument("--max_instances", type=int, default=None, help="Optional cap on number of instances to process after split")
    parser.add_argument("--total_splits", type=int, default=1, help="Total number of worker splits")
    parser.add_argument("--worker_split", type=int, default=1, help="Index of this worker's split")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="API base URL for serving local model")
    parser.add_argument("--max_llm_call_per_run", type=int, default=100, help="Max LLM calls per run")
    parser.add_argument("--max_tokens", type=int, default=108000, help="Context token limit (triggers final answer when exceeded)")
    args = parser.parse_args()

    model = args.model
    model_name = os.path.basename(model.rstrip('/'))
    dataset = args.dataset
    dataset_dir = f"data/{dataset}.jsonl"
    output_dir = args.output_dir or f"output/rollout/{model_name}/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    roll_out_count = args.roll_out_count
    max_instances = args.max_instances
    total_splits = args.total_splits
    worker_split = args.worker_split

    # Validate worker_split
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    print(f"Model name: {model_name}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    # Read dataset
    try:
        if dataset_dir.endswith(".json"):
            with open(dataset_dir, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif dataset_dir.endswith(".jsonl"):
            with open(dataset_dir, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {dataset_dir}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {dataset_dir}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)

    # Split the dataset
    items = items[start_idx:end_idx]

    # Optional cap after split selection
    if max_instances is not None:
        if max_instances < 0:
            print(f"Error: max_instances ({max_instances}) must be >= 0")
            exit(1)
        items = items[:max_instances]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")
    if max_instances is not None:
        print(f"Max instances cap applied: {max_instances}")

    # Create subdirectories per rollout
    print(f"Creating rollout directories...")
    rollout_dirs = {}
    for rollout_idx in range(1, roll_out_count + 1):
        if total_splits > 1:
            rollout_dir = os.path.join(output_dir, f"iter{rollout_idx}_split{worker_split}of{total_splits}")
        else:
            rollout_dir = os.path.join(output_dir, f"iter{rollout_idx}")
        os.makedirs(rollout_dir, exist_ok=True)
        print(f"Rollout directory {rollout_idx}: {rollout_dir}")
        rollout_dirs[rollout_idx] = rollout_dir

    # Check for already processed queries by scanning subdirectories
    processed_queries_per_rollout = {}
    for rollout_idx in range(1, roll_out_count + 1):
        rollout_dir = rollout_dirs[rollout_idx]
        processed_queries = set()
        if os.path.exists(rollout_dir):
            # Scan all JSON files in the rollout directory
            for filename in os.listdir(rollout_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(rollout_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        # Skip invalid files
                        pass
        processed_queries_per_rollout[rollout_idx] = processed_queries

    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for item in items:
            if dataset == "browsecomp" or dataset == "browsecomp-plus":
                question = item.get("problem", "").strip()
                item["question"] = question
            elif dataset == "deepsearchqa":
                question = item.get("problem", "").strip()
                item["question"] = question
            elif dataset == "healthbench":
                question = item.get("prompt_id", "").strip()
                item["question"] = question
            elif dataset == "researchrubrics":
                question = item.get("prompt", "").strip()
                item["question"] = question
            question = item.get("question", "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                tasks_to_run_all.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        llm_cfg = {
            'model': model,
            'model_name': model_name,
            'model_type': model,
            'generate_cfg': {
                'model': f"hosted_vllm/{model_name}",
                'max_tokens': 10000,
                "parallel_tool_calls": False,
                "temperature": 1.0,
                "top_p": 0.95,
                "api_base": args.api_base,
                "api_key": "EMPTY",
            },
        }

        if "glm" in model.lower():
            llm_cfg["model_type"] = "glm"
            llm_cfg["generate_cfg"]["extra_body"] = { "chat_template_kwargs": { "enable_thinking": True, "clear_thinking": False } }
        elif "qwen" in model.lower():
            llm_cfg["model_type"] = "qwen"
            llm_cfg["generate_cfg"]["extra_body"] = { "chat_template_kwargs": { "enable_thinking": True } }
        elif "minimax" in model.lower():
            llm_cfg["model_type"] = "minimax"
            llm_cfg["generate_cfg"]["extra_body"] = {"reasoning_split": True}

        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            function_list=None,
            task=dataset,
            max_llm_call_per_run=args.max_llm_call_per_run,
            max_tokens=args.max_tokens,
        )

        empty_debug_data = {
            "token_lengths_each_step": [],
            "full_messages": [],
            "tool_usage": {},
        }

        # Metadata to include in all output files
        run_metadata = {
            "model": model,
            "output_dir": output_dir,
            "dataset": dataset,
            "roll_out_count": roll_out_count,
            "max_workers": args.max_workers,
            "max_instances": max_instances,
            "total_splits": total_splits,
            "worker_split": worker_split,
            "api_base": args.api_base,
            "max_llm_call_per_run": args.max_llm_call_per_run,
            "max_tokens": args.max_tokens,
        }

        def write_json_file(filepath, data):
            """Write JSON data to file synchronously"""
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        def get_score(data: dict) -> float:
            auto_judge = data.get("auto_judge", {})
            if dataset == "deepsearchqa":
                return 1.0 if auto_judge.get("all_correct") == True else 0.0
            elif dataset in ("healthbench", "researchrubrics"):
                return float(auto_judge.get("metrics", {}).get("overall_score", 0.0))
            else:
                return 1.0 if auto_judge.get("correctness") == "correct" else 0.0

        async def process_tasks():
            semaphore = asyncio.Semaphore(args.max_workers)
            rollout_scores = {i: [] for i in range(1, roll_out_count + 1)}
            
            async def run_task_with_semaphore(task):
                async with semaphore:
                    t_start = asyncio.get_event_loop().time()
                    try:
                        # Use asyncio.wait_for for timeout handling
                        result = await asyncio.wait_for(
                            test_agent._run(task, model),
                            timeout=3600.0  # 60 minutes timeout
                        )
                        return task, result, None, asyncio.get_event_loop().time() - t_start
                    except asyncio.TimeoutError:
                        return task, None, "timeout", asyncio.get_event_loop().time() - t_start
                    except Exception as exc:
                        return task, None, exc, asyncio.get_event_loop().time() - t_start

            async def handle_single_task(task_info, result, error, time):
                """Handle a single completed task: write result to file."""
                rollout_idx = task_info["rollout_idx"]
                rollout_dir = rollout_dirs[rollout_idx]
                
                # Generate unique timestamp-based filename with UUID suffix to prevent collisions
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID for uniqueness
                output_file = os.path.join(rollout_dir, f"run_{ts}_{unique_id}.json")
                
                if error == "timeout":
                    print(f'Timeout (>3600s): "{task_info["item"]["question"]}" (Rollout {rollout_idx})')
                    evaluator = get_evaluator(dataset)
                    error_result = {
                        "metadata": run_metadata,
                        "question": task_info["item"].get("question", ""),
                        "instance": task_info["item"],
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": "Timeout (>3600s)",
                        "messages": [],
                        "prediction": "[Failed]",
                        "time": time,
                        "debug_data": empty_debug_data,
                        "cost": {"rollout": 0.0, "tool": 0.0},
                        "auto_judge": await evaluator.compute_score(prediction="", item=task_info["item"], error=True, err_msg="Timeout (>3600s)"),
                        "termination": "error"
                    }
                    await asyncio.to_thread(write_json_file, output_file, error_result)
                    rollout_scores[rollout_idx].append(0.0)
                elif error is not None:
                    print(f'Task for question "{task_info["item"]["question"]}" (Rollout {rollout_idx}) generated an exception: {error}')
                    evaluator = get_evaluator(dataset)
                    error_result = {
                        "metadata": run_metadata,
                        "question": task_info["item"].get("question", ""),
                        "instance": task_info["item"],
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": f"Future resolution failed: {error}",
                        "messages": [],
                        "prediction": "[Failed]",
                        "time": time,
                        "debug_data": empty_debug_data,
                        "cost": {"rollout": 0.0, "tool": 0.0},
                        "auto_judge": await evaluator.compute_score(prediction="", item=task_info["item"], error=True, err_msg=f"Future resolution failed: {error}"),
                        "termination": "error"
                    }
                    await asyncio.to_thread(write_json_file, output_file, error_result)
                    rollout_scores[rollout_idx].append(0.0)
                else:
                    result_with_metadata = {"metadata": run_metadata, **result}
                    await asyncio.to_thread(write_json_file, output_file, result_with_metadata)
                    rollout_scores[rollout_idx].append(get_score(result))

            # Group tasks by rollout_idx
            tasks_by_rollout = {}
            for task in tasks_to_run_all:
                rollout_idx = task["rollout_idx"]
                if rollout_idx not in tasks_by_rollout:
                    tasks_by_rollout[rollout_idx] = []
                tasks_by_rollout[rollout_idx].append(task)
            
            # Process each rollout sequentially
            for rollout_idx in sorted(tasks_by_rollout.keys()):
                rollout_tasks = tasks_by_rollout[rollout_idx]
                print(f"\n=== Starting Rollout {rollout_idx} ({len(rollout_tasks)} tasks) ===")
                
                # Create tasks for this rollout
                tasks = [run_task_with_semaphore(task) for task in rollout_tasks]
                
                # Process tasks as they complete for this rollout
                for coro in tqdm(asyncio.as_completed(tasks), total=len(rollout_tasks), desc=f"Processing Rollout {rollout_idx}"):
                    task_info, result, error, time = await coro
                    await handle_single_task(task_info, result, error, time)
                
                scores = rollout_scores[rollout_idx]
                mean = sum(scores) / len(scores) if scores else 0.0
                print(f"=== Completed Rollout {rollout_idx} | Score: {mean * 100:.2f}% ({len(scores)} items) ===\n")

        # Run the async processing
        asyncio.run(process_tasks())
        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")
