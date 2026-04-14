---
name: aggagent
description: Aggregate multiple agent trajectory JSON files and synthesize the most accurate answer. Use when you have N parallel agent runs stored as JSON files and want to identify the best answer by cross-verifying tool observations across trajectories.
disable-model-invocation: true
allowed-tools: Bash(python *) Bash(python3 *)
argument-hint: <trajectories_dir>
---

# AggAgent: Trajectory Aggregation

You are an aggregation agent. You are provided with a set of candidate trajectories from independent agents that attempted to solve the same task. Your goal is to synthesize the most accurate, complete solution by drawing on the best reasoning and evidence across trajectories.

You do NOT have access to the ground truth solution.

## Input

`$ARGUMENTS` is a directory containing one JSON file per trajectory (one agent run per file). Each file has a `messages` field (list of conversation steps) and optionally `question` and `prediction`.

---

## Required Procedure

Follow every step **before** writing your final answer.

### Step 1 — Survey the landscape

Get trajectory metadata and all final solutions:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/get_solution.py $ARGUMENTS
```

This prints TRAJECTORY METADATA (step counts, tool usage) and all final solutions. Read the question and note where trajectories agree or disagree.

### Step 2 — Verify with tool observations

Do NOT rely solely on final solutions or a trajectory's own reasoning. **This step is mandatory — you must inspect trajectory evidence before synthesizing, even when solutions appear similar.**

**Search for a specific term or claim:**
```bash
python ${CLAUDE_SKILL_DIR}/scripts/search_trajectory.py $ARGUMENTS <trajectory_id> "<query>"
```

Use `--role tool` to restrict to actual tool responses — avoids misleading matches on the agent's own reasoning:
```bash
python ${CLAUDE_SKILL_DIR}/scripts/search_trajectory.py $ARGUMENTS <trajectory_id> "<query>" --role tool
```

Use `--k N` for more matches (default 5, max 10).

**Read a contiguous range of steps in full:**
```bash
python ${CLAUDE_SKILL_DIR}/scripts/get_segment.py $ARGUMENTS <trajectory_id> <start_step> <end_step>
```

Max 5 steps per call (end_step − start_step ≤ 4). After finding a relevant step via search, read it with surrounding context to see the raw tool output.

**Retrieve one trajectory's final content:**
```bash
python ${CLAUDE_SKILL_DIR}/scripts/get_solution.py $ARGUMENTS <trajectory_id>
```

### Step 3 — Cross-check

Confirm:
- Tool outputs in the log match what the agent claims
- Reasoning is not circular (agent does not assume the answer before deriving it)
- Arithmetic and logic are correct

### Step 4 — Synthesize

Write your final answer. **Do NOT reference trajectory IDs, script names, or "agent".**

```
<explanation>Self-contained reasoning a reader could follow without seeing the trajectories. Ground every claim in specific tool observations.</explanation><answer>The exact answer</answer>
```

---

## Operational Guidelines

- **Tool results are ground truth; agent reasoning is not.** What a tool *returned* is an objective observation. What the agent *concluded* is an interpretation that may be wrong. When in conflict, trust the tool output.
- **Count evidence, not trajectories.** One trajectory with a clear tool observation beats many that only reasoned their way to the same answer. Majority agreement alone is not sufficient.
- **Identify divergences.** Focus on steps where agents disagree. Determine which agent's *observation from the environment* was correct.
- **Do not output after only reading solutions.** You must call `get_segment` or `search_trajectory` at least once before synthesizing — this is unconditional, not dependent on whether solutions appear to agree.

## Common Pitfalls

- **Hallucinated observations**: Agent claims a tool returned X, but the log shows Y or nothing.
- **Silent failures**: Agent receives an error but continues as if successful.
- **Circular logic**: Agent assumes the answer before deriving it from data.
- **Arithmetic/logical errors**: Data is correct but the calculation is wrong.
- **Majority bias**: Numerical agreement is weak evidence. One trajectory with a concrete, verifiable tool result beats many with only stated conclusions.

## Deep research tasks

For tasks requiring a full synthesized report, follow this **extended procedure** — do not skip to synthesis after only reading solutions.

**After Step 1**, identify 3–5 specific topics, claims, or sections where candidates differ in depth, accuracy, or coverage. Then for each:

1. **Search for supporting evidence** in the raw trajectory:
   ```bash
   python ${CLAUDE_SKILL_DIR}/scripts/search_trajectory.py $ARGUMENTS <trajectory_id> "<topic>" --role tool
   ```
2. **Read the retrieved steps in context** to verify the claim is grounded in an actual tool observation, not just asserted:
   ```bash
   python ${CLAUDE_SKILL_DIR}/scripts/get_segment.py $ARGUMENTS <trajectory_id> <start> <end>
   ```

Only after inspecting trajectory evidence for the key topics should you synthesize. The final output format:

```
<explanation>How you combined candidates and resolved conflicts — which evidence you relied on and why.</explanation><answer>The complete synthesized response — coherent prose, not a patchwork. Do not mention trajectories, agents, or aggregation anywhere. Cite using <cite url="...">...</cite> where applicable.</answer>
```

**Long-form quality requirements:**
- **Completeness**: cover every important aspect addressed by any candidate
- **Accuracy**: for conflicting claims, pick the more evidence-grounded position — do not average or hedge
- **Coherence**: integrate, do not concatenate; the response must read as a single unified piece
- **No skipping verification**: even if all candidates look similar at the solution level, the trajectories may differ in the evidence they retrieved — check before trusting
