# Autoresearch at Scale — Agent Rules

# This file must stay in sync with:
# - AGENTS.md
# - .cursor/rules/always.mdc

## Project Overview

Autoresearch is a coordinated multi-agent experiment system for SLURM GPU clusters.
Agents generate training hypotheses, schedule experiments in parallel, coordinate
through a shared Lustre filesystem, and improve results over time.

Core design philosophy:
- minimal infrastructure
- file-backed state
- append-only experiment history
- lightweight coordination
- cluster-safe execution

---

## Architecture

state/state_store.py
  - append experiment results
  - manage claim coordination
  - track best result
  - enforce state invariants

runners/slurm_runner.py
  - render sbatch scripts
  - submit jobs
  - return job IDs

runners/local_runner.py
  - fake SLURM for testing
  - runs with --max-steps 50 --device cpu
  - must implement identical interface to slurm_runner

orchestrator/orchestrator.py
  - main control loop
  - read state, generate hypotheses, claim, launch, ingest, update best

orchestrator/proposal_engine.py
  - generates hypotheses using OpenAI API
  - inputs: prior results, current best, training code
  - output: structured hypothesis dict (see Hypothesis Schema)

safety/ast_check.py
  - static safety validator
  - rejects unsafe code before execution

dashboard/index.html
  - Chart.js monitoring dashboard

---

## Hard Rules

### Execution boundaries
- Never call sbatch, srun, or salloc directly
- All SLURM interaction must go through runners/slurm_runner.py

### Shared state boundaries
- Never write directly to the shared filesystem
- All state writes must go through state/state_store.py

### Path safety
- Never hardcode filesystem paths
- All paths must come from config.py

### Infrastructure simplicity
- No external databases, Redis, Postgres, or message queues
- State is entirely filesystem-backed

### Dependency policy
- Prefer Python standard library
- Avoid new dependencies unless necessary

### Language policy
- Python only
- Existing Node tooling (dashboard) is allowed

---

## State Model

results.jsonl
  - append-only experiment log
  - immutable entries
  - source of truth for completed runs
  - supports concurrent writers

claimed.json
  - claim table with TTL
  - prevents duplicate experiment execution
  - entries expire automatically if workers crash

global_best.json
  - shared register tracking best experiment
  - lock-protected writes
  - always references a record present in results.jsonl

---

## Experiment Record Format

{
  "fingerprint": "abc123",
  "config": {},
  "val_bpb": 0.8734,
  "reward": 0.7489,
  "status": "kept",
  "node": "node-03",
  "timestamp": "2026-03-15T12:04:55Z"
}

---

## Hypothesis Schema

{
  "hypothesis": "one sentence description",
  "diff": "exact Python code change to apply to train.py",
  "predicted_effect": "why this should improve reward",
  "risk": "low | medium | high",
  "fingerprint": "md5 of diff"
}

---

## Reward Signal

reward = (baseline_val_bpb - current_val_bpb) / total_tflops_consumed

Higher is better.

Definitions:
- baseline_val_bpb: val_bpb of the first successful experiment in results.jsonl
- current_val_bpb: val_bpb of the current experiment
- total_tflops_consumed: total TFLOPs consumed by current experiment, read from training output

Fallback:
If total_tflops_consumed is unavailable:
  reward = baseline_val_bpb - current_val_bpb

Reward must be computed only in state/write_result.py or a dedicated reward module.
Never compute reward independently in multiple modules.

---

## Key Interfaces (Do Not Change)

state_store.append_result(record: dict) -> None
state_store.try_claim(fingerprint: str) -> bool
state_store.expire_stale_claims() -> int
state_store.read_global_best() -> dict
state_store.update_global_best(record: dict) -> bool
slurm_runner.submit(job_script: str, exp_id: str) -> str
proposal_engine.next_hypothesis(results: list, best: dict, train_py: str) -> dict
ast_check.is_safe(code: str) -> tuple[bool, str]

---

## System Invariants

1. A fingerprint may have at most one active claim.
2. Every completed experiment must appear exactly once in results.jsonl.
3. A completed experiment must clear or supersede any active claim for the same fingerprint.
4. global_best.json must always reference an experiment present in results.jsonl.
5. local_runner and slurm_runner must implement identical interfaces.
6. The orchestrator must not depend on the execution backend.

---

## Failure Model

Worker crash
  - stale claims expire via TTL
  - experiments can be reclaimed

Job failure
  - failed jobs still produce a terminal result record
  - orchestrator continues operating

Orchestrator restart
  - system resumes from filesystem state
  - no in-memory state required for recovery

---

## Orchestrator Control Loop

1. read results.jsonl
2. read global_best.json
3. generate hypothesis
4. compute fingerprint = md5(diff)
5. attempt try_claim(fingerprint)
6. if claim succeeds: submit job
7. wait for result
8. append_result()
9. update_global_best()
10. repeat

---

## Environment

Cluster: Fluidstack H100, SLURM, Lustre at /mnt/sharefs
Python: /usr/bin/python3, version 3.10.12
Node: v24.14.0 via nvm
Constraints: no sudo, no apt install, no root access

---

## Testing Model

Local testing must occur before cluster execution.

python orchestrator/orchestrator.py --mode local

Runs experiments on CPU with --max-steps 50.
Cluster execution only after local verification passes.

---

## Safety Principles

Generated code must pass static validation before execution.

Reject any code containing:
- filesystem deletion
- network calls
- shell execution via os.system / subprocess
- dynamic imports via __import__
- eval() or exec() on non-literals

---

## Operational Philosophy

Workers propose hypotheses.
Experiments run in parallel.
Results accumulate in shared memory.
The system continuously improves the best-known configuration.
