# SCHEMA.md — Autoresearch at Scale

## Purpose

This document defines the canonical data schemas for the Autoresearch system.

It exists to keep all agents, modules, and runtime components aligned on:

- shared state structure
- hypothesis structure
- experiment result structure
- validation and reward semantics
- state transition rules

If `AGENTS.md` defines architectural behavior, `SCHEMA.md` defines the data contracts.

All modules must treat this file as the source of truth for persisted data formats.

---

# Global Rules

## Serialization
- All persisted records use JSON.
- `results.jsonl` is newline-delimited JSON, one record per line.
- `proposal_events.jsonl` is newline-delimited JSON, one record per line.
- `claimed.json` is a JSON array of claim entries.
- `global_best.json` is a single JSON object.
- All timestamps must be ISO-8601 UTC strings.

Example:
```text
2026-03-14T22:31:05Z
```

## Schema Versioning

Every persisted record must include:
```json
"schema_version": "1.0"
```

Rules:
- backward-compatible field additions do not require a major version bump
- incompatible field changes require a schema version bump
- readers must fail loudly on unsupported major versions

## Field Naming
- Use `snake_case` for all keys
- Avoid abbreviations unless standard in ML infra
- Keep field names stable once introduced

---

# 1. Hypothesis Schema

## Canonical Shape
```json
{
  "schema_version": "1.0",
  "hypothesis": "Reduce learning rate from 3e-4 to 2e-4",
  "diff": "MATRIX_LR = 0.02",
  "predicted_effect": "Lower learning rate may improve validation stability",
  "risk": "low",
  "fingerprint": "d41d8cd98f00b204e9800998ecf8427e"
}
```

## Required Fields

- `schema_version: str`
- `hypothesis: str` — one-sentence description of the proposed change
- `diff: str` — exact code change to apply to train.py
- `predicted_effect: str` — why this should improve reward
- `risk: str` — one of: `low`, `medium`, `high`
- `fingerprint: str` — md5(diff)

## Optional Fields

- `parent_fingerprint: str` — fingerprint of parent hypothesis if mutated
- `proposal_source: str` — one of: `seed`, `random`, `mutate`, `llm`
- `proposer_policy: str` — runtime proposer policy, e.g. `baseline` or `history`
- `condition_id: str` — stable experiment-matrix condition identifier
- `notes: str` — freeform annotation

## Invariants

- fingerprint must be deterministic for the same diff
- hypothesis must describe the same change represented in diff
- risk must be one of the allowed values

---

# 2. Experiment Record Schema

## Canonical Shape
```json
{
  "schema_version": "1.0",
  "fingerprint": "abc123",
  "hypothesis_fingerprint": "d41d8cd98f00b204e9800998ecf8427e",
  "config": {
    "MATRIX_LR": 0.02,
    "DEPTH": 8
  },
  "hypothesis": "Reduce learning rate from 3e-4 to 2e-4",
  "val_bpb": 0.8734,
  "reward": 0.7489,
  "status": "kept",
  "outcome": "succeeded",
  "node": "node-03",
  "runtime_seconds": 301.2,
  "total_tflops_consumed": 12.4,
  "artifacts_path": "runs/abc123/",
  "timestamp": "2026-03-14T22:31:05Z"
}
```

## Required Fields

- `schema_version: str`
- `fingerprint: str` — deterministic experiment identifier
- `config: dict` — applied experiment configuration, JSON-serializable
- `val_bpb: float` — validation bits per byte (lower is better)
- `reward: float` — canonical reward score (higher is better)
- `status: str` — one of: `kept`, `rejected`, `failed`
- `outcome: str` — one of: `succeeded`, `failed`, `timed_out`, `cancelled`
- `node: str` — cluster node or executor identity
- `timestamp: str` — terminal completion timestamp ISO-8601 UTC

## Optional Fields

- `hypothesis: str`
- `hypothesis_fingerprint: str`
- `proposal_source: str`
- `proposer_policy: str`
- `condition_id: str`
- `proposal_notes: str`
- `runtime_seconds: float`
- `total_tflops_consumed: float`
- `artifacts_path: str`
- `stderr_path: str`
- `stdout_path: str`
- `parent_fingerprint: str`
- `slurm_job_id: str`
- `error_message: str` — populated when outcome != succeeded
- `peak_vram_mb: float`
- `mfu_percent: float`
- `total_tokens_M: float`
- `num_steps: int`
- `num_params_M: float`
- `depth: int`

## Invariants

- each completed experiment appears exactly once in results.jsonl
- fingerprint must be unique across terminal experiment records
- reward must be computed using the canonical reward formula
- `failed` status must not be used with `outcome = succeeded`
- `kept` and `rejected` require a valid numeric val_bpb

---

# 2.5 Proposal Event Schema

## Purpose

`proposal_events.jsonl` captures proposal lifecycle events that do not belong in
terminal experiment results, such as duplicate skips or safety rejections.

## Canonical Shape
```json
{
  "timestamp": "2026-03-15T18:12:00Z",
  "event_type": "submitted",
  "worker_id": "supervised-batch",
  "fingerprint": "f6945c53f526f35adc5f4ac555408950",
  "hypothesis": "Increase the learning rate to 0.05...",
  "proposal_source": "llm",
  "proposer_policy": "history",
  "condition_id": "history_aware_proposer",
  "reasoning_present": true,
  "reason": "",
  "job_id": "686"
}
```

## Required Fields

- `timestamp: str`
- `event_type: str` — one of: `generated`, `skipped_completed`, `skipped_claimed`, `rejected_safety`, `submitted`
- `worker_id: str`
- `fingerprint: str`

## Optional Fields

- `hypothesis: str`
- `proposal_source: str`
- `proposer_policy: str`
- `condition_id: str`
- `reasoning_present: bool`
- `reason: str`
- `job_id: str`

---

# 3. Claim Entry Schema

## Canonical Shape
```json
{
  "schema_version": "1.0",
  "fingerprint": "abc123",
  "claimed_by": "worker-2",
  "claim_timestamp": "2026-03-14T22:10:00Z",
  "claimed_at_ts": 1741996200.0,
  "ttl_seconds": 480,
  "state": "running",
  "slurm_job_id": "582941"
}
```

## Required Fields

- `schema_version: str`
- `fingerprint: str`
- `claimed_by: str` — worker identity, e.g. hostname+pid
- `claim_timestamp: str` — ISO-8601 UTC
- `claimed_at_ts: float` — Unix timestamp for TTL comparison
- `ttl_seconds: int` — claim lifetime before expiry
- `state: str` — one of: `proposed`, `submitted`, `running`

## Optional Fields

- `slurm_job_id: str` — present once submitted
- `hypothesis_fingerprint: str`
- `notes: str`

## Invariants

- at most one active claim per fingerprint
- expired claims may be reclaimed
- completed experiments must clear or supersede matching active claims

---

# 4. Global Best Register Schema

## Canonical Shape
```json
{
  "schema_version": "1.0",
  "best_fingerprint": "abc123",
  "best_val_bpb": 0.8734,
  "best_reward": 0.7489,
  "best_config": {
    "MATRIX_LR": 0.02,
    "DEPTH": 8
  },
  "updated_at": "2026-03-14T22:31:05Z"
}
```

## Required Fields

- `schema_version: str`
- `best_fingerprint: str` — fingerprint of current best experiment
- `best_val_bpb: float`
- `best_reward: float`
- `best_config: dict`
- `updated_at: str` — ISO-8601 UTC

## Optional Fields

- `hypothesis: str`
- `hypothesis_fingerprint: str`
- `slurm_job_id: str`

## Invariants

- `best_fingerprint` must reference an experiment present in results.jsonl
- `best_config` must match the referenced experiment record
- writes must be lock-protected

---

# 5. Training Metrics Output Schema

## Purpose

Structured metrics emitted by each training run. Source for ingestion and reward computation.
Parsed from the structured block after `---` in train.py stdout.

## Canonical Shape
```json
{
  "schema_version": "1.0",
  "fingerprint": "abc123",
  "val_bpb": 0.8734,
  "runtime_seconds": 301.2,
  "total_tflops_consumed": 12.4,
  "outcome": "succeeded",
  "timestamp": "2026-03-14T22:31:05Z"
}
```

## Parsing from train.py stdout

train.py emits a structured block after `---`:
```
---
val_bpb:          0.891234
training_seconds: 300.1
total_seconds:    312.4
peak_vram_mb:     18432.0
mfu_percent:      47.23
total_tokens_M:   156.4
num_steps:        487
num_params_M:     85.2
depth:            8
```

total_tflops_consumed is derived as:
```
total_tflops = (total_tokens_M * 1e6 * num_flops_per_token) / 1e12
```

num_flops_per_token is printed earlier in stdout as:
```
Estimated FLOPs per token: 1.234e+10
```

## Required Fields

- `schema_version: str`
- `fingerprint: str`
- `outcome: str` — one of: `succeeded`, `failed`, `timed_out`, `cancelled`
- `timestamp: str`

## Required When outcome = succeeded

- `val_bpb: float`

## Optional Fields

- `runtime_seconds: float`
- `total_tflops_consumed: float`
- `peak_vram_mb: float`
- `mfu_percent: float`
- `total_tokens_M: float`
- `num_steps: int`
- `num_params_M: float`
- `depth: int`
- `error_message: str`

## Invariants

- successful runs must emit valid numeric val_bpb
- failed runs must emit outcome != succeeded
- ingestion must prefer structured metrics over parsing human-readable logs

---

# 6. Reward Signal Schema and Semantics

## Canonical Reward Formula
```
reward = (baseline_val_bpb - current_val_bpb) / total_tflops_consumed
```

Higher is better.

## Definitions

- `baseline_val_bpb` — val_bpb of the first successful experiment in results.jsonl
- `current_val_bpb` — val_bpb of the current experiment
- `total_tflops_consumed` — read from structured training metrics

## Fallback Rule

If total_tflops_consumed is unavailable:
```
reward = baseline_val_bpb - current_val_bpb
```

## Ownership

Reward must be computed in exactly one place: `reward_engine.py` or a dedicated write-result path.
It must not be independently reimplemented in multiple modules.

## Invariants

- identical experiment inputs must yield identical reward
- reward semantics must remain stable across orchestrator, dashboard, and ranking logic

---

# 7. State Transition Model

## Experiment Lifecycle
```
hypothesis proposed
  -> claim created        (state: proposed)
  -> submitted to SLURM   (state: submitted)
  -> running              (state: running)
  -> terminal result appended to results.jsonl
  -> claim cleared
  -> optional global best update
```

## Terminal Experiment Outcomes

- `succeeded`
- `failed`
- `timed_out`
- `cancelled`

## Invariants

- a claim must not remain active forever — TTL enforces expiry
- terminal experiment records are append-only
- global best may only update from terminal experiment records

---

# 8. Local Runner Compatibility Contract

## Requirements

- `local_runner` and `slurm_runner` must accept equivalent logical inputs
- both must produce output compatible with the training metrics schema
- orchestrator must not contain execution-backend-specific branching beyond runner selection

## Local Runner Defaults

- CPU device
- max-steps 50
- fast deterministic smoke-test behavior

---

# 9. File Layout Semantics

## State Directory
```
state/
  results.jsonl
  claimed.json
  global_best.json
```

## Run Directory
```
runs/<fingerprint>/
  metrics.json
  stdout.log
  stderr.log
```

---

# 10. Validation Rules

## Readers must reject

- missing required fields
- unsupported schema versions
- invalid enum values
- malformed timestamps
- non-numeric val_bpb for successful runs

## Writers must guarantee

- UTF-8 JSON output
- atomic replace for mutable JSON files where possible
- append-only behavior for results.jsonl
- no partial record emission on success path

---

# 11. Recommended Runtime Enforcement

Enforce this document at runtime using:
- `state/schemas.py` — typed dataclasses
- contract tests in `tests/`

Validated models:
- `Hypothesis`
- `ExperimentRecord`
- `ClaimEntry`
- `GlobalBestRegister`
- `TrainingMetrics`

---

# 12. Sync Rule

This file must stay aligned with:
- `AGENTS.md`
- `.cursor/rules/always.mdc`
- `state/schemas.py`

If schemas change in one place, update all contract sources.
