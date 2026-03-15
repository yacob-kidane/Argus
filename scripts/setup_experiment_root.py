#!/usr/bin/env python3
"""
Create an isolated experiment root for a formal matrix condition.

This keeps state, logs, jobs, reasoning logs, and runtime secrets together
under a single shared root so SLURM and the dashboard resolve paths correctly.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))

import config
from state.schemas import GlobalBestRegister


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def initialize_empty_state(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "results.jsonl").write_text("", encoding="utf-8")
    (state_dir / "promoted.jsonl").write_text("", encoding="utf-8")
    (state_dir / "proposal_events.jsonl").write_text("", encoding="utf-8")
    write_json(state_dir / "claimed.json", [])
    write_json(state_dir / "global_best.json", GlobalBestRegister.empty().to_dict())


def snapshot_state(source_state_dir: Path, snapshot_root: Path) -> str | None:
    if not source_state_dir.exists():
        return None
    snapshot_name = f"state_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_path = snapshot_root / snapshot_name
    shutil.copytree(source_state_dir, snapshot_path, dirs_exist_ok=True)
    return str(snapshot_path)


def build_budget_manifest(args: argparse.Namespace, root: Path, snapshot_path: str | None) -> dict:
    return {
        "generated_at": utc_now(),
        "condition_id": args.condition_id,
        "proposer_policy": args.proposer_policy,
        "shared_root": str(root),
        "state_dir": str(root / "state"),
        "logs_dir": str(root / "logs"),
        "jobs_dir": str(root / "jobs"),
        "source_state_snapshot": snapshot_path,
        "train_py_source": args.train_py_source or config.TRAIN_PY_SOURCE,
        "program_md": args.program_md or config.PROGRAM_MD,
        "openai_model": config.OPENAI_MODEL,
        "max_concurrent": args.max_concurrent if args.max_concurrent is not None else config.MAX_CONCURRENT,
        "poll_interval_secs": args.poll_interval_secs if args.poll_interval_secs is not None else config.POLL_INTERVAL_SECS,
        "slurm_partition": config.SLURM_PARTITION,
        "slurm_gpus_per_job": config.SLURM_GPUS_PER_JOB,
        "slurm_cpus_per_gpu": config.SLURM_CPUS_PER_GPU,
        "slurm_mem_gb": config.SLURM_MEM_GB,
        "slurm_time_minutes": args.slurm_time_minutes if args.slurm_time_minutes is not None else config.SLURM_TIME_MINUTES,
        "ttl_seconds": config.TTL_SECONDS,
        "tinker_threshold": config.TINKER_THRESHOLD,
        "tinker_promotion_threshold": args.tinker_promotion_threshold
        if args.tinker_promotion_threshold is not None
        else config.TINKER_PROMOTION_THRESHOLD,
        "notes": args.notes or "",
    }


def write_notes_template(root: Path, manifest: dict) -> None:
    notes = f"""# {manifest['condition_id']}

## Purpose
- TODO: describe the condition and why it exists.

## Runtime
- `PROPOSER_POLICY={manifest['proposer_policy']}`
- `AUTORESEARCH_CONDITION_ID={manifest['condition_id']}`
- `AUTORESEARCH_SHARED_DIR={manifest['shared_root']}`
- `AUTORESEARCH_STATE_DIR={manifest['state_dir']}`

## Budget
- Max concurrent: {manifest['max_concurrent']}
- SLURM time minutes: {manifest['slurm_time_minutes']}
- Train source: `{manifest['train_py_source']}`
- Program file: `{manifest['program_md']}`

## Evidence
- TODO: add screenshots, outcome notes, and summary observations.
"""
    write_text(root / "notes.md", notes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up an isolated experiment root")
    parser.add_argument("--root", required=True, help="Target experiment root directory")
    parser.add_argument("--condition-id", required=True, help="Stable condition identifier")
    parser.add_argument(
        "--proposer-policy",
        default="history",
        choices=["history", "baseline"],
        help="Runtime proposer policy for this condition",
    )
    parser.add_argument(
        "--source-state-dir",
        default=str(REPO_ROOT / "state"),
        help="Source state directory to snapshot before the condition",
    )
    parser.add_argument(
        "--seed-from-state",
        action="store_true",
        help="Seed the isolated state directory from the source state snapshot",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing state, logs, jobs, and generated summaries before setup",
    )
    parser.add_argument(
        "--preserve-logs",
        action="store_true",
        help="When resetting, keep existing logs so completed runs can be re-ingested",
    )
    parser.add_argument(
        "--preserve-jobs",
        action="store_true",
        help="When resetting, keep existing job artifacts such as hypothesis.json",
    )
    parser.add_argument("--train-py-source", default="", help="Override train.py source path for the manifest")
    parser.add_argument("--program-md", default="", help="Override program.md path for the manifest")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Override max concurrent for the manifest")
    parser.add_argument("--poll-interval-secs", type=int, default=None, help="Override poll interval for the manifest")
    parser.add_argument("--slurm-time-minutes", type=int, default=None, help="Override SLURM time budget for the manifest")
    parser.add_argument(
        "--tinker-promotion-threshold",
        type=float,
        default=None,
        help="Override promotion threshold for the manifest",
    )
    parser.add_argument("--notes", default="", help="Freeform notes stored in the manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    state_dir = root / "state"
    logs_dir = root / "logs"
    jobs_dir = root / "jobs"
    snapshots_dir = root / "snapshots"

    if args.reset and root.exists():
        reset_dirs = [state_dir]
        if not args.preserve_jobs:
            reset_dirs.append(jobs_dir)
        if not args.preserve_logs:
            reset_dirs.append(logs_dir)
        for directory in reset_dirs:
            if directory.exists():
                shutil.rmtree(directory)
        for file_path in (root / "summary.json", root / "checks.json"):
            if file_path.exists():
                file_path.unlink()

    root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_state(Path(args.source_state_dir), snapshots_dir)
    initialize_empty_state(state_dir)

    if args.seed_from_state and snapshot_path:
        shutil.rmtree(state_dir)
        shutil.copytree(snapshot_path, state_dir)

    copy_if_exists(REPO_ROOT / ".env", root / ".env")
    copy_if_exists(REPO_ROOT / "jobs" / "write_result.py", jobs_dir / "write_result.py")

    manifest = build_budget_manifest(args, root, snapshot_path)
    write_json(root / "budget.json", manifest)
    write_notes_template(root, manifest)

    print(json.dumps({
        "root": str(root),
        "condition_id": args.condition_id,
        "proposer_policy": args.proposer_policy,
        "snapshot_path": snapshot_path,
        "seeded_from_state": args.seed_from_state,
    }, indent=2))


if __name__ == "__main__":
    main()
