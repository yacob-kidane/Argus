#!/usr/bin/env python3
"""
Summarize one isolated experiment condition into summary.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

EST_COST_PER_GPU_HOUR = 4.0


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def numeric(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_summary(root: Path) -> dict:
    budget = read_json(root / "budget.json")
    checks = read_json(root / "checks.json")
    results = read_jsonl(root / "state" / "results.jsonl")
    proposal_events = read_jsonl(root / "state" / "proposal_events.jsonl")
    promotions = read_jsonl(root / "state" / "promoted.jsonl")

    succeeded = sorted(
        [r for r in results if str(r.get("outcome", "")).lower() == "succeeded"],
        key=lambda r: r.get("timestamp", ""),
    )
    baseline = succeeded[0] if succeeded else None
    post_baseline = succeeded[1:] if len(succeeded) > 1 else []
    best_record = min(succeeded, key=lambda r: numeric(r.get("val_bpb"), float("inf"))) if succeeded else None

    baseline_val = numeric(baseline.get("val_bpb")) if baseline else None
    best_val = numeric(best_record.get("val_bpb")) if best_record else None
    best_abs_improvement = (
        baseline_val - best_val
        if baseline_val is not None and best_val is not None
        else None
    )
    total_tflops = sum(numeric(r.get("total_tflops_consumed")) for r in succeeded)
    gpu_hours = sum(numeric(r.get("runtime_seconds")) for r in succeeded) / 3600.0
    discovery_per_tflop = (
        best_abs_improvement / total_tflops
        if best_abs_improvement is not None and total_tflops > 0
        else None
    )
    estimated_cost_usd = gpu_hours * EST_COST_PER_GPU_HOUR if gpu_hours else 0.0
    cost_per_discovery = (
        estimated_cost_usd / best_abs_improvement
        if best_abs_improvement is not None and best_abs_improvement > 0
        else None
    )

    baseline_ts = parse_ts(baseline.get("timestamp")) if baseline else None
    first_improvement = next(
        (r for r in post_baseline if numeric(r.get("val_bpb"), float("inf")) < baseline_val),
        None,
    ) if baseline else None
    first_improvement_ts = parse_ts(first_improvement.get("timestamp")) if first_improvement else None
    time_to_first_improvement = (
        int((first_improvement_ts - baseline_ts).total_seconds())
        if baseline_ts and first_improvement_ts
        else None
    )

    rejected_runs = sum(1 for r in post_baseline if r.get("status") == "rejected")
    failed_runs = sum(1 for r in results if r.get("outcome") != "succeeded")
    duplicate_skips = sum(
        1 for event in proposal_events
        if event.get("event_type") in {"skipped_completed", "skipped_claimed"}
    )

    summary = {
        "condition": budget.get("condition_id") or root.name,
        "proposer_policy": budget.get("proposer_policy", ""),
        "runs": len(succeeded),
        "best_val_bpb": best_val,
        "best_absolute_improvement": best_abs_improvement,
        "total_tflops": total_tflops,
        "gpu_hours": gpu_hours,
        "discovery_per_tflop": discovery_per_tflop,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_per_discovery": cost_per_discovery,
        "time_to_first_improvement_seconds": time_to_first_improvement,
        "rejected_runs": rejected_runs,
        "failed_runs": failed_runs,
        "duplicate_skips": duplicate_skips,
        "submitted_proposals": sum(1 for e in proposal_events if e.get("event_type") == "submitted"),
        "generated_proposals": sum(1 for e in proposal_events if e.get("event_type") == "generated"),
        "promotion_records": len(promotions),
        "baseline_fingerprint": baseline.get("fingerprint") if baseline else "",
        "best_fingerprint": best_record.get("fingerprint") if best_record else "",
    }
    if checks:
        checks_list = checks.get("checks", [])
        summary["checks"] = checks_list
        summary["checks_passed"] = sum(1 for c in checks_list if c.get("passed"))
        summary["checks_failed"] = sum(1 for c in checks_list if not c.get("passed"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one experiment condition")
    parser.add_argument("--root", required=True, help="Experiment root directory")
    parser.add_argument("--output", default="", help="Optional explicit output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    summary = build_summary(root)
    output_path = Path(args.output).resolve() if args.output else root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
