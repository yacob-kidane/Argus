#!/usr/bin/env python3
"""
Run isolated systems-correctness checks for one condition root.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_empty_state(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "results.jsonl").write_text("", encoding="utf-8")
    (state_dir / "promoted.jsonl").write_text("", encoding="utf-8")
    (state_dir / "proposal_events.jsonl").write_text("", encoding="utf-8")
    (state_dir / "claimed.json").write_text("[]\n", encoding="utf-8")
    (state_dir / "global_best.json").write_text(
        json.dumps({
            "best_fingerprint": None,
            "best_val_bpb": 999.0,
            "best_reward": -999.0,
            "last_updated": "",
            "schema_version": "1.0",
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def import_state_modules(root: Path):
    os.environ["AUTORESEARCH_SHARED_DIR"] = str(root)
    os.environ["AUTORESEARCH_STATE_DIR"] = str(root / "state")
    sys.path.insert(0, str(REPO_ROOT))
    from state import state_store
    from state.schemas import ExperimentRecord
    return state_store, ExperimentRecord


def build_record(ExperimentRecord, fingerprint: str, val_bpb: float, reward: float) -> dict:
    return ExperimentRecord(
        fingerprint=fingerprint,
        config={"SYSTEM_TEST": fingerprint},
        val_bpb=val_bpb,
        reward=reward,
        status="kept" if reward >= 0 else "rejected",
        outcome="succeeded",
        node="systems-check",
        timestamp=utc_now(),
        hypothesis=f"system test for {fingerprint}",
        hypothesis_fingerprint=fingerprint,
        proposal_source="system_test",
        proposer_policy="history",
        condition_id="",
    ).to_dict()


def run_claim_dedupe(root: Path) -> list[dict]:
    state_dir = root / "state"
    initialize_empty_state(state_dir)
    state_store, ExperimentRecord = import_state_modules(root)

    fingerprint = "systems-claim-dedupe-fp"
    first = state_store.try_claim(fingerprint, claimed_by="worker-a")
    second = state_store.try_claim(fingerprint, claimed_by="worker-b")
    record = build_record(ExperimentRecord, fingerprint, 1.1, 0.0)
    state_store.append_result(record)
    state_store.update_global_best(record)
    state_store.release_claim(fingerprint)
    third = state_store.try_claim(fingerprint, claimed_by="worker-c")

    return [
        {"name": "first_claim_succeeds", "passed": first is True},
        {"name": "duplicate_claim_blocked", "passed": second is False},
        {"name": "completed_fingerprint_not_reclaimable", "passed": third is False},
    ]


def _claim_worker(args: tuple[str, str, int]) -> bool:
    root_str, fingerprint, idx = args
    root = Path(root_str)
    state_store, _ = import_state_modules(root)
    return state_store.try_claim(fingerprint, claimed_by=f"worker-{idx}")


def run_concurrency(root: Path) -> list[dict]:
    state_dir = root / "state"
    initialize_empty_state(state_dir)
    state_store, _ = import_state_modules(root)

    fingerprint = "systems-concurrency-fp"
    with mp.Pool(8) as pool:
        outcomes = pool.map(
            _claim_worker,
            [(str(root), fingerprint, idx) for idx in range(8)],
        )

    success_count = sum(1 for ok in outcomes if ok)
    active_count = len(state_store.list_active_fingerprints())
    state_store.release_claim(fingerprint)
    return [
        {"name": "exactly_one_process_claims", "passed": success_count == 1, "observed": success_count},
        {"name": "one_active_claim_recorded", "passed": active_count == 1, "observed": active_count},
    ]


def run_global_best(root: Path) -> list[dict]:
    state_dir = root / "state"
    initialize_empty_state(state_dir)
    state_store, ExperimentRecord = import_state_modules(root)

    worse = build_record(ExperimentRecord, "systems-best-worse", 1.2, 0.0)
    better = build_record(ExperimentRecord, "systems-best-better", 1.0, 0.01)
    later_worse = build_record(ExperimentRecord, "systems-best-later-worse", 1.1, -0.01)

    for record in (worse, better, later_worse):
        state_store.append_result(record)
        state_store.update_global_best(record)

    best = state_store.read_global_best().to_dict()
    return [
        {
            "name": "best_fingerprint_tracks_lowest_val_bpb",
            "passed": best.get("best_fingerprint") == "systems-best-better",
            "observed": best.get("best_fingerprint"),
        },
        {
            "name": "best_val_bpb_is_updated",
            "passed": abs(float(best.get("best_val_bpb", 999.0)) - 1.0) < 1e-9,
            "observed": best.get("best_val_bpb"),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated systems correctness checks")
    parser.add_argument("--root", required=True, help="Condition root directory")
    parser.add_argument(
        "--test",
        required=True,
        choices=["claim_dedupe", "concurrency", "global_best"],
        help="Which systems check set to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    if args.test == "claim_dedupe":
        checks = run_claim_dedupe(root)
    elif args.test == "concurrency":
        checks = run_concurrency(root)
    else:
        checks = run_global_best(root)

    payload = {
        "generated_at": utc_now(),
        "test": args.test,
        "checks": checks,
    }
    (root / "checks.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
