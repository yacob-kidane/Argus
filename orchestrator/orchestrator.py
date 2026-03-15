"""
Main orchestrator loop.
Coordinates the research cycle:
  read state → generate hypothesis → claim → submit → ingest → repeat

Usage:
    python3 orchestrator/orchestrator.py --mode slurm   # real cluster
    python3 orchestrator/orchestrator.py --mode local   # local CPU testing
    python3 orchestrator/orchestrator.py --mode dry     # dry run, no jobs
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from state import state_store
from state.schemas import Hypothesis
from orchestrator.proposal_engine import (
    append_reasoning_log,
    next_hypothesis,
    seed_hypothesis,
)
from safety.ast_check import is_safe, log_rejection

MAX_EXPERIMENTS = int(os.getenv("AUTORESEARCH_MAX_EXPERIMENTS", "0"))


def _get_runner(mode: str):
    if mode == "slurm":
        from runners.slurm_runner import submit
        return submit
    elif mode == "local":
        from runners.local_runner import submit
        return submit
    elif mode == "dry":
        from runners.slurm_runner import submit
        return lambda h: submit(h, dry_run=True)
    else:
        raise ValueError(f"unknown mode: {mode}")


def _log(msg: str, worker_id: str | None = None) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    prefix = f"[{worker_id}] " if worker_id else ""
    print(f"[{ts}] {prefix}{msg}", flush=True)


def _append_proposal_event(
    event_type: str,
    worker_id: str,
    hypothesis: Hypothesis,
    reason: str = "",
    job_id: str = "",
) -> None:
    """Write an append-only proposal event for later matrix analysis."""
    state_store.append_proposal_event({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "worker_id": worker_id,
        "fingerprint": hypothesis.fingerprint,
        "hypothesis": hypothesis.hypothesis,
        "proposal_source": hypothesis.proposal_source,
        "proposer_policy": hypothesis.proposer_policy,
        "condition_id": hypothesis.condition_id,
        "reasoning_present": bool(hypothesis.notes),
        "reason": reason,
        "job_id": job_id,
    })


def _count_slurm_jobs() -> int:
    """Count active SLURM jobs for this user."""
    try:
        from runners.slurm_runner import get_active_job_count
        return get_active_job_count()
    except Exception:
        return 0


def _result_count() -> int:
    """Count completed experiment records from shared state."""
    try:
        return len(state_store.read_results())
    except Exception:
        return 0


def run_loop(
    mode: str = "slurm",
    max_iterations: int | None = None,
    stop_after: int | None = None,
) -> None:
    """
    Main research loop.
    Runs until interrupted or max_iterations reached.

    Args:
        mode: slurm | local | dry
        max_iterations: stop after N hypothesis submissions (None = forever)
        stop_after: stop after N seconds (None = forever)
    """
    state_store.ensure_state_dirs()
    submit = _get_runner(mode)
    worker_id = os.environ.get("AUTORESEARCH_WORKER_ID", f"worker-{os.getpid()}")
    proposer_policy = config.PROPOSER_POLICY
    condition_id = config.CONDITION_ID

    _log(f"Orchestrator starting — pid={os.getpid()} mode={mode} "
         f"max_concurrent={config.MAX_CONCURRENT} "
         f"poll={config.POLL_INTERVAL_SECS}s "
         f"policy={proposer_policy} "
         f"condition={condition_id or '-'}", worker_id)

    iteration      = 0
    submitted      = 0
    skipped_dup    = 0
    skipped_safety = 0
    t_start        = time.time()

    # Load train.py once — reread on each iteration to pick up any manual edits
    def read_train_py():
        with open(config.TRAIN_PY_SOURCE, "r") as f:
            return f.read()

    while True:
        iteration += 1

        # ── Stop conditions ────────────────────────────────────────────────
        if max_iterations and submitted >= max_iterations:
            _log(f"Reached max_iterations={max_iterations}, stopping.", worker_id)
            break
        if stop_after and (time.time() - t_start) >= stop_after:
            _log(f"Reached stop_after={stop_after}s, stopping.", worker_id)
            break

        # ── Concurrency check ──────────────────────────────────────────────
        active_claims = state_store.count_active_claims()
        if mode == "slurm":
            active_jobs = _count_slurm_jobs()
        else:
            active_jobs = active_claims

        if active_jobs >= config.MAX_CONCURRENT:
            _log(f"At capacity ({active_jobs}/{config.MAX_CONCURRENT}) "
                 f"— waiting {config.POLL_INTERVAL_SECS}s", worker_id)
            time.sleep(config.POLL_INTERVAL_SECS)
            continue

        # ── Read state ─────────────────────────────────────────────────────
        results = state_store.read_results()
        best    = state_store.read_global_best()
        active_fps = state_store.list_active_fingerprints()

        if MAX_EXPERIMENTS > 0 and len(results) >= MAX_EXPERIMENTS:
            _log(
                f"STOP budget reached results={len(results)} max={MAX_EXPERIMENTS}",
                worker_id,
            )
            break

        _log(f"iter={iteration} results={len(results)} "
             f"active={active_jobs}/{config.MAX_CONCURRENT} "
             f"best_val_bpb={best.best_val_bpb:.4f} "
             f"best_reward={best.best_reward:.6f}", worker_id)

        # ── Expire stale claims ────────────────────────────────────────────
        expired = state_store.expire_stale_claims()
        if expired:
            _log(f"Expired {expired} stale claims", worker_id)

        # ── Generate hypothesis ────────────────────────────────────────────
        try:
            if not results:
                # First run — use seed to establish baseline
                hypothesis = seed_hypothesis()
                hypothesis.condition_id = condition_id
                append_reasoning_log(worker_id, {
                    "fingerprint": hypothesis.fingerprint,
                    "hypothesis": hypothesis.hypothesis,
                    "reasoning": hypothesis.notes or "seed fallback",
                    "predicted_effect": hypothesis.predicted_effect,
                    "risk": hypothesis.risk,
                    "diff": hypothesis.diff,
                })
                _log("Using seed hypothesis (no results yet)", worker_id)
            else:
                hypothesis = next_hypothesis(
                    results=results,
                    best=best.to_dict(),
                    train_py=read_train_py(),
                    active_fingerprints=active_fps,
                    worker_id=worker_id,
                    proposer_policy=proposer_policy,
                    condition_id=condition_id,
                )
        except Exception as e:
            _log(f"ERROR generating hypothesis: {e}", worker_id)
            time.sleep(config.POLL_INTERVAL_SECS)
            continue

        _log(f"Hypothesis: {hypothesis.hypothesis[:80]}", worker_id)
        _log(f"Diff: {hypothesis.diff[:60]}", worker_id)
        _log(f"Fingerprint: {hypothesis.fingerprint}", worker_id)
        _append_proposal_event("generated", worker_id, hypothesis)

        # ── Deduplication check ────────────────────────────────────────────
        past_fps = {r.get("fingerprint") for r in results}
        if hypothesis.fingerprint in past_fps:
            _log(f"SKIP — already completed: {hypothesis.fingerprint[:8]}", worker_id)
            _append_proposal_event(
                "skipped_completed",
                worker_id,
                hypothesis,
                reason="fingerprint already completed",
            )
            skipped_dup += 1
            time.sleep(2)
            continue

        # ── Claim ──────────────────────────────────────────────────────────
        claimed = state_store.try_claim(
            hypothesis.fingerprint,
            claimed_by=f"{worker_id}-pid{os.getpid()}",
        )
        if not claimed:
            _log(f"CLAIM failed fingerprint={hypothesis.fingerprint}", worker_id)
            _log(f"SKIP — already claimed: {hypothesis.fingerprint[:8]}", worker_id)
            _append_proposal_event(
                "skipped_claimed",
                worker_id,
                hypothesis,
                reason="fingerprint already claimed",
            )
            skipped_dup += 1
            time.sleep(2)
            continue
        _log(f"CLAIM success fingerprint={hypothesis.fingerprint}", worker_id)

        # ── Safety check ───────────────────────────────────────────────────
        with open(config.TRAIN_PY_SOURCE, "r") as f:
            source = f.read()
        from runners.local_runner import _apply_diff
        patched = _apply_diff(source, hypothesis.diff)
        ok, reason = is_safe(patched)
        if not ok:
            log_rejection(hypothesis.fingerprint, reason)
            state_store.release_claim(hypothesis.fingerprint)
            _log(f"SKIP — safety check failed: {reason}", worker_id)
            _append_proposal_event(
                "rejected_safety",
                worker_id,
                hypothesis,
                reason=reason,
            )
            skipped_safety += 1
            time.sleep(2)
            continue

        # ── Submit ─────────────────────────────────────────────────────────
        try:
            handle = submit(hypothesis)
            submitted += 1
            state_store.update_claim_state(
                hypothesis.fingerprint, "submitted",
                slurm_job_id=handle.get("job_id", "")
            )
            _log(f"SUBMITTED job_id={handle['job_id']} "
                 f"fp={hypothesis.fingerprint[:8]} "
                 f"runner={handle['runner']}", worker_id)
            _append_proposal_event(
                "submitted",
                worker_id,
                hypothesis,
                job_id=handle.get("job_id", ""),
            )
        except Exception as e:
            state_store.release_claim(hypothesis.fingerprint)
            _log(f"ERROR submitting job: {e}", worker_id)
            time.sleep(config.POLL_INTERVAL_SECS)
            continue

        # ── Stats ──────────────────────────────────────────────────────────
        _log(f"Stats — submitted={submitted} "
             f"skipped_dup={skipped_dup} "
             f"skipped_safety={skipped_safety}", worker_id)

        # ── Wait before next iteration ─────────────────────────────────────
        time.sleep(config.POLL_INTERVAL_SECS)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch orchestrator")
    parser.add_argument(
        "--mode", choices=["slurm", "local", "dry"],
        default="slurm",
        help="Execution backend (default: slurm)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Stop after N submissions",
    )
    parser.add_argument(
        "--stop-after", type=int, default=None,
        help="Stop after N seconds",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Shorthand for --mode dry",
    )
    args = parser.parse_args()

    mode = "dry" if args.dry_run else args.mode

    try:
        run_loop(
            mode=mode,
            max_iterations=args.max_iterations,
            stop_after=args.stop_after,
        )
    except KeyboardInterrupt:
        print("\n[orchestrator] interrupted by user")


if __name__ == "__main__":
    main()
