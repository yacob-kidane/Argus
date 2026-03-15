"""
Shared filesystem state management.
All state writes go through this module — never write state files directly.

Coordination primitives:
  results.jsonl    — G-Set CRDT, append-only
  claimed.json     — optimistic lock with TTL
  global_best.json — lock-protected shared register
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

from filelock import FileLock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from state.schemas import (
    ExperimentRecord, ClaimEntry, GlobalBestRegister, TrainingMetrics
)

# ── Locks ──────────────────────────────────────────────────────────────────
_best_lock    = FileLock(config.GLOBAL_BEST_JSON + ".lock", timeout=10)
_claimed_lock = FileLock(config.CLAIMED_JSON + ".lock", timeout=10)


# ── results.jsonl ──────────────────────────────────────────────────────────

def append_result(record: dict) -> None:
    """
    Append one experiment record to results.jsonl.
    POSIX append is atomic for writes under 4KB — safe for concurrent writers.
    """
    line = json.dumps(record) + "\n"
    with open(config.RESULTS_JSONL, "a") as f:
        f.write(line)


def read_results() -> list[dict]:
    """Read all experiment records. Returns [] if file empty or missing."""
    if not os.path.exists(config.RESULTS_JSONL):
        return []
    records = []
    with open(config.RESULTS_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_baseline_val_bpb() -> float | None:
    """
    Return val_bpb of the first successful experiment.
    Returns None if no successful experiment exists yet.
    """
    for record in read_results():
        if record.get("outcome") == "succeeded" and "val_bpb" in record:
            return float(record["val_bpb"])
    return None


def has_completed_result(fingerprint: str) -> bool:
    """Return True if this fingerprint already has a terminal result."""
    return any(record.get("fingerprint") == fingerprint for record in read_results())


def can_submit(fingerprint: str) -> bool:
    """
    Return True if a fingerprint is eligible for submission.
    A fingerprint is submit-eligible iff it is not completed and not actively claimed.
    """
    if has_completed_result(fingerprint):
        return False
    return fingerprint not in list_active_fingerprints()


def read_promotions() -> list[dict]:
    """Read promotion records from promoted.jsonl."""
    if not os.path.exists(config.PROMOTED_JSONL):
        return []
    records = []
    with open(config.PROMOTED_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def has_been_promoted(fingerprint: str) -> bool:
    """Return True if a fingerprint already has a promotion record."""
    return any(record.get("fingerprint") == fingerprint for record in read_promotions())


def append_promotion(record: dict) -> None:
    """Append one promotion record to promoted.jsonl."""
    line = json.dumps(record) + "\n"
    with open(config.PROMOTED_JSONL, "a") as f:
        f.write(line)


def read_proposal_events() -> list[dict]:
    """Read proposal event records from proposal_events.jsonl."""
    if not os.path.exists(config.PROPOSAL_EVENTS_JSONL):
        return []
    records = []
    with open(config.PROPOSAL_EVENTS_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_proposal_event(record: dict) -> None:
    """Append one proposal event to proposal_events.jsonl."""
    line = json.dumps(record) + "\n"
    with open(config.PROPOSAL_EVENTS_JSONL, "a") as f:
        f.write(line)


# ── claimed.json ───────────────────────────────────────────────────────────

def _read_claimed_raw() -> list[dict]:
    if not os.path.exists(config.CLAIMED_JSON):
        return []
    with open(config.CLAIMED_JSON, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _write_claimed_raw(claims: list[dict]) -> None:
    with open(config.CLAIMED_JSON, "w") as f:
        json.dump(claims, f, indent=2)


def expire_stale_claims() -> int:
    """
    Remove claims older than TTL_SECONDS.
    Returns number of claims removed.
    """
    now = time.time()
    with _claimed_lock:
        claims = _read_claimed_raw()
        fresh = [c for c in claims
                 if now - c.get("claimed_at_ts", 0) < config.TTL_SECONDS]
        removed = len(claims) - len(fresh)
        if removed > 0:
            _write_claimed_raw(fresh)
    return removed


def try_claim(fingerprint: str, claimed_by: str = "orchestrator") -> bool:
    """
    Attempt to claim an experiment by fingerprint.
    Returns True if claim succeeded, False if already claimed or completed.
    fingerprint = md5(diff)
    """
    now = time.time()
    ts = datetime.now(timezone.utc).isoformat()
    with _claimed_lock:
        # Terminal results are append-only, so a completed fingerprint is never claimable again.
        if has_completed_result(fingerprint):
            return False
        claims = _read_claimed_raw()
        # Expire stale claims inline
        claims = [c for c in claims
                  if now - c.get("claimed_at_ts", 0) < config.TTL_SECONDS]
        # Check if already claimed
        if any(c["fingerprint"] == fingerprint for c in claims):
            return False
        # Add new claim
        entry = ClaimEntry(
            fingerprint=fingerprint,
            claimed_by=claimed_by,
            claim_timestamp=ts,
            claimed_at_ts=now,
            ttl_seconds=config.TTL_SECONDS,
            state="proposed",
        )
        claims.append(entry.to_dict())
        _write_claimed_raw(claims)
        return True


def update_claim_state(fingerprint: str, state: str,
                       slurm_job_id: str = "") -> None:
    """Update the state of an existing claim (proposed -> submitted -> running)."""
    assert state in ("proposed", "submitted", "running")
    with _claimed_lock:
        claims = _read_claimed_raw()
        for c in claims:
            if c["fingerprint"] == fingerprint:
                c["state"] = state
                if slurm_job_id:
                    c["slurm_job_id"] = slurm_job_id
                break
        _write_claimed_raw(claims)


def release_claim(fingerprint: str) -> None:
    """Remove a claim after experiment completes or fails."""
    with _claimed_lock:
        claims = _read_claimed_raw()
        claims = [c for c in claims if c["fingerprint"] != fingerprint]
        _write_claimed_raw(claims)


def list_active_fingerprints() -> list[str]:
    """Return currently claimed fingerprints (unexpired)."""
    now = time.time()
    with _claimed_lock:
        claims = _read_claimed_raw()
        return [c["fingerprint"] for c in claims
                if now - c.get("claimed_at_ts", 0) < config.TTL_SECONDS]


def count_active_claims() -> int:
    """Return number of currently active (unexpired) claims."""
    return len(list_active_fingerprints())


# ── global_best.json ───────────────────────────────────────────────────────

def read_global_best() -> GlobalBestRegister:
    """Read current best experiment register."""
    with _best_lock:
        if not os.path.exists(config.GLOBAL_BEST_JSON):
            return GlobalBestRegister.empty()
        with open(config.GLOBAL_BEST_JSON, "r") as f:
            try:
                return GlobalBestRegister.from_dict(json.load(f))
            except (json.JSONDecodeError, Exception):
                return GlobalBestRegister.empty()


def _should_update_global_best(current: GlobalBestRegister, record: dict) -> bool:
    """Return True if this successful record should replace the current best."""
    if record.get("outcome") != "succeeded":
        return False

    if not current.best_fingerprint:
        return True

    new_reward = float(record.get("reward", float("-inf")))
    best_reward = float(
        current.best_reward if current.best_reward is not None else float("-inf")
    )
    if new_reward > best_reward:
        return True

    if new_reward == best_reward:
        new_val_bpb = float(record.get("val_bpb", float("inf")))
        best_val_bpb = float(
            current.best_val_bpb if current.best_val_bpb is not None else float("inf")
        )
        return new_val_bpb < best_val_bpb

    return False


def _write_global_best_raw(register: GlobalBestRegister) -> None:
    """Atomically replace global_best.json on shared storage."""
    tmp_path = config.GLOBAL_BEST_JSON + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(register.to_dict(), f, indent=2)
    os.replace(tmp_path, config.GLOBAL_BEST_JSON)


def update_global_best(record: dict) -> bool:
    """
    Update global best if record has higher reward.
    Returns True if updated, False if not better.
    Lock-protected. Invariant: always references a record in results.jsonl.
    """
    with _best_lock:
        current = read_global_best()
        if not _should_update_global_best(current, record):
            return False

        updated = GlobalBestRegister(
            best_fingerprint=record.get("fingerprint"),
            best_val_bpb=record.get("val_bpb", 999.0),
            best_reward=record.get("reward", 0.0),
            best_config=record.get("config", {}),
            updated_at=datetime.now(timezone.utc).isoformat(),
            hypothesis=record.get("hypothesis", ""),
            hypothesis_fingerprint=record.get("hypothesis_fingerprint", ""),
            slurm_job_id=record.get("slurm_job_id", ""),
        )
        _write_global_best_raw(updated)
        return True
    return False


# ── Setup ──────────────────────────────────────────────────────────────────

def ensure_state_dirs() -> None:
    """Create all required directories and initialize state files."""
    for d in [config.STATE_DIR, config.JOBS_DIR, config.LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(config.RESULTS_JSONL):
        open(config.RESULTS_JSONL, "w").close()
    if not os.path.exists(config.PROMOTED_JSONL):
        open(config.PROMOTED_JSONL, "w").close()
    if not os.path.exists(config.PROPOSAL_EVENTS_JSONL):
        open(config.PROPOSAL_EVENTS_JSONL, "w").close()
    if not os.path.exists(config.CLAIMED_JSON):
        _write_claimed_raw([])
    if not os.path.exists(config.GLOBAL_BEST_JSON):
        with open(config.GLOBAL_BEST_JSON, "w") as f:
            json.dump(GlobalBestRegister.empty().to_dict(), f, indent=2)


if __name__ == "__main__":
    ensure_state_dirs()
    open(config.RESULTS_JSONL, "w").close()
    open(config.PROMOTED_JSONL, "w").close()
    _write_claimed_raw([])
    _write_global_best_raw(GlobalBestRegister.empty())
    print("[state_store] dirs ok")

    # Test append + read
    now = datetime.now(timezone.utc).isoformat()
    test_record = ExperimentRecord(
        fingerprint="test001",
        config={"MATRIX_LR": 0.02},
        val_bpb=0.95,
        reward=0.05,
        status="kept",
        outcome="succeeded",
        node="local",
        timestamp=now,
        hypothesis="test hypothesis",
    ).to_dict()
    append_result(test_record)
    results = read_results()
    assert any(r["fingerprint"] == "test001" for r in results)
    print(f"[state_store] append ok — {len(results)} records")

    # Test baseline
    baseline = get_baseline_val_bpb()
    assert baseline is not None
    print(f"[state_store] baseline val_bpb: {baseline}")

    # Test claim lifecycle
    fp = "testfingerprint001"
    assert try_claim(fp) == True,  "first claim should succeed"
    assert try_claim(fp) == False, "duplicate claim should fail"
    assert count_active_claims() >= 1
    update_claim_state(fp, "submitted", slurm_job_id="99999")
    update_claim_state(fp, "running")
    release_claim(fp)
    assert try_claim(fp) == True, "claim after release should succeed"
    release_claim(fp)
    print("[state_store] claim lifecycle ok")

    # Completed fingerprints are never claimable again
    append_result({**test_record, "fingerprint": "completed001"})
    assert try_claim("completed001") == False, "completed fingerprint should not claim"
    assert can_submit("completed001") == False
    print("[state_store] completed fingerprint dedupe ok")

    # Test promotion append + dedupe
    promo = {
        "schema_version": "1.0",
        "fingerprint": "promoted001",
        "tinker_status": "submitted",
        "timestamp": now,
    }
    append_promotion(promo)
    assert has_been_promoted("promoted001") == True
    assert has_been_promoted("missing001") == False
    print("[state_store] promotion tracking ok")

    # Test global best bootstrap + updates
    bootstrap = {
        **test_record,
        "fingerprint": "test000",
        "reward": 0.0,
        "status": "rejected",
    }
    r1 = {**test_record, "fingerprint": "test002", "reward": 0.10}
    r2 = {**test_record, "fingerprint": "test003", "reward": 0.20}
    r3 = {**test_record, "fingerprint": "test004", "reward": 0.05}
    update_global_best(bootstrap)
    best = read_global_best()
    assert best.best_fingerprint == "test000"
    assert best.best_reward == 0.0
    append_result(r1)
    append_result(r2)
    append_result(r3)
    update_global_best(r1)
    update_global_best(r2)
    updated = update_global_best(r3)  # should not update
    assert not updated
    best = read_global_best()
    assert best.best_reward == 0.20, f"expected 0.20 got {best.best_reward}"
    assert best.best_fingerprint == "test003"
    print(f"[state_store] global_best ok — best reward: {best.best_reward}")

    print("[state_store] ALL TESTS PASSED")
