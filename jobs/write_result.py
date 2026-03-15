"""
Result writer — called at end of each SLURM job.
Parses train.py stdout, computes reward, writes to shared state.

Usage:
    python3 jobs/write_result.py \
        --fingerprint <md5_of_diff> \
        --stdout-file <path_to_stdout_log> \
        --hypothesis <quoted_string> \
        --hypothesis-fingerprint <str> \
        --config-json <json_string> \
        --node <node_name> \
        --job-id <slurm_job_id>
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from orchestrator import promotion_engine, tinker_scaleup
from state.schemas import ExperimentRecord, TrainingMetrics
from state import state_store


def compute_reward(metrics: TrainingMetrics) -> float:
    """
    Canonical reward computation. Single source of truth.
    reward = (baseline_val_bpb - current_val_bpb) / total_tflops_consumed
    Fallback: reward = baseline_val_bpb - current_val_bpb
    See SCHEMA.md §6.
    """
    if metrics.outcome != "succeeded":
        return 0.0

    baseline = state_store.get_baseline_val_bpb()
    if baseline is None:
        # First experiment — it becomes the baseline
        baseline = metrics.val_bpb

    improvement = baseline - metrics.val_bpb

    if metrics.total_tflops_consumed > 0:
        return improvement / metrics.total_tflops_consumed
    else:
        # Fallback: no tflops available
        return improvement


def _load_hypothesis_metadata(fingerprint: str) -> dict:
    """Load hypothesis metadata saved alongside the job, if present."""
    hypothesis_path = os.path.join(config.JOBS_DIR, fingerprint, "hypothesis.json")
    if not os.path.exists(hypothesis_path):
        return {}
    try:
        with open(hypothesis_path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def ingest(
    fingerprint: str,
    stdout_file: str,
    hypothesis: str = "",
    hypothesis_fingerprint: str = "",
    config_json: str = "{}",
    node: str = "unknown",
    job_id: str = "",
) -> ExperimentRecord:
    """
    Parse stdout, compute reward, write result, update global best.
    Returns the ExperimentRecord that was written.
    """
    now = datetime.now(timezone.utc).isoformat()
    baseline_before = state_store.get_baseline_val_bpb()

    # Read stdout
    stdout = ""
    if os.path.exists(stdout_file):
        with open(stdout_file, "r") as f:
            stdout = f.read()
    else:
        stdout = "FAIL\nerror: stdout file not found"

    # Parse training metrics
    metrics = TrainingMetrics.parse_stdout(stdout, fingerprint)

    # Compute reward
    reward = compute_reward(metrics)

    # Determine status
    if metrics.outcome != "succeeded":
        status = "failed"
    elif reward > 0:
        status = "kept"
    else:
        status = "rejected"

    # Parse config
    try:
        exp_config = json.loads(config_json)
    except json.JSONDecodeError:
        exp_config = {}
    hypothesis_meta = _load_hypothesis_metadata(fingerprint)
    hypothesis = hypothesis or hypothesis_meta.get("hypothesis", "")
    hypothesis_fingerprint = (
        hypothesis_fingerprint
        or hypothesis_meta.get("fingerprint", "")
    )

    # Build record
    record = ExperimentRecord(
        fingerprint=fingerprint,
        config=exp_config,
        val_bpb=metrics.val_bpb,
        reward=reward,
        status=status,
        outcome=metrics.outcome,
        node=node,
        timestamp=now,
        hypothesis=hypothesis,
        hypothesis_fingerprint=hypothesis_fingerprint,
        proposal_source=hypothesis_meta.get("proposal_source", ""),
        proposer_policy=hypothesis_meta.get("proposer_policy", ""),
        condition_id=hypothesis_meta.get("condition_id", ""),
        proposal_notes=hypothesis_meta.get("notes", ""),
        runtime_seconds=metrics.runtime_seconds,
        total_tflops_consumed=metrics.total_tflops_consumed,
        peak_vram_mb=metrics.peak_vram_mb,
        mfu_percent=metrics.mfu_percent,
        total_tokens_M=metrics.total_tokens_M,
        num_steps=metrics.num_steps,
        num_params_M=metrics.num_params_M,
        depth=metrics.depth,
        slurm_job_id=job_id,
        error_message=metrics.error_message,
    )

    # Write to shared state
    record_dict = record.to_dict()
    state_store.append_result(record_dict)
    state_store.update_global_best(record_dict)
    state_store.release_claim(fingerprint)
    _maybe_promote(record_dict, baseline_before)

    return record


def _maybe_promote(record: dict, baseline_before: float | None) -> None:
    """
    Fire a minimal, idempotent Tinker promotion after the result is durable.
    """
    if not config.ENABLE_TINKER_PROMOTION:
        return

    fingerprint = record.get("fingerprint", "")
    if not fingerprint or state_store.has_been_promoted(fingerprint):
        return

    should_promote, reason = promotion_engine.should_promote(
        record=record,
        baseline_val_bpb=baseline_before,
        threshold=config.TINKER_PROMOTION_THRESHOLD,
    )
    if not should_promote:
        return

    current_val_bpb = float(record.get("val_bpb", float("inf")))
    relative = promotion_engine.relative_improvement(baseline_before, current_val_bpb)
    tinker_status = "failed"
    tinker_request_id = ""

    try:
        tinker_response = tinker_scaleup.launch_tinker_scaleup(record)
        raw_status = str(tinker_response.get("status", "failed"))
        if raw_status == "succeeded":
            tinker_status = "submitted"
        elif raw_status.startswith("skipped"):
            tinker_status = "failed"
        else:
            tinker_status = "failed"
        tinker_request_id = str(tinker_response.get("request_id", ""))
    except Exception as exc:
        raw_status = f"error: {exc}"

    promotion_record = {
        "schema_version": "1.0",
        "fingerprint": fingerprint,
        "slurm_job_id": record.get("slurm_job_id", ""),
        "val_bpb": record.get("val_bpb"),
        "reward": record.get("reward"),
        "baseline_val_bpb": baseline_before,
        "relative_improvement": relative,
        "promotion_reason": reason,
        "promotion_threshold": config.TINKER_PROMOTION_THRESHOLD,
        "tinker_status": tinker_status,
        "tinker_request_id": tinker_request_id,
        "tinker_raw_status": raw_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    state_store.append_promotion(promotion_record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprint",           required=True)
    parser.add_argument("--stdout-file",           required=True)
    parser.add_argument("--hypothesis",            default="")
    parser.add_argument("--hypothesis-fingerprint", default="")
    parser.add_argument("--config-json",           default="{}")
    parser.add_argument("--node",                  default="unknown")
    parser.add_argument("--job-id",                default="")
    args = parser.parse_args()

    state_store.ensure_state_dirs()

    record = ingest(
        fingerprint=args.fingerprint,
        stdout_file=args.stdout_file,
        hypothesis=args.hypothesis,
        hypothesis_fingerprint=args.hypothesis_fingerprint,
        config_json=args.config_json,
        node=args.node,
        job_id=args.job_id,
    )

    print(f"[write_result] status={record.status} "
          f"outcome={record.outcome} "
          f"val_bpb={record.val_bpb:.6f} "
          f"reward={record.reward:.6f}")


if __name__ == "__main__":
    # Smoke test mode
    if "--test" in sys.argv:
        import tempfile

        state_store.ensure_state_dirs()
        open(config.RESULTS_JSONL, "w").close()
        open(config.PROMOTED_JSONL, "w").close()

        fake_stdout = """
Estimated FLOPs per token: 1.234e+10
step 00487 (100.0%) | loss: 0.912345
---
val_bpb:          0.8734
training_seconds: 301.2
total_seconds:    312.4
peak_vram_mb:     18432.0
mfu_percent:      47.23
total_tokens_M:   156.4
num_steps:        487
num_params_M:     85.2
depth:            8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log",
                                         delete=False) as f:
            f.write(fake_stdout)
            tmp_path = f.name

        baseline_stdout = fake_stdout.replace("0.8734", "0.9200")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log",
                                         delete=False) as f:
            f.write(baseline_stdout)
            baseline_path = f.name

        original_threshold = config.TINKER_PROMOTION_THRESHOLD
        original_enable = config.ENABLE_TINKER_PROMOTION
        config.TINKER_PROMOTION_THRESHOLD = 0.0
        config.ENABLE_TINKER_PROMOTION = True

        baseline_fp = "writeresult_test_baseline"
        state_store.try_claim(baseline_fp)
        baseline_record = ingest(
            fingerprint=baseline_fp,
            stdout_file=baseline_path,
            hypothesis="Baseline run",
            node="test-node",
            job_id="00000",
        )

        # Register a fake claim first
        fp = "writeresult_test_001"
        state_store.try_claim(fp)

        record = ingest(
            fingerprint=fp,
            stdout_file=tmp_path,
            hypothesis="Test: reduce matrix LR",
            node="test-node",
            job_id="00000",
        )

        os.unlink(tmp_path)
        os.unlink(baseline_path)
        config.TINKER_PROMOTION_THRESHOLD = original_threshold
        config.ENABLE_TINKER_PROMOTION = original_enable

        assert record.outcome == "succeeded", f"expected succeeded: {record.outcome}"
        assert record.val_bpb == 0.8734,      f"val_bpb mismatch: {record.val_bpb}"
        assert record.status in ("kept", "rejected")
        assert record.num_steps == 487
        assert baseline_record.val_bpb == 0.9200

        # Verify it's in results.jsonl
        results = state_store.read_results()
        assert any(r["fingerprint"] == fp for r in results)

        promotions = state_store.read_promotions()
        assert any(p["fingerprint"] == fp for p in promotions)

        # Verify claim was released
        assert fp not in state_store.list_active_fingerprints()

        print(f"[write_result] test ok — "
              f"val_bpb={record.val_bpb} "
              f"reward={record.reward:.6f} "
              f"status={record.status}")
        print("[write_result] ALL TESTS PASSED")
    else:
        main()
