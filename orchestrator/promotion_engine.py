"""
Minimal promotion predicate for Tinker scale-up.
"""
from __future__ import annotations


def relative_improvement(baseline_val_bpb: float, current_val_bpb: float) -> float:
    """
    Compute relative improvement where lower val_bpb is better.
    Returns 0.0 if the baseline is missing or invalid.
    """
    if baseline_val_bpb <= 0:
        return 0.0
    return (baseline_val_bpb - current_val_bpb) / baseline_val_bpb


def should_promote(
    record: dict,
    baseline_val_bpb: float | None,
    threshold: float,
) -> tuple[bool, str]:
    """
    Promote iff the run succeeded and beat the baseline by more than threshold.
    """
    if record.get("outcome") != "succeeded":
        return False, "outcome_not_succeeded"

    if baseline_val_bpb is None:
        return False, "baseline_missing"

    current_val_bpb = float(record.get("val_bpb", float("inf")))
    improvement = relative_improvement(baseline_val_bpb, current_val_bpb)
    if improvement > threshold:
        return True, "relative_improvement_gt_threshold"
    return False, "below_threshold"


if __name__ == "__main__":
    ok, reason = should_promote(
        {"outcome": "succeeded", "val_bpb": 0.95},
        baseline_val_bpb=1.0,
        threshold=0.03,
    )
    assert ok and reason == "relative_improvement_gt_threshold"

    ok, reason = should_promote(
        {"outcome": "failed", "val_bpb": 0.90},
        baseline_val_bpb=1.0,
        threshold=0.03,
    )
    assert not ok and reason == "outcome_not_succeeded"

    ok, reason = should_promote(
        {"outcome": "succeeded", "val_bpb": 0.99},
        baseline_val_bpb=1.0,
        threshold=0.03,
    )
    assert not ok and reason == "below_threshold"

    print("[promotion_engine] ALL TESTS PASSED")
