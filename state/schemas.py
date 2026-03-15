"""
Canonical data schemas for autoresearch-scale.
Source of truth: SCHEMA.md
All modules must import from here — never define record shapes inline.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone
import time

SCHEMA_VERSION = "1.0"


@dataclass
class Hypothesis:
    """Output of proposal_engine.next_hypothesis(). See SCHEMA.md §1."""
    hypothesis: str
    diff: str
    predicted_effect: str
    risk: str                        # low | medium | high
    fingerprint: str                 # md5(diff)
    schema_version: str              = SCHEMA_VERSION
    parent_fingerprint: Optional[str] = None
    proposal_source: str             = "llm"
    proposer_policy: str             = ""
    condition_id: str                = ""
    notes: str                       = ""

    def __post_init__(self):
        assert self.risk in ("low", "medium", "high"), \
            f"risk must be low/medium/high, got: {self.risk}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class ExperimentRecord:
    """One completed experiment. Appended to results.jsonl. See SCHEMA.md §2."""
    fingerprint: str
    config: dict
    val_bpb: float
    reward: float
    status: str                      # kept | rejected | failed
    outcome: str                     # succeeded | failed | timed_out | cancelled
    node: str
    timestamp: str
    schema_version: str              = SCHEMA_VERSION
    hypothesis: str                  = ""
    hypothesis_fingerprint: str      = ""
    proposal_source: str             = ""
    proposer_policy: str             = ""
    condition_id: str                = ""
    proposal_notes: str              = ""
    runtime_seconds: float           = 0.0
    total_tflops_consumed: float     = 0.0
    artifacts_path: str              = ""
    slurm_job_id: str                = ""
    error_message: str               = ""
    peak_vram_mb: float              = 0.0
    mfu_percent: float               = 0.0
    total_tokens_M: float            = 0.0
    num_steps: int                   = 0
    num_params_M: float              = 0.0
    depth: int                       = 0
    parent_fingerprint: str          = ""

    def __post_init__(self):
        assert self.status in ("kept", "rejected", "failed"), \
            f"status must be kept/rejected/failed, got: {self.status}"
        assert self.outcome in ("succeeded", "failed", "timed_out", "cancelled"), \
            f"outcome must be succeeded/failed/timed_out/cancelled, got: {self.outcome}"
        if self.status in ("kept", "rejected"):
            assert isinstance(self.val_bpb, (int, float)) and self.val_bpb < 999, \
                f"kept/rejected records require valid val_bpb, got: {self.val_bpb}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRecord":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class ClaimEntry:
    """Active experiment reservation. Lives in claimed.json. See SCHEMA.md §3."""
    fingerprint: str
    claimed_by: str
    claim_timestamp: str             # ISO-8601 UTC
    claimed_at_ts: float             # Unix timestamp for TTL
    ttl_seconds: int
    state: str                       # proposed | submitted | running
    schema_version: str              = SCHEMA_VERSION
    slurm_job_id: str                = ""
    hypothesis_fingerprint: str      = ""
    notes: str                       = ""

    def __post_init__(self):
        assert self.state in ("proposed", "submitted", "running"), \
            f"state must be proposed/submitted/running, got: {self.state}"

    def is_expired(self) -> bool:
        return time.time() - self.claimed_at_ts > self.ttl_seconds

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimEntry":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class GlobalBestRegister:
    """Current best known experiment. Lives in global_best.json. See SCHEMA.md §4."""
    best_fingerprint: Optional[str]
    best_val_bpb: float
    best_reward: float
    best_config: dict
    updated_at: str
    schema_version: str              = SCHEMA_VERSION
    hypothesis: str                  = ""
    hypothesis_fingerprint: str      = ""
    slurm_job_id: str                = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def empty(cls) -> "GlobalBestRegister":
        return cls(
            best_fingerprint=None,
            best_val_bpb=999.0,
            best_reward=0.0,
            best_config={},
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalBestRegister":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class TrainingMetrics:
    """Parsed output from train.py stdout. See SCHEMA.md §5."""
    fingerprint: str
    outcome: str                     # succeeded | failed | timed_out | cancelled
    timestamp: str
    schema_version: str              = SCHEMA_VERSION
    val_bpb: float                   = 999.0
    runtime_seconds: float           = 0.0
    total_tflops_consumed: float     = 0.0
    peak_vram_mb: float              = 0.0
    mfu_percent: float               = 0.0
    total_tokens_M: float            = 0.0
    num_steps: int                   = 0
    num_params_M: float              = 0.0
    depth: int                       = 0
    error_message: str               = ""

    def __post_init__(self):
        assert self.outcome in ("succeeded", "failed", "timed_out", "cancelled"), \
            f"outcome invalid: {self.outcome}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMetrics":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def parse_stdout(cls, stdout: str, fingerprint: str) -> "TrainingMetrics":
        """
        Parse train.py stdout into TrainingMetrics.
        Looks for structured block after '---' line.
        Also extracts 'Estimated FLOPs per token' for tflops calculation.
        """
        now = datetime.now(timezone.utc).isoformat()

        if "FAIL" in stdout:
            return cls(fingerprint=fingerprint, outcome="failed",
                       timestamp=now, error_message="training fast-fail")

        # Parse structured block after ---
        metrics = {}
        in_block = False
        flops_per_token = None

        for line in stdout.splitlines():
            # Extract flops per token (printed early in stdout)
            if "Estimated FLOPs per token:" in line:
                try:
                    flops_per_token = float(line.split(":")[-1].strip())
                except ValueError:
                    pass

            if line.strip() == "---" or line.rstrip().endswith("---"):
                in_block = True
                continue

            if in_block and ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    pass

        if "val_bpb" not in metrics:
            return cls(fingerprint=fingerprint, outcome="failed",
                       timestamp=now, error_message="val_bpb not found in output")

        # Compute tflops
        total_tflops = 0.0
        if flops_per_token and metrics.get("total_tokens_M", 0) > 0:
            total_tokens = metrics["total_tokens_M"] * 1e6
            total_tflops = (total_tokens * flops_per_token) / 1e12

        return cls(
            fingerprint=fingerprint,
            outcome="succeeded",
            timestamp=now,
            val_bpb=metrics.get("val_bpb", 999.0),
            runtime_seconds=metrics.get("training_seconds", 0.0),
            total_tflops_consumed=total_tflops,
            peak_vram_mb=metrics.get("peak_vram_mb", 0.0),
            mfu_percent=metrics.get("mfu_percent", 0.0),
            total_tokens_M=metrics.get("total_tokens_M", 0.0),
            num_steps=int(metrics.get("num_steps", 0)),
            num_params_M=metrics.get("num_params_M", 0.0),
            depth=int(metrics.get("depth", 0)),
        )


if __name__ == "__main__":
    now = datetime.now(timezone.utc).isoformat()

    # Hypothesis
    h = Hypothesis(
        hypothesis="Reduce matrix LR",
        diff="MATRIX_LR = 0.02",
        predicted_effect="More stable training",
        risk="low",
        fingerprint="abc123",
    )
    assert Hypothesis.from_dict(h.to_dict()).fingerprint == "abc123"
    print("[schemas] Hypothesis ok")

    # ExperimentRecord
    r = ExperimentRecord(
        fingerprint="abc123",
        config={"MATRIX_LR": 0.02},
        val_bpb=0.91,
        reward=0.05,
        status="kept",
        outcome="succeeded",
        node="node-01",
        timestamp=now,
    )
    assert ExperimentRecord.from_dict(r.to_dict()).val_bpb == 0.91
    print("[schemas] ExperimentRecord ok")

    # ClaimEntry
    c = ClaimEntry(
        fingerprint="abc123",
        claimed_by="orchestrator",
        claim_timestamp=now,
        claimed_at_ts=time.time(),
        ttl_seconds=480,
        state="submitted",
    )
    assert not c.is_expired()
    print("[schemas] ClaimEntry ok")

    # GlobalBestRegister
    g = GlobalBestRegister.empty()
    assert g.best_val_bpb == 999.0
    assert GlobalBestRegister.from_dict(g.to_dict()).best_reward == 0.0
    print("[schemas] GlobalBestRegister ok")

    # TrainingMetrics.parse_stdout
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
    m = TrainingMetrics.parse_stdout(fake_stdout, "abc123")
    assert m.outcome == "succeeded"
    assert m.val_bpb == 0.8734
    assert m.total_tflops_consumed > 0
    print(f"[schemas] TrainingMetrics ok — tflops={m.total_tflops_consumed:.3f}")

    # FAIL detection
    m2 = TrainingMetrics.parse_stdout("FAIL\n", "abc123")
    assert m2.outcome == "failed"
    print("[schemas] TrainingMetrics FAIL detection ok")

    print("[schemas] ALL CONTRACT TESTS PASSED")
