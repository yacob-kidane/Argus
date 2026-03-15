"""
Central configuration for autoresearch-scale.
All paths and runtime settings live here.
Never hardcode paths elsewhere.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SHARED_DIR      = os.environ.get("AUTORESEARCH_SHARED_DIR",
                  "/mnt/sharefs/user13/autoresearch-scale")
STATE_DIR       = os.environ.get("AUTORESEARCH_STATE_DIR",
                  f"{SHARED_DIR}/state")
JOBS_DIR        = f"{SHARED_DIR}/jobs"
LOGS_DIR        = f"{SHARED_DIR}/logs"
REASONING_JSONL = str(Path(LOGS_DIR) / "agent_reasoning.jsonl")

TRAIN_PY_SOURCE = os.environ.get("AUTORESEARCH_TRAIN_PY",
                  "/mnt/sharefs/user13/karpathy-autoresearch/train.py")
PROGRAM_MD      = os.environ.get("AUTORESEARCH_PROGRAM_MD",
                  "/mnt/sharefs/user13/karpathy-autoresearch/program.md")

# ── State files ────────────────────────────────────────────────────────────
RESULTS_JSONL   = f"{STATE_DIR}/results.jsonl"
CLAIMED_JSON    = f"{STATE_DIR}/claimed.json"
GLOBAL_BEST_JSON = f"{STATE_DIR}/global_best.json"
PROMOTED_JSONL  = f"{STATE_DIR}/promoted.jsonl"
PROPOSAL_EVENTS_JSONL = f"{STATE_DIR}/proposal_events.jsonl"

# ── SLURM ──────────────────────────────────────────────────────────────────
SLURM_PARTITION      = os.environ.get("AUTORESEARCH_PARTITION", "priority")
SLURM_GPUS_PER_JOB   = int(os.environ.get("AUTORESEARCH_GPUS", "1"))
SLURM_CPUS_PER_GPU   = int(os.environ.get("AUTORESEARCH_CPUS", "8"))
SLURM_MEM_GB         = int(os.environ.get("AUTORESEARCH_MEM_GB", "64"))
SLURM_TIME_MINUTES   = int(os.environ.get("AUTORESEARCH_TIME_MINUTES", "8"))

# ── Orchestrator ───────────────────────────────────────────────────────────
MAX_CONCURRENT       = int(os.environ.get("AUTORESEARCH_MAX_CONCURRENT", "4"))
POLL_INTERVAL_SECS   = int(os.environ.get("AUTORESEARCH_POLL_SECS", "10"))
TTL_SECONDS          = int(os.environ.get("AUTORESEARCH_TTL_SECS", "480"))
MAX_CONTEXT_RESULTS  = int(os.environ.get("AUTORESEARCH_MAX_RESULTS", "20"))
PROPOSER_POLICY      = os.environ.get("PROPOSER_POLICY", "history")
CONDITION_ID         = os.environ.get("AUTORESEARCH_CONDITION_ID", "")

# ── Reward ─────────────────────────────────────────────────────────────────
REWARD_SIGNAL        = os.environ.get("AUTORESEARCH_REWARD_SIGNAL",
                       "val_improvement_per_tflop")

# ── Sandbox ────────────────────────────────────────────────────────────────
SANDBOX_MODE         = os.environ.get("AUTORESEARCH_SANDBOX_MODE", "off")
# off | verify | full

# ── API ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL         = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ── Tinker scale-up ────────────────────────────────────────────────────────
TINKER_THRESHOLD     = float(os.environ.get("AUTORESEARCH_TINKER_THRESHOLD", "0.5"))
TINKER_PROMOTION_THRESHOLD = float(
    os.environ.get("AUTORESEARCH_TINKER_PROMOTION_THRESHOLD", "0.03")
)
ENABLE_TINKER_PROMOTION = os.environ.get(
    "AUTORESEARCH_ENABLE_TINKER_PROMOTION", "1"
) == "1"

# ── Training output parsing ────────────────────────────────────────────────
# Keys expected in the structured block after "---" in train.py stdout
TRAIN_OUTPUT_KEYS = [
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
]

# ── Hyperparameter edit zone ───────────────────────────────────────────────
# The agent is ONLY allowed to edit lines in this section of train.py.
# write_result.py uses TRAIN_OUTPUT_KEYS to parse structured output.
HYPERPARAM_SECTION_START = "# Hyperparameters (edit these directly"
HYPERPARAM_SECTION_END   = "# ---------------------------------------------------------------------------\n# Setup"

# ── Sanity check ───────────────────────────────────────────────────────────
def validate():
    import sys
    errors = []
    if not os.path.exists(TRAIN_PY_SOURCE):
        errors.append(f"train.py not found at {TRAIN_PY_SOURCE}")
    if not os.path.exists(STATE_DIR):
        errors.append(f"state dir not found: {STATE_DIR}")
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set")
    if errors:
        for e in errors:
            print(f"[config] ERROR: {e}", file=sys.stderr)
        return False
    return True

if __name__ == "__main__":
    ok = validate()
    print("[config] OK" if ok else "[config] FAILED")
    import json
    settings = {k: v for k, v in globals().items()
                if k.isupper() and not k.startswith("_")
                and not k.endswith("KEY")}  # never print secrets
    print(json.dumps(settings, indent=2, default=str))
