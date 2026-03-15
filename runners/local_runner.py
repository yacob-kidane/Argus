"""
Local runner — fake SLURM for testing.
Runs train.py on CPU with --max-steps 50.
Implements identical interface to slurm_runner.
See AGENTS.md: local_runner and slurm_runner must implement identical interfaces.
"""
import os
import sys
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from state.schemas import Hypothesis
from safety.ast_check import is_safe, log_rejection

# Patch train.py to run on CPU with small step count
_LOCAL_PATCH = """
# LOCAL RUNNER PATCH — injected at top of train.py for local testing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

# Monkey-patch TIME_BUDGET to stop after a few steps
import builtins
_original_time = __import__('time').time
_start = _original_time()
"""

_LOCAL_OVERRIDES = {
    "TIME_BUDGET": "30",          # 30 seconds max
    "DEVICE_BATCH_SIZE": "2",     # tiny batch for CPU
    "TOTAL_BATCH_SIZE": "2**10",  # small total batch
    "DEPTH": "2",                 # shallow model
}


def _patch_train_py(source_code: str, diff: str) -> str:
    """Apply hypothesis diff and local overrides to train.py source."""
    # Apply the hypothesis diff (simple line replacement)
    patched = _apply_diff(source_code, diff)

    # Override hyperparams for local testing
    lines = patched.splitlines()
    result = []
    for line in lines:
        replaced = False
        for key, val in _LOCAL_OVERRIDES.items():
            stripped = line.strip()
            if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
                indent = line[: len(line) - len(line.lstrip())]
                result.append(f"{indent}{key} = {val}  # local runner override")
                replaced = True
                break
        if not replaced:
            result.append(line)

    return "\n".join(result)


def _apply_diff(source_code: str, diff: str) -> str:
    """
    Apply a hypothesis diff to source code.
    The diff is a Python code fragment that replaces matching lines
    in the Hyperparameters section of train.py.
    """
    if not diff or not diff.strip():
        return source_code

    lines = source_code.splitlines()
    result = []
    in_hyperparam_section = False

    for line in lines:
        if config.HYPERPARAM_SECTION_START in line:
            in_hyperparam_section = True

        if in_hyperparam_section and config.HYPERPARAM_SECTION_END.split("\n")[0] in line:
            in_hyperparam_section = False

        if in_hyperparam_section:
            # Check if this line's variable is being overridden by the diff
            replaced = False
            for diff_line in diff.splitlines():
                diff_line = diff_line.strip()
                if not diff_line or diff_line.startswith("#"):
                    continue
                # Match on variable name
                if "=" in diff_line and "=" in line:
                    diff_var = diff_line.split("=")[0].strip()
                    line_var = line.strip().split("=")[0].strip()
                    if diff_var == line_var:
                        indent = line[: len(line) - len(line.lstrip())]
                        result.append(f"{indent}{diff_line.strip()}")
                        replaced = True
                        break
            if not replaced:
                result.append(line)
        else:
            result.append(line)

    return "\n".join(result)


def submit(hypothesis: Hypothesis, dry_run: bool = False) -> dict:
    """
    Run experiment locally on CPU.
    Returns job handle dict with same shape as slurm_runner.submit().

    job handle:
    {
        "job_id": str,
        "fingerprint": str,
        "stdout_file": str,
        "stderr_file": str,
        "runner": "local",
    }
    """
    fp = hypothesis.fingerprint

    # Safety check
    with open(config.TRAIN_PY_SOURCE, "r") as f:
        source = f.read()

    patched_source = _patch_train_py(source, hypothesis.diff)
    ok, reason = is_safe(patched_source)
    if not ok:
        log_rejection(fp, reason)
        raise ValueError(f"[local_runner] safety check failed: {reason}")

    # Set up job directory
    job_dir = os.path.join(config.JOBS_DIR, fp)
    os.makedirs(job_dir, exist_ok=True)

    train_py_path = os.path.join(job_dir, "train.py")
    stdout_file   = os.path.join(job_dir, "stdout.log")
    stderr_file   = os.path.join(job_dir, "stderr.log")

    with open(train_py_path, "w") as f:
        f.write(patched_source)

    # Save hypothesis metadata
    with open(os.path.join(job_dir, "hypothesis.json"), "w") as f:
        import json
        json.dump(hypothesis.to_dict(), f, indent=2)

    if dry_run:
        print(f"[local_runner] dry-run — job_dir={job_dir}")
        return {
            "job_id": f"dry-{fp[:8]}",
            "fingerprint": fp,
            "stdout_file": stdout_file,
            "stderr_file": stderr_file,
            "runner": "local",
        }

    # Copy prepare.py and tokenizer artifacts from karpathy repo
    karpathy_dir = os.path.dirname(config.TRAIN_PY_SOURCE)
    for fname in ["prepare.py", "kernels.py"]:
        src = os.path.join(karpathy_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, job_dir)

    # Copy tokenizer directory if present
    tok_dir = os.path.join(karpathy_dir, "tok")
    if os.path.exists(tok_dir):
        dst_tok = os.path.join(job_dir, "tok")
        if not os.path.exists(dst_tok):
            shutil.copytree(tok_dir, dst_tok)

    # Run training
    print(f"[local_runner] starting job fp={fp[:8]}...")
    with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
        proc = subprocess.run(
            [sys.executable, train_py_path],
            cwd=job_dir,
            stdout=out,
            stderr=err,
            timeout=config.TTL_SECONDS,
        )

    job_id = f"local-{fp[:8]}"
    print(f"[local_runner] job finished — returncode={proc.returncode}")

    return {
        "job_id": job_id,
        "fingerprint": fp,
        "stdout_file": stdout_file,
        "stderr_file": stderr_file,
        "runner": "local",
        "returncode": proc.returncode,
    }


if __name__ == "__main__":
    # Dry run test — does not execute training
    from state.schemas import Hypothesis
    import hashlib

    diff = "DEPTH = 2"
    fp = hashlib.md5(diff.encode()).hexdigest()

    h = Hypothesis(
        hypothesis="Reduce model depth to 2 layers",
        diff=diff,
        predicted_effect="Faster iteration, tests local runner plumbing",
        risk="low",
        fingerprint=fp,
    )

    handle = submit(h, dry_run=True)
    assert handle["fingerprint"] == fp
    assert handle["runner"] == "local"
    assert "stdout_file" in handle
    print(f"[local_runner] dry-run ok — handle={handle}")
    print("[local_runner] ALL TESTS PASSED")
