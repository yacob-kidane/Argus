"""
SLURM runner — submits experiments as sbatch jobs.
Implements identical interface to local_runner.
All sbatch calls go through this module — never call sbatch directly elsewhere.
See AGENTS.md hard rules.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from state.schemas import Hypothesis
from state import state_store
from safety.ast_check import is_safe, log_rejection


def _apply_diff(source_code: str, diff: str) -> str:
    """
    Apply hypothesis diff to train.py source.
    Replaces matching variable assignments in the hyperparameter section.
    Identical logic to local_runner._apply_diff — kept in sync manually.
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
            replaced = False
            for diff_line in diff.splitlines():
                diff_line = diff_line.strip()
                if not diff_line or diff_line.startswith("#"):
                    continue
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


def _render_job_script(
    job_dir: str,
    train_py_path: str,
    stdout_file: str,
    stderr_file: str,
    fingerprint: str,
    hypothesis: str,
    hypothesis_fingerprint: str,
    exp_config_json: str,
) -> str:
    """Render the sbatch job script for this experiment."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    write_result_path = os.path.join(
        repo_root, "jobs", "write_result.py"
    )

    return f"""#!/bin/bash
#SBATCH --job-name=autoexp-{fingerprint[:8]}
#SBATCH --partition={config.SLURM_PARTITION}
#SBATCH --gpus={config.SLURM_GPUS_PER_JOB}
#SBATCH --cpus-per-gpu={config.SLURM_CPUS_PER_GPU}
#SBATCH --mem={config.SLURM_MEM_GB}G
#SBATCH --time=00:{config.SLURM_TIME_MINUTES:02d}:00
#SBATCH --output={stdout_file}
#SBATCH --error={stderr_file}

set -e

# CUDA library paths for H100 nodes
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
export USE_FLASH_ATTN=0
export USE_TORCH_COMPILE=0

# Triton cache on local NVMe
export TRITON_CACHE_DIR=/tmp/triton_cache_{fingerprint[:8]}
mkdir -p $TRITON_CACHE_DIR

# Use fast local NVMe scratch for all intermediate files
WORKDIR=/tmp/exp_{fingerprint[:16]}
mkdir -p $WORKDIR

# Copy job files to local scratch
cp {train_py_path} $WORKDIR/train.py

# Copy karpathy repo support files
KARPATHY_DIR={os.path.dirname(config.TRAIN_PY_SOURCE)}
for f in prepare.py kernels.py pyproject.toml uv.lock; do
    [ -f $KARPATHY_DIR/$f ] && cp $KARPATHY_DIR/$f $WORKDIR/
done
[ -d $KARPATHY_DIR/tok ] && cp -r $KARPATHY_DIR/tok $WORKDIR/tok

# Run training from local scratch — all intermediate writes go to /tmp
cd $WORKDIR
echo "[job] fingerprint={fingerprint}"
echo "[job] workdir=$WORKDIR"
echo "[job] stdout_file={stdout_file}"
echo "[job] stderr_file={stderr_file}"
echo "[job] USE_FLASH_ATTN=$USE_FLASH_ATTN USE_TORCH_COMPILE=$USE_TORCH_COMPILE"

# Source env
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# Load shared runtime secrets/config for both training and promotion hooks.
if [ -f "{config.SHARED_DIR}/.env" ]; then
    set -a
    . "{config.SHARED_DIR}/.env"
    set +a
fi
export AUTORESEARCH_SHARED_DIR="{config.SHARED_DIR}"
export AUTORESEARCH_STATE_DIR="{config.STATE_DIR}"
export AUTORESEARCH_CONDITION_ID="{config.CONDITION_ID}"
export PROPOSER_POLICY="{config.PROPOSER_POLICY}"
echo "[job] env OPENAI_API_KEY=$([ -n "$OPENAI_API_KEY" ] && echo set || echo missing) TINKER_API_KEY=$([ -n "$TINKER_API_KEY" ] && echo set || echo missing)"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Install tinker using Python 3.11 from pyenv
export PYENV_ROOT="/mnt/sharefs/user13/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)"
pyenv local 3.11.9 2>/dev/null || true

# Run via uv — stdout/stderr are already captured by SLURM file paths above.
uv run train.py

# Write result to shared Lustre (one small write)
python3 {write_result_path} \\
    --fingerprint "{fingerprint}" \\
    --stdout-file "{stdout_file}" \\
    --hypothesis "{hypothesis.replace('"', "'")}" \\
    --hypothesis-fingerprint "{hypothesis_fingerprint}" \\
    --config-json '{exp_config_json}' \\
    --node "$SLURMD_NODENAME" \\
    --job-id "$SLURM_JOB_ID"

# Clean up local scratch
rm -rf $WORKDIR
"""


def submit(hypothesis: Hypothesis, dry_run: bool = False) -> dict:
    """
    Submit experiment as a SLURM batch job.
    Returns job handle dict with same shape as local_runner.submit().

    job handle:
    {
        "job_id": str,
        "fingerprint": str,
        "stdout_file": str,
        "stderr_file": str,
        "runner": "slurm",
    }
    """
    fp = hypothesis.fingerprint

    # Read and patch train.py
    with open(config.TRAIN_PY_SOURCE, "r") as f:
        source = f.read()

    patched_source = _apply_diff(source, hypothesis.diff)

    # Safety check before touching cluster
    ok, reason = is_safe(patched_source)
    if not ok:
        log_rejection(fp, reason)
        raise ValueError(f"[slurm_runner] safety check failed: {reason}")

    # Set up job directory on shared Lustre
    job_dir = os.path.join(config.JOBS_DIR, fp)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    train_py_path = os.path.join(job_dir, "train.py")
    stdout_file   = os.path.join(config.LOGS_DIR, f"{fp[:16]}.out")
    stderr_file   = os.path.join(config.LOGS_DIR, f"{fp[:16]}.err")
    script_path   = os.path.join(job_dir, "job.sh")

    # Write patched train.py
    with open(train_py_path, "w") as f:
        f.write(patched_source)

    # Save hypothesis metadata
    with open(os.path.join(job_dir, "hypothesis.json"), "w") as f:
        json.dump(hypothesis.to_dict(), f, indent=2)

    # Extract config snapshot from diff
    exp_config = {}
    for line in hypothesis.diff.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, val = line.partition("=")
            exp_config[key.strip()] = val.strip()
    exp_config_json = json.dumps(exp_config)

    # Render job script
    script = _render_job_script(
        job_dir=job_dir,
        train_py_path=train_py_path,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        fingerprint=fp,
        hypothesis=hypothesis.hypothesis,
        hypothesis_fingerprint=hypothesis.fingerprint,
        exp_config_json=exp_config_json,
    )

    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    if dry_run:
        print(f"[slurm_runner] dry-run — script written to {script_path}")
        print("--- job script ---")
        print(script)
        print("--- end ---")
        return {
            "job_id": f"dry-{fp[:8]}",
            "fingerprint": fp,
            "stdout_file": stdout_file,
            "stderr_file": stderr_file,
            "runner": "slurm",
            "script_path": script_path,
        }

    # Submit to SLURM
    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[slurm_runner] sbatch failed: {result.stderr.strip()}"
        )

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r"(\d+)", result.stdout)
    job_id = match.group(1) if match else "unknown"

    # Update claim state
    state_store.update_claim_state(fp, "submitted", slurm_job_id=job_id)

    print(f"[slurm_runner] submitted job_id={job_id} fp={fp[:8]}")

    return {
        "job_id": job_id,
        "fingerprint": fp,
        "stdout_file": stdout_file,
        "stderr_file": stderr_file,
        "runner": "slurm",
        "script_path": script_path,
    }


def cancel(job_id: str) -> None:
    """Cancel a running SLURM job."""
    subprocess.run(["scancel", job_id], check=True)
    print(f"[slurm_runner] cancelled job_id={job_id}")


def get_active_job_count() -> int:
    """Return number of currently running/pending jobs for this user."""
    result = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", "user13"),
         "-h", "--format=%i"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return 0
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    return len(lines)


if __name__ == "__main__":
    import hashlib
    from state.schemas import Hypothesis

    diff = "DEPTH = 4\nMATRIX_LR = 0.03"
    fp = hashlib.md5(diff.encode()).hexdigest()

    h = Hypothesis(
        hypothesis="Reduce depth to 4 and lower matrix LR",
        diff=diff,
        predicted_effect="Smaller model, more conservative LR",
        risk="low",
        fingerprint=fp,
    )

    # Dry run only — never touch real SLURM in tests
    handle = submit(h, dry_run=True)
    assert handle["fingerprint"] == fp
    assert handle["runner"] == "slurm"
    assert os.path.exists(handle["script_path"])

    # Verify job script content
    with open(handle["script_path"]) as f:
        script_content = f.read()
    assert "#SBATCH --gpus=1" in script_content
    assert f"autoexp-{fp[:8]}" in script_content
    assert "write_result.py" in script_content
    assert "WORKDIR=/tmp" in script_content
    print("[slurm_runner] job script ok")

    # Verify active job count works
    count = get_active_job_count()
    print(f"[slurm_runner] active jobs: {count}")

    print("[slurm_runner] ALL TESTS PASSED")
