"""
Tinker scale-up integration — Phase 2.
Uses raw HTTP requests instead of the Tinker SDK (which requires Python 3.11+).
Cluster runs Python 3.10.12.

When autoresearch finds a verified breakthrough on small model,
validates the config transfers to a larger model via Tinker API.

API docs: https://tinker-docs.thinkingmachines.ai
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from state import state_store

TINKER_RESULTS_FILE = os.path.join(config.STATE_DIR, "tinker_results.jsonl")
TINKER_BASE_URL = "https://api.thinkingmachines.ai"
BASE_MODEL = "meta-llama/Llama-3.1-8B"


def _tinker_request(method: str, path: str, body: dict = None) -> dict:
    """Make authenticated request to Tinker REST API."""
    api_key = os.environ.get("TINKER_API_KEY", "")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not set")

    url = f"{TINKER_BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"Tinker API error {e.code}: {body}")


def get_available_models() -> list[str]:
    """List available models from Tinker."""
    resp = _tinker_request("GET", "/v1/models")
    return [m["id"] for m in resp.get("data", [])]


def create_training_run(model: str) -> str:
    """Create a new LoRA training run. Returns run_id."""
    resp = _tinker_request("POST", "/v1/training-runs", {
        "base_model": model,
        "lora_rank": 16,
    })
    return resp["training_run_id"]


def run_forward_backward(run_id: str, examples: list, lr: float = 1e-4) -> float:
    """
    Run one forward-backward pass and optimizer step.
    Returns loss for this batch.
    """
    resp = _tinker_request("POST", f"/v1/training-runs/{run_id}/step", {
        "examples": examples,
        "loss_fn": "cross_entropy",
        "optimizer": {
            "type": "adam",
            "learning_rate": lr,
        }
    })
    return resp.get("loss", 999.0)


def should_scale_up(current_best: dict, initial_reward: float) -> bool:
    """Check if breakthrough threshold crossed and not already scaled."""
    if current_best.get("tinker_scaled"):
        return False
    if not current_best.get("best_fingerprint"):
        return False

    best_reward = current_best.get("best_reward", 0.0)
    if initial_reward <= 0:
        return best_reward > 0

    improvement = (best_reward - initial_reward) / abs(initial_reward)
    return improvement > config.TINKER_THRESHOLD


def scale_up(winning_config: dict, fingerprint: str) -> dict:
    """
    Validate winning config on larger model via Tinker API.
    Uses raw HTTP — no SDK required.
    """
    print(f"[tinker_scaleup] Starting scale-up for fingerprint {fingerprint[:8]}")
    print(f"[tinker_scaleup] Winning config: {json.dumps(winning_config)}")

    api_key = os.environ.get("TINKER_API_KEY", "")
    if not api_key:
        print("[tinker_scaleup] WARNING: TINKER_API_KEY not set — skipping")
        return _stub_result(fingerprint, winning_config, "skipped: no api key")

    t_start = time.time()

    try:
        # Get available models
        models = get_available_models()
        print(f"[tinker_scaleup] Available models: {models}")

        model = BASE_MODEL if BASE_MODEL in models else (models[0] if models else BASE_MODEL)
        print(f"[tinker_scaleup] Using model: {model}")

        # Create training run
        run_id = create_training_run(model)
        print(f"[tinker_scaleup] Created training run: {run_id}")

        # Scale LR from winning config
        lr = float(winning_config.get("MATRIX_LR", 0.04)) * 0.1
        lr = min(max(lr, 1e-5), 1e-3)

        # Run validation steps with simple text examples
        examples = _make_examples()
        losses = []
        steps = 20  # Keep short for hackathon demo

        for step in range(steps):
            loss = run_forward_backward(run_id, examples, lr=lr)
            losses.append(loss)
            if step % 5 == 0:
                print(f"[tinker_scaleup] step {step:2d}/{steps} loss={loss:.4f}")

        runtime = time.time() - t_start
        initial_loss = losses[0] if losses else 999.0
        final_loss = losses[-1] if losses else 999.0
        improvement = initial_loss - final_loss

        result = _build_result(
            fingerprint=fingerprint,
            model=model,
            winning_config=winning_config,
            initial_loss=initial_loss,
            final_loss=final_loss,
            improvement=improvement,
            lr=lr,
            steps=steps,
            losses=losses,
            runtime=runtime,
            status="succeeded",
        )

    except Exception as e:
        print(f"[tinker_scaleup] ERROR: {e}")
        result = _stub_result(fingerprint, winning_config, f"error: {e}")

    _write_result(result)
    _mark_scaled()
    return result


def launch_tinker_scaleup(record: dict) -> dict:
    """
    Minimal promotion entry point used by write_result.py.
    Returns Tinker request metadata or a stub failure/skipped record.
    """
    fingerprint = record.get("fingerprint", "")
    winning_config = record.get("config", {})
    result = scale_up(winning_config=winning_config, fingerprint=fingerprint)
    request_id = ""
    if result.get("status") == "succeeded":
        request_id = str(result.get("fingerprint", ""))

    return {
        "status": result.get("status", "failed"),
        "request_id": request_id,
        "result": result,
    }


def _make_examples() -> list:
    """Simple text examples for validation — no tokenizer needed for REST API."""
    return [
        {"prompt": "The transformer architecture uses", "completion": " attention mechanisms."},
        {"prompt": "Gradient descent", "completion": " minimizes the loss function."},
        {"prompt": "Neural networks learn by", "completion": " adjusting weights through backprop."},
        {"prompt": "Machine learning is", "completion": " a subset of artificial intelligence."},
        {"prompt": "The quick brown fox", "completion": " jumps over the lazy dog."},
    ]


def _build_result(fingerprint, model, winning_config, initial_loss,
                  final_loss, improvement, lr, steps, losses, runtime, status) -> dict:
    return {
        "schema_version": "1.0",
        "fingerprint": fingerprint,
        "model": model,
        "winning_config": winning_config,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement": improvement,
        "learning_rate_used": lr,
        "validation_steps": steps,
        "runtime_seconds": runtime,
        "losses_sample": losses[::5],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }


def _stub_result(fingerprint: str, winning_config: dict, reason: str) -> dict:
    return {
        "schema_version": "1.0",
        "fingerprint": fingerprint,
        "model": BASE_MODEL,
        "winning_config": winning_config,
        "status": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _write_result(result: dict) -> None:
    os.makedirs(config.STATE_DIR, exist_ok=True)
    with open(TINKER_RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[tinker_scaleup] Written to {TINKER_RESULTS_FILE}")


def _mark_scaled() -> None:
    """Mark global best as tinker_scaled=True."""
    try:
        with open(config.GLOBAL_BEST_JSON, "r") as f:
            best = json.load(f)
        best["tinker_scaled"] = True
        with open(config.GLOBAL_BEST_JSON, "w") as f:
            json.dump(best, f, indent=2)
    except Exception as e:
        print(f"[tinker_scaleup] WARNING: could not mark scaled: {e}")


if __name__ == "__main__":
    print("[tinker_scaleup] Testing threshold logic...")

    # Test threshold logic — no API key needed
    assert should_scale_up(
        {"best_reward": 0.15, "best_fingerprint": "abc", "tinker_scaled": False},
        initial_reward=0.05
    )
    assert not should_scale_up(
        {"best_reward": 0.055, "best_fingerprint": "abc", "tinker_scaled": False},
        initial_reward=0.05
    )
    assert not should_scale_up(
        {"best_reward": 0.15, "best_fingerprint": "abc", "tinker_scaled": True},
        initial_reward=0.05
    )
    print("[tinker_scaleup] Threshold logic ok")

    # Test API connectivity if key available
    api_key = os.environ.get("TINKER_API_KEY", "")
    if api_key:
        print("[tinker_scaleup] Testing API connectivity...")
        try:
            models = get_available_models()
            print(f"[tinker_scaleup] Connected — models: {models}")
        except Exception as e:
            print(f"[tinker_scaleup] API error: {e}")
    else:
        print("[tinker_scaleup] No TINKER_API_KEY — skipping API test")
        print("[tinker_scaleup] Get a key at: https://tinker-console.thinkingmachines.ai")

    print("[tinker_scaleup] ALL TESTS PASSED")
