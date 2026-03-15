"""
Proposal engine — generates experiment hypotheses using OpenAI API.
Single responsibility: turn current state into a structured Hypothesis.
See SCHEMA.md §1 and AGENTS.md interfaces.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from state.schemas import Hypothesis

try:
    from openai import OpenAI
    _client = OpenAI(api_key=config.OPENAI_API_KEY)
except ImportError:
    _client = None


SYSTEM_PROMPT = """You are an ML research agent proposing the next training experiment.

Your goal is to maximize discovery efficiency:
(model quality improvement) / (compute spent)

Read the recent experiment history and current best result.
Identify patterns in what helped and what hurt.
Propose one concrete next experiment.

Return valid JSON only with these fields:
- hypothesis
- diff
- predicted_effect
- risk
- fingerprint
- reasoning

Requirements:
- reasoning must reference the observed experiment history
- keep reasoning under 120 words
- propose only one concrete experiment
- diff must be precise and executable
- risk must be one of: low, medium, high
"""

BASELINE_SYSTEM_PROMPT = """You are an ML research agent proposing the next training experiment.

Your goal is to propose one concrete hyperparameter change without using prior
experiment history. Reason from the research objective and the current editable
training configuration only.

Return valid JSON only with these fields:
- hypothesis
- diff
- predicted_effect
- risk
- fingerprint
- reasoning

Requirements:
- reasoning should be brief and based on first-principles intuition
- keep reasoning under 120 words
- propose only one concrete experiment
- diff must be precise and executable
- risk must be one of: low, medium, high
"""


def _load_program_md() -> str:
    """Load program.md research strategy."""
    if os.path.exists(config.PROGRAM_MD):
        with open(config.PROGRAM_MD, "r", encoding="utf-8") as f:
            return f.read()
    return "Minimize val_bpb on the nanochat training loop."


def _load_train_py() -> str:
    """Load current train.py source."""
    with open(config.TRAIN_PY_SOURCE, "r", encoding="utf-8") as f:
        return f.read()


def _extract_hyperparam_section(train_py: str) -> str:
    """Extract only the hyperparameter section from train.py."""
    lines = train_py.splitlines()
    result = []
    in_section = False
    for line in lines:
        if config.HYPERPARAM_SECTION_START in line:
            in_section = True
        if in_section and config.HYPERPARAM_SECTION_END.split("\n")[0] in line:
            break
        if in_section:
            result.append(line)
    return "\n".join(result)


def fingerprint_from_diff(diff: str) -> str:
    """Return the canonical fingerprint for a proposal diff."""
    return hashlib.md5(diff.encode("utf-8")).hexdigest()


def normalize_proposer_policy(policy: str | None) -> str:
    """Normalize runtime proposer policy names."""
    value = str(policy or "").strip().lower()
    if value in {"history", "history_aware", "history-aware"}:
        return "history"
    if value in {"baseline", "random"}:
        return "baseline"
    return "history"


def summarize_recent_results(results: list[dict], limit: int = 8) -> list[dict]:
    """Return a compact newest-first summary of recent terminal experiment records."""
    terminal = [
        r for r in results
        if str(r.get("outcome", "")).lower() in {
            "succeeded", "failed", "timed_out", "cancelled"
        }
    ]
    terminal = sorted(terminal, key=lambda r: r.get("timestamp", ""), reverse=True)
    summary = []
    for record in terminal[:limit]:
        summary.append({
            "fingerprint": record.get("fingerprint"),
            "status": record.get("status"),
            "outcome": record.get("outcome"),
            "val_bpb": record.get("val_bpb"),
            "reward": record.get("reward"),
            "config": record.get("config", {}),
            "hypothesis": record.get("hypothesis", ""),
        })
    return summary


def build_history_context(results: list[dict], best: dict | None) -> str:
    """Build compact JSON context for the proposer from recent history."""
    from state import state_store

    promotions = state_store.read_promotions()
    recent_promotions = sorted(
        promotions, key=lambda r: r.get("timestamp", ""), reverse=True
    )[:4]
    payload = {
        "current_best": best or {},
        "recent_results": summarize_recent_results(results, limit=8),
        "recent_promotions": [
            {
                "fingerprint": row.get("fingerprint"),
                "relative_improvement": row.get("relative_improvement"),
                "promotion_reason": row.get("promotion_reason"),
                "tinker_status": row.get("tinker_status"),
                "timestamp": row.get("timestamp"),
            }
            for row in recent_promotions
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def append_reasoning_log(
    worker_id: str,
    proposal: dict,
    path: Path | None = None,
) -> None:
    """Append a proposal reasoning trace to the shared log."""
    if path is None:
        path = Path(config.REASONING_JSONL)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker_id": worker_id,
        "fingerprint": proposal.get("fingerprint", ""),
        "hypothesis": proposal.get("hypothesis", ""),
        "reasoning": proposal.get("reasoning", ""),
        "predicted_effect": proposal.get("predicted_effect", ""),
        "risk": proposal.get("risk", ""),
        "diff": proposal.get("diff", ""),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _build_prompt(
    results: list[dict],
    best: dict | None,
    train_py: str,
    active_fingerprints: list[str],
    proposer_policy: str,
) -> str:
    """Build the user prompt for hypothesis generation."""
    program_text = _load_program_md()
    hyperparam_section = _extract_hyperparam_section(train_py)

    if proposer_policy == "baseline":
        return f"""Research objective:
{program_text}

Current Hyperparameter Section (the ONLY code you may change):
```python
{hyperparam_section}
```

Currently in-flight fingerprints (do not duplicate these):
{json.dumps(active_fingerprints, indent=2)}

Constraints:
- Do not use prior experiment history when choosing the next change
- Only change values in the hyperparameter section shown above
- Do not change variable names, only assigned values
- Make one focused change or a tightly related set of changes
- Do not repeat experiments already currently in flight
- Keep the proposal concrete and executable

Propose the next experiment."""

    history_context = build_history_context(results, best)

    return f"""Research objective:
{program_text}

Current Hyperparameter Section (the ONLY code you may change):
```python
{hyperparam_section}
```

Currently in-flight fingerprints (do not duplicate these):
{json.dumps(active_fingerprints, indent=2)}

Experiment history:
{history_context}

Constraints:
- Only change values in the hyperparameter section shown above
- Do not change variable names, only assigned values
- Make one focused change or a tightly related set of changes
- Do not repeat experiments already tried or currently in flight
- Keep the proposal concrete and executable

Propose the next experiment."""


def next_hypothesis(
    results: list[dict],
    best: dict,
    train_py: str,
    active_fingerprints: list[str] | None = None,
    worker_id: str = "unknown",
    proposer_policy: str = "history",
    condition_id: str = "",
) -> Hypothesis:
    """
    Generate next experiment hypothesis.
    Returns a Hypothesis with fingerprint = md5(diff).
    """
    if active_fingerprints is None:
        active_fingerprints = []
    proposer_policy = normalize_proposer_policy(proposer_policy)

    if _client is None:
        raise RuntimeError("openai package not installed")

    prompt = _build_prompt(
        results,
        best,
        train_py,
        active_fingerprints,
        proposer_policy,
    )

    response = _client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    BASELINE_SYSTEM_PROMPT
                    if proposer_policy == "baseline"
                    else SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=700,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"[proposal_engine] invalid JSON from model: {e}\n{raw}")

    for field in ("hypothesis", "diff", "predicted_effect", "risk", "reasoning"):
        if field not in data:
            raise ValueError(f"[proposal_engine] missing field: {field}")

    diff = str(data.get("diff", "")).strip()
    if not diff:
        raise ValueError("[proposal_engine] diff must not be empty")

    risk = str(data.get("risk", "low")).strip().lower()
    if risk not in ("low", "medium", "high"):
        risk = "low"

    reasoning = str(data.get("reasoning", "")).strip() or "No reasoning provided."

    proposal = {
        "hypothesis": str(data.get("hypothesis", "")).strip(),
        "diff": diff,
        "predicted_effect": str(data.get("predicted_effect", "")).strip(),
        "risk": risk,
        "reasoning": reasoning,
    }
    proposal["fingerprint"] = fingerprint_from_diff(proposal["diff"])

    append_reasoning_log(worker_id, proposal, Path(config.REASONING_JSONL))

    return Hypothesis(
        hypothesis=proposal["hypothesis"],
        diff=proposal["diff"],
        predicted_effect=proposal["predicted_effect"],
        risk=proposal["risk"],
        fingerprint=proposal["fingerprint"],
        proposal_source="llm",
        proposer_policy=proposer_policy,
        condition_id=condition_id,
        notes=proposal["reasoning"],
    )


def seed_hypothesis() -> Hypothesis:
    """
    Return a safe seed hypothesis for the very first experiment.
    Used when results.jsonl is empty — establishes the baseline.
    """
    diff = "# baseline — no changes"
    return Hypothesis(
        hypothesis="Baseline run — no changes to establish val_bpb reference",
        diff=diff,
        predicted_effect="Establishes baseline val_bpb for reward computation",
        risk="low",
        fingerprint=fingerprint_from_diff(diff),
        proposal_source="seed",
        proposer_policy="seed",
        notes="seed fallback",
    )


if __name__ == "__main__":
    from state import state_store

    state_store.ensure_state_dirs()
    train_py = _load_train_py()

    # Test seed hypothesis
    seed = seed_hypothesis()
    assert seed.proposal_source == "seed"
    assert seed.fingerprint == fingerprint_from_diff(seed.diff)
    print(f"[proposal_engine] seed hypothesis ok: {seed.hypothesis}")

    # Test prompt building
    prompt = _build_prompt([], {}, train_py, [])
    assert "hyperparameter" in prompt.lower()
    assert "MATRIX_LR" in prompt
    print("[proposal_engine] prompt building ok")

    # Test real API call
    if not config.OPENAI_API_KEY:
        print("[proposal_engine] skipping API test — no OPENAI_API_KEY")
    else:
        print("[proposal_engine] calling OpenAI API...")
        h = next_hypothesis(
            results=[],
            best={},
            train_py=train_py,
            active_fingerprints=[],
            worker_id="selftest",
        )
        assert h.fingerprint == fingerprint_from_diff(h.diff)
        assert h.risk in ("low", "medium", "high")
        assert len(h.hypothesis) > 10
        assert len(h.diff) > 0
        print(f"[proposal_engine] API ok")
        print(f"  hypothesis: {h.hypothesis}")
        print(f"  diff:       {h.diff}")
        print(f"  risk:       {h.risk}")
        print(f"  fingerprint:{h.fingerprint}")

    print("[proposal_engine] ALL TESTS PASSED")
