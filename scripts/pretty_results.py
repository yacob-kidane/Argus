"""
Pretty printer for results.jsonl — pipe tail -f into this for live monitoring.
Usage: tail -f state/results.jsonl | python3 scripts/pretty_results.py
"""
import json
import sys

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"

def color_status(status: str) -> str:
    if status == "kept":
        return f"{GREEN}✓ kept{RESET}"
    elif status == "rejected":
        return f"{YELLOW}✗ rejected{RESET}"
    elif status == "failed":
        return f"{RED}✗ failed{RESET}"
    return status

def format_record(r: dict) -> str:
    status  = color_status(r.get("status", "?"))
    node    = r.get("node", "?")[:12]
    val_bpb = r.get("val_bpb", 0)
    reward  = r.get("reward", 0)
    fp      = r.get("fingerprint", "")[:8]
    hyp     = r.get("hypothesis", "")[:70]
    ts      = r.get("timestamp", "")[-8:][:8]  # HH:MM:SSZ

    return (
        f"{GRAY}[{ts}]{RESET} {status} "
        f"{CYAN}[{node}]{RESET} "
        f"val_bpb={BOLD}{val_bpb:.4f}{RESET} "
        f"reward={BOLD}{reward:.6f}{RESET} "
        f"{GRAY}fp={fp}{RESET}\n"
        f"  {GRAY}→{RESET} {hyp}"
    )

def main():
    print(f"{BOLD}Autoresearch live results{RESET} — waiting for experiments...\n")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            print(format_record(r))
            print()
            sys.stdout.flush()
        except json.JSONDecodeError:
            print(f"{GRAY}{line}{RESET}")

if __name__ == "__main__":
    main()
