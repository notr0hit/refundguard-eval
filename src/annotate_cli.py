"""Blinded terminal CLI for Human-in-the-Loop (HITL) RLHF preference grading."""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "test_cases.jsonl"
RESULTS_PATH = ROOT_DIR / "results" / "raw_runs.json"
PREFS_PATH = ROOT_DIR / "annotation" / "rlhf_preferences.jsonl"

FAILURE_TAGS = [
    "HALLUCINATED_POLICY",
    "CONTRADICTION",
    "SYCOPHANCY",
    "INJECTION_SUCCESS",
    "MISSED_EXCEPTION",
    "MISSED_ESCALATION",
    "WRONG_PARTIAL",
    "TONE_ISSUE",
]


def load_cases() -> dict[str, dict]:
    """Load test cases indexed by case_id."""
    cases = {}
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cases[item["id"]] = item
    return cases


def load_existing_ids() -> set[str]:
    """Return case_ids already graded in rlhf_preferences.jsonl."""
    if not PREFS_PATH.exists():
        return set()
    graded = set()
    with PREFS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                graded.add(json.loads(line)["case_id"])
    return graded


def prompt_score(label: str) -> int:
    """Prompt for a 1-5 rubric score."""
    while True:
        val = input(f"  Score {label} (1-5): ").strip()
        if val in {"1", "2", "3", "4", "5"}:
            return int(val)
        print("  Please enter an integer from 1 to 5.")


def prompt_winner() -> str:
    """Prompt for preference winner or quit."""
    while True:
        val = input("  Winner [A / B / TIE / BOTH_BAD] (or 'q' to quit): ").strip().upper()
        if val in {"A", "B", "TIE", "BOTH_BAD", "Q"}:
            return val
        print("  Invalid choice. Enter A, B, TIE, BOTH_BAD, or q.")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {RESULTS_PATH}. Run `python src/run_eval.py` first.")

    cases = load_cases()
    runs = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_case[r["case_id"]].append(r)

    graded_ids = load_existing_ids()
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)

    pending = [cid for cid, items in by_case.items() if len(items) >= 2 and cid not in graded_ids]
    print(f"Loaded {len(by_case)} cases. Already graded: {len(graded_ids)}. Remaining: {len(pending)}")
    if not pending:
        print("All available pairs are already graded!")
        return

    rng = random.Random(42)
    for cid in pending:
        case = cases[cid]
        pair = rng.sample(by_case[cid], 2)
        run_a, run_b = pair[0], pair[1]

        print("\n" + "=" * 72)
        print(f"Case {cid} | Domain: {case['domain']} | Lang: {case['language']} | Attack: {case['attack_type']}")
        print(f"Expected: {case['expected_decision']} | Clauses: {case['expected_clauses']}")
        print(f"Notes: {case['notes']}")
        print("-" * 72)
        print(f"CUSTOMER PROMPT:\n{case['prompt']}")
        print("-" * 72)
        print(f"[MODEL A] Decision={run_a['model_decision']} | Clauses={run_a['cited_clauses']} | Contra={run_a['contradiction']}")
        print(run_a["customer_response"])
        print("-" * 72)
        print(f"[MODEL B] Decision={run_b['model_decision']} | Clauses={run_b['cited_clauses']} | Contra={run_b['contradiction']}")
        print(run_b["customer_response"])
        print("=" * 72)

        winner = prompt_winner()
        if winner == "Q":
            print("Progress saved. Exiting.")
            break

        score_a = prompt_score("Model A")
        score_b = prompt_score("Model B")
        print(f"  Available failure tags: {', '.join(FAILURE_TAGS)}")
        tags_raw = input("  Failure tags (comma-separated, or blank): ").strip()
        tags = [t.strip().upper() for t in tags_raw.split(",") if t.strip()]
        rationale = input("  Short rationale: ").strip()

        winner_model = (
            run_a["model"] if winner == "A"
            else run_b["model"] if winner == "B"
            else winner
        )

        record = {
            "case_id": cid,
            "domain": case["domain"],
            "language": case["language"],
            "attack_type": case["attack_type"],
            "model_a": run_a["model"],
            "model_b": run_b["model"],
            "score_a": score_a,
            "score_b": score_b,
            "winner_blinded": winner,
            "winner_model": winner_model,
            "failure_tags": tags,
            "rationale": rationale,
        }
        with PREFS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Saved annotation for {cid}.")


if __name__ == "__main__":
    main()

