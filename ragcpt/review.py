"""Human-in-the-loop review helper. Two modes — you click/judge, you never hand-code JSONL.

  mode="audit"  : SPOT-CHECK THE AUTO-CLEANER. Shows up to N kept QA pairs with their source
                  paragraph; you type y/n (is the answer really grounded?). Prints precision = y/N.
                  Throwaway diagnostic — confirms the grounding cleaner is trustworthy.

  mode="curate" : BUILD THE GOLD SET. Teacher drafts candidate gold Q&A from sections; for each you
                  [a]pprove / [e]dit / [s]kip. Approved items append to eval/gold.jsonl.
                  Keep trick/comparison/unknown items hand-written (add them directly) to avoid
                  teacher-distribution leakage in the yardstick.

Terminal-based (input()). Run interactively tomorrow.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .config import load_config
from .io_utils import read_jsonl
from .llm_client import chat_json, teacher_client


def audit(n: int = 30, qa: str | None = None) -> float:
    cfg = load_config()
    qa_path = Path(qa) if qa else cfg.qa_path
    rows = [r for r in read_jsonl(qa_path) if r["qa_type"] != "unknown"]
    random.shuffle(rows)
    rows = rows[:n]
    if not rows:
        print("No QA pairs to audit. Run gen_qa + filter_qa first.")
        return 0.0

    good = 0
    for i, r in enumerate(rows, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(rows)}] SOURCE PARAGRAPH:\n{r['chunk_text'][:1200]}")
        print(f"\nQ: {r['question']}\nA: {r['answer']}")
        ans = input("Is the answer fully supported by the source? [y/n/q] ").strip().lower()
        if ans == "q":
            break
        good += int(ans == "y")
    checked = i
    precision = good / checked * 100 if checked else 0
    print(f"\n[audit] grounded precision = {good}/{checked} = {precision:.0f}%")
    print("  >= ~90% -> trust the cleaner. Lower -> tighten the judge prompt in filter_qa.py.")
    return precision


_DRAFT = """From this carrier document section, draft {k} factual gold-eval questions with their \
correct answers (answerable ONLY from the section). Mix a direct-fact Q and a "what changed from the \
3GPP default?" Q if possible.

Carrier: {carrier}
SECTION:
\"\"\"
{text}
\"\"\"

Return JSON: {{"items": [{{"question":"...","expected":"...","category":"fact"|"comparison"}}]}}
"""


def curate(k_per_section: int = 1, chunks: str | None = None, gold: str | None = None) -> Path:
    cfg = load_config()
    chunks_path = Path(chunks) if chunks else cfg.chunks_path
    gold_path = Path(gold) if gold else cfg.gold_path
    client, model = teacher_client()

    rows = list(read_jsonl(chunks_path))
    print(f"Curating gold from {len(rows)} sections. For each candidate: [a]pprove [e]dit [s]kip [q]uit")
    approved: list[dict] = []
    for ch in rows:
        try:
            draft = chat_json(client, model, [{"role": "user", "content": _DRAFT.format(
                k=k_per_section, carrier=ch["carrier"], text=ch["text"][:5000])}])
        except Exception as e:
            print(f"  draft failed for {ch['id']}: {e}")
            continue
        for item in draft.get("items", []):
            print("\n" + "-" * 70)
            print(f"Q: {item.get('question')}\nExpected: {item.get('expected')}\nCategory: {item.get('category')}")
            choice = input("[a/e/s/q] ").strip().lower()
            if choice == "q":
                _append(gold_path, approved)
                print(f"[curate] saved {len(approved)} items -> {gold_path}")
                return gold_path
            if choice == "s":
                continue
            if choice == "e":
                item["question"] = input(f"  question [{item.get('question')}]: ") or item["question"]
                item["expected"] = input(f"  expected [{item.get('expected')}]: ") or item["expected"]
            approved.append({"question": item["question"], "expected": item["expected"],
                             "category": item.get("category", "fact")})
    _append(gold_path, approved)
    print(f"[curate] saved {len(approved)} items -> {gold_path}")
    print("  Reminder: hand-add trick/comparison/unknown items directly to gold.jsonl (leakage guard).")
    return gold_path


def _append(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
