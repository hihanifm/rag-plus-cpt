"""Score a served model (base or tuned) against gold.jsonl + chat_probes.jsonl.

gold.jsonl record: {"question","expected","category"}  (category includes "unknown" for
insufficient-evidence items). Grading is LLM-judged (correct / incorrect), plus an "unknown" item is
counted as a deferral if the model declines instead of inventing a value.

Writes eval_report.json. Compares against pipeline.yaml `success` thresholds.

FAIRNESS RULE (do not violate): base and tuned must be evaluated under IDENTICAL conditions —
same system prompt, same chat template, same decoding params (temperature/max_tokens/seed), same
gold set, same questions. Any improvement must come from training, not prompt/decoding differences.
This is enforced structurally: both go through the single `_answer()` using `EVAL_DECODING`, and both
are served as Qwen3-14B (same template). Never special-case base vs tuned here.
"""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .config import load_config
from .io_utils import read_jsonl
from .llm_client import chat, chat_json, endpoint_client, judge_client

_SYSTEM = (Path(__file__).parent / "reasoning_skeleton.md").read_text()

# Shared decoding params — applied IDENTICALLY to base and tuned (the fairness rule).
EVAL_DECODING = {"temperature": 0.0, "max_tokens": 768, "seed": 0}

_GRADE = """Grade whether the MODEL ANSWER is correct given the EXPECTED answer. Be strict about \
specific values but allow paraphrase.

QUESTION: {q}
EXPECTED: {expected}
MODEL ANSWER: {a}

Return JSON: {{"correct": true|false}}.
"""

_DEFER = """Did the MODEL ANSWER decline to give an exact value and instead defer to checking the \
source / RAG (rather than inventing a specific value)?

MODEL ANSWER: {a}

Return JSON: {{"deferred": true|false}}.
"""


def _leakage_check(cfg, gold: list[dict]) -> None:
    """Warn (loudly) if any gold question leaks into training QA — eval validity depends on it."""
    from .leakage import find_leaks
    if not cfg.qa_path.exists():
        return
    train_qs = [r["question"] for r in read_jsonl(cfg.qa_path)]
    leaks = find_leaks([g["question"] for g in gold], train_qs)
    if leaks:
        print(f"\n  ⚠️  LEAKAGE WARNING: {len(leaks)} gold question(s) overlap training QA "
              f"(eval will overstate lift). Drop them from gold or from qa.jsonl:")
        for lk in leaks[:10]:
            print(f"     [{lk['kind']}] gold: {lk['gold'][:90]}")
    else:
        print("  ✓ leakage check: no gold question overlaps training QA")


def _answer(client, model, question: str) -> str:
    # Same system prompt + same EVAL_DECODING for base AND tuned — fairness is structural here.
    return chat(client, model, [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": question},
    ], **EVAL_DECODING)


def evaluate(which: str = "tuned", out: str | None = None) -> dict:
    """which: 'tuned' or 'base'. Returns the report dict and writes eval_report.json."""
    cfg = load_config()
    client, model = endpoint_client(which)
    judge, jmodel = judge_client()

    gold = list(read_jsonl(cfg.gold_path))
    _leakage_check(cfg, gold)
    probes_path = cfg.chat_probes_path
    probes = list(read_jsonl(probes_path)) if probes_path.exists() else []

    correct = 0
    unknown_total = 0
    unknown_deferred = 0
    rows = []
    for item in tqdm(gold, desc=f"eval[{which}]"):
        a = _answer(client, model, item["question"])
        is_unknown = item.get("category") == "unknown"
        rec = {"question": item["question"], "answer": a, "category": item.get("category")}
        if is_unknown:
            unknown_total += 1
            v = chat_json(judge, jmodel, [{"role": "user", "content": _DEFER.format(a=a)}])
            deferred = bool(v.get("deferred"))
            unknown_deferred += int(deferred)
            rec["deferred"] = deferred
        else:
            v = chat_json(judge, jmodel, [{"role": "user", "content": _GRADE.format(
                q=item["question"], expected=item.get("expected", ""), a=a)}])
            ok = bool(v.get("correct"))
            correct += int(ok)
            rec["correct"] = ok
        rows.append(rec)

    # chat/thinking probes: just record outputs for human glance (regression guard)
    probe_rows = [{"prompt": p.get("prompt", ""), "answer": _answer(client, model, p.get("prompt", ""))}
                  for p in probes]

    n_knowledge = len(gold) - unknown_total
    defer_pct = (unknown_deferred / unknown_total * 100) if unknown_total else 100.0
    report = {
        "which": which,
        "model": model,
        "gold_total": len(gold),
        "knowledge_correct": correct,
        "knowledge_total": n_knowledge,
        "unknown_deferred": unknown_deferred,
        "unknown_total": unknown_total,
        "unknown_defer_pct": round(defer_pct, 1),
        "success_check": _check(cfg, correct, defer_pct),
        "rows": rows,
        "probes": probe_rows,
    }
    out_path = Path(out) if out else cfg.data_dir / f"eval_report_{which}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[eval:{which}] knowledge {correct}/{n_knowledge}, "
          f"unknown deferred {unknown_deferred}/{unknown_total} ({defer_pct:.0f}%) -> {out_path}")
    return report


def _check(cfg, correct: int, defer_pct: float) -> dict:
    pass_min = cfg.get("success", "gold_pass_min", default=28)
    defer_min = cfg.get("success", "unknown_defer_min_pct", default=80)
    return {
        "knowledge_pass": correct >= pass_min,
        "defer_pass": defer_pct >= defer_min,
        "note": "Compare tuned vs base reports; PASS needs tuned >> base AND defer >= threshold "
                "AND no chat-probe regression (eyeball probes).",
    }
