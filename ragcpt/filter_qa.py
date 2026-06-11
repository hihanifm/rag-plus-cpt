"""Auto-cleaner: keep a QA pair only if its answer is supported by the source paragraph.

The judge LLM returns {grounded: bool, reason}. Plain/think pairs are rejected if not grounded.
`unknown` pairs are kept by definition (they are deliberately not answerable from the section — the
"defer to RAG" behavior). Near-duplicate questions are dropped.

After this runs, SPOT-CHECK it with `ragcpt review --mode audit` (judge 30 kept pairs by hand) to
confirm the cleaner itself is reliable before trusting the whole set.
Output: data/qa.jsonl
"""
from __future__ import annotations

import re
from pathlib import Path

from tqdm import tqdm

from .config import load_config
from .io_utils import read_jsonl, write_jsonl
from .llm_client import chat_json, judge_client

_JUDGE = """Decide whether the ANSWER is fully supported by the SOURCE paragraph. The answer is \
"grounded" only if every fact in it can be verified from the SOURCE alone (no outside knowledge).

SOURCE:
\"\"\"
{chunk}
\"\"\"

QUESTION: {q}
ANSWER: {a}

Return JSON: {{"grounded": true|false, "reason": "<one sentence>"}}.
"""


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def filter_qa(qa_raw: str | None = None, out: str | None = None) -> Path:
    cfg = load_config()
    in_path = Path(qa_raw) if qa_raw else cfg.data_dir / "qa_raw.jsonl"
    out_path = Path(out) if out else cfg.data_dir / "qa.jsonl"
    client, model = judge_client()

    seen: set[str] = set()
    kept: list[dict] = []
    n_in = 0
    for row in tqdm(list(read_jsonl(in_path)), desc="filter_qa"):
        n_in += 1
        key = _norm(row["question"])
        if key in seen:
            continue
        if row["qa_type"] == "unknown":
            seen.add(key)
            kept.append(row)
            continue
        # strip any <think> block before grounding-judging the final answer
        ans = re.sub(r"<think>.*?</think>", "", row["answer"], flags=re.DOTALL).strip()
        try:
            verdict = chat_json(client, model, [{"role": "user", "content": _JUDGE.format(
                chunk=row["chunk_text"], q=row["question"], a=ans)}])
        except Exception as e:
            print(f"[filter_qa] judge failed ({row['chunk_id']}): {e}; dropping")
            continue
        if verdict.get("grounded"):
            seen.add(key)
            row["grounded_reason"] = verdict.get("reason", "")
            kept.append(row)

    n = write_jsonl(out_path, kept)
    rate = (n / n_in * 100) if n_in else 0
    print(f"[filter_qa] kept {n}/{n_in} ({rate:.0f}%) -> {out_path}")
    return out_path
