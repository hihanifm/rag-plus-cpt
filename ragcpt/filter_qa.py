"""filter_qa — grounding auto-cleaner.

CONTRACT
  in:   data/qa_raw.jsonl   (qa_type, question, answer, chunk_text, chunk_id, ...)
  out:  data/qa.jsonl       (kept)        + data/qa_rejected.jsonl (dropped, with reject_reason)
  invariants:
    - `unknown` pairs are kept unjudged (they are the RAG-handoff; not answerable from the section)
    - non-unknown pairs kept only if the answer is supported by chunk_text
    - questions deduped (normalized); first occurrence wins
    - **batched per-chunk judge**: all QA from one chunk judged in ONE call (same source) -> ~3-4x
      fewer requests (eases RPM); a chunk's call failing rejects that chunk's pairs (logged, retriable)
    - every input pair is accounted for: kept ∪ rejected (no silent drops, except judge errors logged)
  acceptance: spot-check precision >= ~90%  (`ragcpt review` / run_mvp audit)

The judge LLM returns {grounded: bool, reason}. After this runs, SPOT-CHECK it (judge ~30 kept pairs
by hand) to confirm the cleaner is reliable before trusting the whole set.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .config import load_config
from .io_utils import read_jsonl, write_jsonl
from .llm_client import chat_json, judge_client

_JUDGE = """Decide whether the ANSWER is fully supported by the SOURCE paragraph. The answer is \
"grounded" only if every fact in it can be verified from the SOURCE alone (no outside knowledge).

Judge ALL the numbered (QUESTION, ANSWER) pairs below against the SAME source.

SOURCE:
\"\"\"
{chunk}
\"\"\"

PAIRS:
{pairs}

Return JSON with one verdict per pair, same order/index:
{{"verdicts": [{{"i": 1, "grounded": true|false, "reason": "<one sentence>"}}, ...]}}.
"""


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _partition(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pure: dedup by normalized question; split into (unknown_auto_keep, candidates_needing_judge).
    First occurrence of a question wins. Testable without any API."""
    seen: set[str] = set()
    unknown_keep: list[dict] = []
    candidates: list[dict] = []
    for row in rows:
        key = _norm(row["question"])
        if key in seen:
            continue
        seen.add(key)
        (unknown_keep if row.get("qa_type") == "unknown" else candidates).append(row)
    return unknown_keep, candidates


def _group_by_chunk(rows: list[dict]) -> dict[str, list[dict]]:
    """Pure: group QA by chunk_id (preserving order) so each chunk is judged in one batched call."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["chunk_id"], []).append(r)
    return groups


def filter_qa(qa_raw: str | None = None, out: str | None = None) -> Path:
    cfg = load_config()
    in_path = Path(qa_raw) if qa_raw else cfg.qa_raw_path
    out_path = Path(out) if out else cfg.qa_path
    client, model = judge_client()

    concurrency = cfg.get("gen_qa", "concurrency", default=8)
    rejected_path = cfg.qa_rejected_path

    all_rows = list(read_jsonl(in_path))
    n_in = len(all_rows)
    unknown_keep, candidates = _partition(all_rows)
    groups = _group_by_chunk(candidates)   # one judge call per chunk (they share the source)

    def judge_group(items: list[dict]) -> tuple[list[dict], list[dict]]:
        """Judge all of a chunk's QA in ONE call. Returns (kept, rejected)."""
        pairs = []
        for i, r in enumerate(items, 1):
            ans = re.sub(r"<think>.*?</think>", "", r["answer"], flags=re.DOTALL).strip()
            pairs.append(f"{i}. QUESTION: {r['question']}\n   ANSWER: {ans}")
        try:
            resp = chat_json(client, model, [{"role": "user", "content": _JUDGE.format(
                chunk=items[0]["chunk_text"], pairs="\n".join(pairs))}])
            verdicts = {v.get("i"): v for v in resp.get("verdicts", [])}
        except Exception as e:
            print(f"[filter_qa] batch judge failed ({items[0]['chunk_id']}): {e}")
            for r in items:
                r["reject_reason"] = f"judge error: {e}"
            return [], list(items)
        kept_l, rej_l = [], []
        for i, r in enumerate(items, 1):
            v = verdicts.get(i) or {}
            if v.get("grounded"):
                r["grounded_reason"] = v.get("reason", "")
                kept_l.append(r)
            else:
                r["reject_reason"] = v.get("reason", "not grounded / no verdict")
                rej_l.append(r)
        return kept_l, rej_l

    kept: list[dict] = list(unknown_keep)
    rejected: list[dict] = []
    group_list = list(groups.values())
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for kept_l, rej_l in tqdm(ex.map(judge_group, group_list), total=len(group_list),
                                  desc="filter_qa"):
            kept.extend(kept_l)
            rejected.extend(rej_l)

    n = write_jsonl(out_path, kept)
    write_jsonl(rejected_path, rejected)          # evidence: never silently drop rejects
    rate = (n / n_in * 100) if n_in else 0
    print(f"[filter_qa] kept {n}/{n_in} ({rate:.0f}%) -> {out_path}  "
          f"| rejected {len(rejected)} -> {rejected_path}")
    return out_path
