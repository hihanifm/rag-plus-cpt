"""Leakage guard — no gold question (or close paraphrase) may appear in training QA.

HARD RULE: the gold eval set must be independent of training data. Since gold and training QA are
both generated from the same chunks by the same teacher, overlap is the *default* risk — a gold Q
that also trained the model makes the eval overstate lift (train/test contamination).

Stage 0: cheap, no-dep checks —
  - normalized exact match (lowercase, strip punctuation/space)
  - high-threshold token-Jaccard (catches reordering / trivial paraphrase)
Stage A: upgrade to true fuzzy (rapidfuzz / embedding similarity) for harder paraphrases.

Warns (never auto-deletes): gold is sacred, so a human decides whether to drop the gold item or the
training QA.
"""
from __future__ import annotations

import re

_JACCARD_THRESHOLD = 0.85


def normalize(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", q.lower())).strip()


def _tokens(q: str) -> set[str]:
    return set(normalize(q).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_leaks(gold_questions: list[str], train_questions: list[str],
               jaccard_threshold: float = _JACCARD_THRESHOLD) -> list[dict]:
    """Return gold questions that leak into training QA, with the match + how it matched."""
    train_norm = {normalize(q): q for q in train_questions}
    train_tok = [(_tokens(q), q) for q in train_questions]
    leaks: list[dict] = []
    for g in gold_questions:
        gn = normalize(g)
        if gn in train_norm:
            leaks.append({"gold": g, "match": train_norm[gn], "kind": "exact"})
            continue
        gt = _tokens(g)
        best = max(((_jaccard(gt, tt), tq) for tt, tq in train_tok), default=(0.0, ""))
        if best[0] >= jaccard_threshold:
            leaks.append({"gold": g, "match": best[1], "kind": f"jaccard={best[0]:.2f}"})
    return leaks
