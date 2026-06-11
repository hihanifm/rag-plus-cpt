from ragcpt import filter_qa


def _row(qa_type, q):
    return {"qa_type": qa_type, "question": q, "answer": "a", "chunk_text": "src", "chunk_id": "c::0"}


def test_unknown_pairs_auto_kept_not_judged():
    rows = [_row("unknown", "exact value?"), _row("plain", "what is X?")]
    unknown_keep, candidates = filter_qa._partition(rows)
    assert [r["question"] for r in unknown_keep] == ["exact value?"]
    assert [r["question"] for r in candidates] == ["what is X?"]


def test_dedup_first_occurrence_wins():
    rows = [_row("plain", "What is X?"), _row("plain", "what is x?  "), _row("plain", "What is Y?")]
    _, candidates = filter_qa._partition(rows)
    # the two "what is x" normalize equal -> one kept; plus the distinct Y
    assert len(candidates) == 2
    assert candidates[0]["question"] == "What is X?"


def test_every_input_accounted_for():
    rows = [_row("unknown", "u1"), _row("plain", "p1"), _row("plain", "p2"), _row("unknown", "u1")]
    unknown_keep, candidates = filter_qa._partition(rows)
    # u1 deduped -> 1 unknown, 2 candidates; total unique = 3
    assert len(unknown_keep) + len(candidates) == 3
