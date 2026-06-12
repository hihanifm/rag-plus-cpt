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


def _row_c(chunk_id, q):
    r = _row("plain", q)
    r["chunk_id"] = chunk_id
    return r


def test_group_by_chunk_batches_same_source():
    rows = [_row_c("c::0", "q1"), _row_c("c::0", "q2"), _row_c("c::1", "q3")]
    groups = filter_qa._group_by_chunk(rows)
    assert set(groups) == {"c::0", "c::1"}
    assert len(groups["c::0"]) == 2 and len(groups["c::1"]) == 1
    # one judge call per chunk instead of per pair: 2 calls for 3 pairs
    assert len(groups) == 2
