import random

from ragcpt import build_sft


def _chunk(i):
    return {"id": f"d::{i}", "carrier": "Verizon", "doc_type": "spec", "doc_name": "d",
            "heading": f"H{i}", "header": "[Carrier: Verizon | Type: spec | Doc: d]", "text": "body " * 20}


def _qa(t):
    return {"qa_type": t, "question": f"q-{t}", "answer": "a", "carrier": "Verizon",
            "header": "[Carrier: Verizon | Type: spec | Doc: d]", "chunk_id": "d::0", "chunk_text": "src"}


def test_raw_sample_is_messages():
    s = build_sft._raw_sample(_chunk(0), random.Random(0))
    roles = [m["role"] for m in s["messages"]]
    assert roles == ["user", "assistant"]
    assert s["sample_type"] == "raw_text"
    assert "[Carrier: Verizon" in s["messages"][1]["content"]


def test_qa_sample_has_system():
    s = build_sft._qa_sample(_qa("think"))
    roles = [m["role"] for m in s["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert s["sample_type"] == "qa_think"


def test_apply_mix_covers_all_sections_and_hits_ratio():
    rng = random.Random(0)
    raw = [build_sft._raw_sample(_chunk(i), rng) for i in range(30)]
    qa = {"plain": [build_sft._qa_sample(_qa("plain")) for _ in range(50)],
          "think": [build_sft._qa_sample(_qa("think")) for _ in range(50)],
          "unknown": [build_sft._qa_sample(_qa("unknown")) for _ in range(10)]}
    mix = {"raw_text": 30, "qa_plain": 30, "qa_think": 30, "qa_unknown": 10}
    out = build_sft._apply_mix(raw, qa, mix, rng)

    types = [s["sample_type"] for s in out]
    # every raw section is present (full coverage)
    assert types.count("raw_text") == 30
    # qa buckets scaled ~ to weights relative to raw (base_per_weight = 30/30 = 1 sample per weight pt)
    assert types.count("qa_plain") == 30
    assert types.count("qa_think") == 30
    assert types.count("qa_unknown") == 10


def test_apply_mix_upsamples_small_pool():
    rng = random.Random(0)
    raw = [build_sft._raw_sample(_chunk(i), rng) for i in range(30)]
    qa = {"plain": [build_sft._qa_sample(_qa("plain")) for _ in range(5)], "think": [], "unknown": []}
    mix = {"raw_text": 30, "qa_plain": 30, "qa_think": 0, "qa_unknown": 0}
    out = build_sft._apply_mix(raw, qa, mix, rng)
    # target 30 from a pool of 5 -> upsampled with replacement
    assert [s["sample_type"] for s in out].count("qa_plain") == 30
