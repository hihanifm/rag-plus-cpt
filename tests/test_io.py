from ragcpt import io_utils


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1}, {"b": "two"}, {"c": [1, 2, 3]}]
    p = tmp_path / "x.jsonl"
    n = io_utils.write_jsonl(p, rows)
    assert n == 3
    back = list(io_utils.read_jsonl(p))
    assert back == rows


def test_read_skips_blank_lines(tmp_path):
    p = tmp_path / "y.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n')
    assert list(io_utils.read_jsonl(p)) == [{"a": 1}, {"b": 2}]
