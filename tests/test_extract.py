from pathlib import Path

from ragcpt import extract


def test_infer_metadata_from_folder():
    root = Path("/docs")
    meta = extract.infer_metadata(Path("/docs/Verizon/requirements/VZW_Attach_v3.docx"), root)
    assert meta["carrier"] == "Verizon"
    assert meta["doc_type"] == "requirements"
    assert meta["doc_name"] == "VZW_Attach_v3"


def test_infer_metadata_from_filename():
    root = Path("/docs")
    meta = extract.infer_metadata(Path("/docs/att_testplan_5g.docx"), root)
    assert meta["carrier"] == "AT&T"
    assert meta["doc_type"] == "test_plan"


def test_infer_metadata_unknown():
    root = Path("/docs")
    meta = extract.infer_metadata(Path("/docs/random_file.docx"), root)
    assert meta["carrier"] == "Unknown"
    assert meta["doc_type"] == "unknown"


def test_header_format():
    meta = {"carrier": "Verizon", "doc_type": "spec", "doc_name": "X"}
    assert extract.header(meta, "Heading") == "[Carrier: Verizon | Type: spec | Doc: X]"
