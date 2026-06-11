"""Parse .docx → sectioned chunks with metadata.

Stage 0: metadata (carrier / doc_type / doc_name) is inferred deterministically from the path +
filename (cheap, no API). Stage A swaps in LLM enrichment (find intro, richer metadata) behind the
same chunk schema. Output: data/chunks.jsonl, one record per section.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from docx import Document

from .config import load_config

_CARRIERS = {
    "verizon": "Verizon", "vzw": "Verizon", "vz": "Verizon",
    "att": "AT&T", "at&t": "AT&T", "att_": "AT&T",
    "tmobile": "T-Mobile", "tmo": "T-Mobile", "t-mobile": "T-Mobile",
}
_DOC_TYPES = {
    "requirement": "requirements", "req": "requirements",
    "testplan": "test_plan", "test_plan": "test_plan", "test": "test_plan",
    "spec": "spec", "specification": "spec",
}


def infer_metadata(path: Path, docs_root: Path) -> dict:
    """Carrier/doc_type/doc_name from path tokens. A hint, overridable later by LLM enrichment."""
    tokens = []
    rel = path.relative_to(docs_root) if str(path).startswith(str(docs_root)) else path
    tokens += [p.lower() for p in rel.parts]
    tokens += re.split(r"[ _\-.]+", path.stem.lower())
    blob = " ".join(tokens)

    carrier = "Unknown"
    for key, val in _CARRIERS.items():
        if key in blob:
            carrier = val
            break
    doc_type = "unknown"
    for key, val in _DOC_TYPES.items():
        if key in blob:
            doc_type = val
            break
    return {"carrier": carrier, "doc_type": doc_type, "doc_name": path.stem}


def _iter_sections(doc: Document) -> Iterator[tuple[str, str]]:
    """Yield (heading, body_text) sections. A new section starts at each Heading-styled paragraph."""
    heading = "(intro)"
    buf: list[str] = []
    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style = (para.style.name if para.style else "") or ""
        if style.startswith("Heading") or style == "Title":
            if buf:
                yield heading, "\n".join(buf)
                buf = []
            heading = text
        else:
            buf.append(text)
    if buf:
        yield heading, "\n".join(buf)


def header(meta: dict, heading: str) -> str:
    return f"[Carrier: {meta['carrier']} | Type: {meta['doc_type']} | Doc: {meta['doc_name']}]"


def extract(docs_dir: str | None = None, out: str | None = None) -> Path:
    cfg = load_config()
    docs_root = Path(docs_dir) if docs_dir else cfg.docs_dir
    out_path = Path(out) if out else cfg.data_dir / "chunks.jsonl"

    files = sorted(docs_root.rglob("*.docx"))
    files = [f for f in files if not f.name.startswith("~$")]  # skip Word lockfiles
    if not files:
        raise FileNotFoundError(
            f"No .docx found under {docs_root}. Drop the ~10 Verizon docs there first."
        )

    n_chunks = 0
    with open(out_path, "w") as fh:
        for f in files:
            meta = infer_metadata(f, docs_root)
            doc = Document(str(f))
            for i, (heading, body) in enumerate(_iter_sections(doc)):
                if len(body) < 40:  # skip trivial fragments
                    continue
                rec = {
                    "id": f"{meta['doc_name']}::{i}",
                    "carrier": meta["carrier"],
                    "doc_type": meta["doc_type"],
                    "doc_name": meta["doc_name"],
                    "heading": heading,
                    "header": header(meta, heading),
                    "text": body,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
    print(f"[extract] {len(files)} docs -> {n_chunks} sections -> {out_path}")
    return out_path
