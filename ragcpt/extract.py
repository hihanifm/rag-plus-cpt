"""Parse source docs (.pdf / .docx / .txt) → sectioned chunks with metadata.

Stage 0: metadata (carrier / doc_type / doc_name) is inferred deterministically from the path +
filename (cheap, no API). Stage A swaps in LLM enrichment (find intro, richer metadata) behind the
same chunk schema. Output: data/chunks.jsonl, one record per section.

PDF sectioning is page-based: cover + table-of-contents pages are skipped, remaining pages are merged
into ~target-size chunks. Simple and robust for the Verizon spec PDFs; can be upgraded to LLM
segmentation later if QA quality needs it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from .config import load_config

_CARRIERS = {
    "verizon": "Verizon", "vzw": "Verizon", "vz": "Verizon",
    "att": "AT&T", "at&t": "AT&T",
    "tmobile": "T-Mobile", "tmo": "T-Mobile", "t-mobile": "T-Mobile",
}
_DOC_TYPES = {
    "requirement": "requirements", "req": "requirements",
    "testplan": "test_plan", "test-plan": "test_plan", "test_plan": "test_plan", "test": "test_plan",
    "spec": "spec", "specification": "spec",
}

_TARGET_CHARS = 2500
_TOC_LINE = re.compile(r"\.{4,}")          # dotted leaders, e.g. "1.1 INTRO ....... 4"


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


def header(meta: dict, heading: str) -> str:
    return f"[Carrier: {meta['carrier']} | Type: {meta['doc_type']} | Doc: {meta['doc_name']}]"


# ---------- per-format section readers: each yields (heading, body_text) ----------

def _docx_sections(path: Path) -> Iterator[tuple[str, str]]:
    from docx import Document
    doc = Document(str(path))
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


def _is_toc_or_cover(text: str, page_index: int) -> bool:
    if not text.strip():
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    toc_lines = sum(1 for ln in lines if _TOC_LINE.search(ln))
    if toc_lines >= 4:                       # TOC page (dotted leaders)
        return True
    if page_index == 0 and len(text) < 600 and "page 1 of" in text.lower():
        return True                          # cover page
    return False


def _pack(pages: list[str], target: int) -> Iterator[tuple[str, str]]:
    """Merge consecutive content pages into ~target-char chunks; heading = first line of chunk.

    TODO(Stage A): smarter, section-aware chunking. Current behavior packs whole PAGES to a soft
    ~target size — clubs small pages, never cuts mid-sentence, but is NOT aware of logical section
    headings (a chunk can straddle two sections; a long section splits at page breaks; an oversized
    page becomes one big chunk). Upgrade options: split on the docs' own numbering (`^\\d+(\\.\\d+)* `
    / `VZ_REQ_` IDs), or LLM-driven segmentation (LLM-first), or recursive paragraph/sentence split
    with small overlap. MVP page-packing is adequate for the litmus test.
    """
    buf: list[str] = []
    size = 0
    for t in pages:
        buf.append(t)
        size += len(t)
        if size >= target:
            body = "\n".join(buf).strip()
            yield _first_line(body), body
            buf, size = [], 0
    if buf:
        body = "\n".join(buf).strip()
        if body:
            yield _first_line(body), body


def _first_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip():
            return ln.strip()[:120]
    return "(section)"


def _pdf_sections(path: Path) -> Iterator[tuple[str, str]]:
    import pdfplumber
    content_pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            if _is_toc_or_cover(t, i):
                continue
            content_pages.append(t)
    yield from _pack(content_pages, _TARGET_CHARS)


def _txt_sections(path: Path) -> Iterator[tuple[str, str]]:
    text = path.read_text(errors="ignore")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    yield from _pack(paras, _TARGET_CHARS)


_READERS = {".docx": _docx_sections, ".pdf": _pdf_sections, ".txt": _txt_sections}


def extract(docs_dir: str | None = None, out: str | None = None) -> Path:
    cfg = load_config()
    docs_root = Path(docs_dir) if docs_dir else cfg.docs_dir
    out_path = Path(out) if out else cfg.chunks_path

    files = sorted(f for f in docs_root.rglob("*")
                   if f.suffix.lower() in _READERS and not f.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(
            f"No .pdf/.docx/.txt found under {docs_root}. Drop your source docs there first."
        )

    n_chunks = 0
    with open(out_path, "w") as fh:
        for f in files:
            meta = infer_metadata(f, docs_root)
            reader = _READERS[f.suffix.lower()]
            try:
                sections = list(reader(f))
            except Exception as e:
                print(f"[extract] FAILED {f.name}: {e}")
                continue
            for i, (heading, body) in enumerate(sections):
                if len(body) < 40:
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
