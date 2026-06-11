"""Teacher (OpenAI GPT) generates QA per section: plain / structured-<think> / unknown.

- plain   : instruction -> direct answer (fast stage-1 mode)
- think   : instruction -> short structured <think> (baseline/deviation/confidence) + answer
- unknown : instruction whose exact answer is NOT in the section -> a "defer to RAG" response
            (teaches the model where its grounded knowledge ends — the RAG handoff)

Answers must be grounded in the section. The auto-cleaner (filter_qa) verifies that afterwards.
Output: data/qa_raw.jsonl  (records tagged with `qa_type` + source chunk id/text).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .config import load_config
from .io_utils import read_jsonl, write_jsonl
from .llm_client import chat_json, teacher_client

_SKELETON = (Path(__file__).parent / "reasoning_skeleton.md").read_text()

_PLAIN_THINK_PROMPT = """You are creating supervised fine-tuning data from a single section of a \
carrier telecom document. Use ONLY facts stated in the section — do not add outside knowledge.

Carrier: {carrier}
Document: {doc_name} ({doc_type})
Section heading: {heading}

SECTION TEXT:
\"\"\"
{text}
\"\"\"

Focus on **substantive carrier-technical content**: requirements, procedures, message flows,
parameters/values, configurations, device/network behaviors, test conditions and expected results.

IMPORTANT — skip non-technical boilerplate. If this section is mostly **document metadata**
(revision history, version numbers, dates, authors, change logs, table of contents, scope/
applicability statements, acronym/glossary lists, references, legal/confidentiality notices) and
contains no substantive carrier-technical content, return an EMPTY list: {{"pairs": []}}.
Never ask about revision dates, version numbers, who edited the doc, or what an acronym expands to.

Otherwise produce up to {n_plain} PLAIN and up to {n_think} THINK question-answer pairs about the
technical content of THIS section.
- Questions must be answerable solely from the section text and be about carrier behavior/requirements.
- PLAIN: a direct concise answer, no reasoning shown.
- THINK: the answer preceded by a short structured <think> block in exactly this shape:
{skeleton_think}

Return JSON: {{"pairs": [{{"qa_type": "plain"|"think", "question": "...", "answer": "..."}}]}}
For THINK pairs, put the <think>...</think> block at the start of "answer", then the final answer.
"""

_UNKNOWN_PROMPT = """You are creating "insufficient evidence" training samples that teach a model to \
DEFER to retrieval (RAG) instead of guessing.

Carrier: {carrier}
Document: {doc_name}
Section heading: {heading}

SECTION TEXT:
\"\"\"
{text}
\"\"\"

If this section is only document metadata (revision history, ToC, glossary, dates, scope) with no
carrier-technical topic to anchor to, return EMPTY: {{"pairs": []}}.

Otherwise write {n_unknown} question(s) that sound like a user asking for an EXACT carrier-specific \
value (timer, ID, threshold, parameter) on the section's technical topic that is NOT actually stated \
in this section. The answer must be a polite refusal-to-guess that points to verifying the source \
document, e.g.:
"I don't have grounded carrier-specific information to answer that exactly. Please verify against the \
source document (RAG)."

Return JSON: {{"pairs": [{{"qa_type": "unknown", "question": "...", "answer": "..."}}]}}
"""

_THINK_SHAPE = """<think>
- 3GPP baseline: ...
- Carrier-specific deviation: ...
- Confidence: ...
</think>"""


def gen_qa(chunks: str | None = None, out: str | None = None, limit: int | None = None) -> Path:
    cfg = load_config()
    chunks_path = Path(chunks) if chunks else cfg.chunks_path
    out_path = Path(out) if out else cfg.qa_raw_path
    client, model = teacher_client()

    n_plain = cfg.get("gen_qa", "per_chunk_plain", default=2)
    n_think = cfg.get("gen_qa", "per_chunk_think", default=2)
    n_unknown = cfg.get("gen_qa", "per_chunk_unknown", default=1)
    max_chars = cfg.get("gen_qa", "max_chunk_chars", default=6000)

    concurrency = cfg.get("gen_qa", "concurrency", default=8)
    rows = list(read_jsonl(chunks_path))
    if limit:
        rows = rows[:limit]

    def process(ch: dict) -> tuple[dict, list[dict], bool]:
        """All teacher calls for one section. Returns (chunk, rows, failed). `failed` is True if any
        call errored (e.g. rate limit exhausted retries) so we can retry that section — never silently
        lose data. An empty result with failed=False is legitimate (metadata-only section)."""
        text = ch["text"][:max_chars]
        common = dict(carrier=ch["carrier"], doc_name=ch["doc_name"],
                      doc_type=ch.get("doc_type", ""), heading=ch["heading"], text=text)
        rows_out: list[dict] = []
        failed = False
        try:
            pt = chat_json(client, model, [{"role": "user", "content": _PLAIN_THINK_PROMPT.format(
                n_plain=n_plain, n_think=n_think, skeleton_think=_THINK_SHAPE, **common)}])
            rows_out += [_mk(ch, p) for p in pt.get("pairs", [])]
        except Exception as e:
            failed = True
            print(f"[gen_qa] plain/think failed for {ch['id']}: {e}")
        if n_unknown:
            try:
                uk = chat_json(client, model, [{"role": "user", "content": _UNKNOWN_PROMPT.format(
                    n_unknown=n_unknown, **common)}])
                rows_out += [_mk(ch, p) for p in uk.get("pairs", [])]
            except Exception as e:
                failed = True
                print(f"[gen_qa] unknown failed for {ch['id']}: {e}")
        return ch, rows_out, failed

    def run_pass(chunks: list[dict], workers: int, desc: str) -> tuple[list[dict], list[dict]]:
        out: list[dict] = []
        failed: list[dict] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process, ch) for ch in chunks]
            for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
                ch, ch_rows, ch_failed = f.result()
                out.extend(ch_rows)
                if ch_failed:
                    failed.append(ch)
        return out, failed

    out_rows, failed = run_pass(rows, concurrency, "gen_qa")
    # Retry pass at lower concurrency so transient rate-limit drops are recovered, not lost.
    if failed:
        print(f"[gen_qa] retrying {len(failed)} sections that errored (lower concurrency)...")
        retry_rows, still_failed = run_pass(failed, max(2, concurrency // 4), "gen_qa-retry")
        out_rows += retry_rows
        if still_failed:
            print(f"[gen_qa] ⚠️  {len(still_failed)} sections still failed after retry "
                  f"(consider lower concurrency or a higher-TPM model/tier).")

    n = write_jsonl(out_path, out_rows)
    print(f"[gen_qa] {len(rows)} sections -> {n} raw QA -> {out_path}")
    return out_path


def _mk(chunk: dict, pair: dict) -> dict:
    return {
        "qa_type": pair.get("qa_type", "plain"),
        "question": pair["question"],
        "answer": pair["answer"],
        "carrier": chunk["carrier"],
        "header": chunk["header"],
        "chunk_id": chunk["id"],
        "chunk_text": chunk["text"],
    }
