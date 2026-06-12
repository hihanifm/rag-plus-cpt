# Task: Fix chunking (coherent, structured-source-based)

> Handoff brief for a focused session. Read this + `CLAUDE.md` + `SPEC.md` (extract contract), then continue.

## Problem
The MVP pipeline currently chunks PDFs by **page-packing** (`ragcpt/extract.py` `_pack`/`_pdf_sections`):
whole pages merged to a soft ~2500-char target. On the real Verizon corpus (17 docs, some 328-page
spec PDFs) this produces **incoherent chunks** — confirmed by inspection:
- **Test cases are split mid-content** (e.g. `VZ_TC_USIMISIMINT_6234` cut at "if the UICC respon…").
- Repeated **header/footer noise** ("Verizon Wireless … / Test Plan / Page N of M") on every page.
- **Near-empty pages**, mangled **tables** (test steps are tables; `extract_text` flattens them).
- Wrong headings (`_first_line` grabs the running page header).
- No PDF outline/bookmarks (checked: 0 items).

Garbage chunks → garbage QA → weak training. **Train is on hold until chunking is coherent.**

## Decision (confirmed with user) — use STRUCTURED SOURCES, stop fighting PDFs
| Doc type | Source | Chunk = | Detect by |
|---|---|---|---|
| **Requirements** | manager's **JSON** (predefined grammar; already parses reqs into coherent text) | one grammar unit (e.g. one requirement) | `.json` input |
| **Test plans** | **Excel** export | **one row = one full test case** | filename suffix "test plan" / `.xlsx` |
| (PDF) | deprecated for these; keep only as fallback for unconverted docs | | |

Constraints (from user): chunk by the doc's own taxonomy; coherent, self-contained units;
**no overlap** (training data, not embedding/RAG); **no `VZ_REQ_` micro-splitting**. LLM-first OK
but structured sources make it unnecessary here.

## What's needed from the user (samples to design the readers)
1. **One manager JSON file** (requirements) → confirm grammar/schema: what is one unit, what fields.
2. **One test-plan Excel** (.xlsx) → confirm columns; one row = one test case (user confirmed).

## Implementation plan
1. Extend `ragcpt/extract.py` with two readers, routed by extension/suffix (keep the
   `[Carrier | Type | Doc | …]` metadata header + the `chunks.jsonl` record schema:
   `{id, carrier, doc_type, doc_name, heading, header, text}`):
   - **`_json_sections`** (requirements): parse manager's JSON per its grammar → one chunk per unit;
     `heading` = the unit's title/id; `text` = the coherent unit text.
   - **`_xlsx_sections`** (test plans): one row → one chunk; pick the columns that form the test-case
     body (purpose/steps/expected) as `text`, test-case id as `heading`; `doc_type=test_plan`.
   - Keep `_pdf_sections` as fallback only.
   - Add `openpyxl` (xlsx) to `requirements.txt`.
2. Update `SPEC.md` extract contract; the page-packing chunking TODO largely dissolves for these sources.
3. Regenerate: `python run_mvp.py prep` (gpt-5-mini, concurrency 8 — see pipeline.yaml; rate-limit
   lessons already baked in) → inspect chunks for coherence → then audit → curate → train.

## Current repo state (as of handoff)
- Clean-but-INCOHERENT dataset exists in `data/` (built on page-packed chunks) — will be regenerated.
- Engine = `ragcpt/` library; `run_mvp.py` orchestrates Stage 0; CLI `cli.py`. Tests: `pytest` (19).
- Teacher/judge = gpt-5-mini (`OPENAI_API_KEY`); gold = Claude if `ANTHROPIC_API_KEY` else teacher.
- Stage-A TODO backlog lives in `SPEC.md` (filter retry, taxonomy chunking, domain-agnostic skeleton).
- `plan.md` is FROZEN (ADR); put design changes in `SPEC.md` + code, not plan.md.

## First action for the new session
Ask the user for the two sample files (JSON + xlsx), inspect their shapes, then build the two readers.
Don't build blind — design from the real schemas.
