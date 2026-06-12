# SPEC — module contracts (living)

What each pipeline stage **guarantees**: inputs → outputs → invariants → acceptance. This is the
*what* (kept current with the code). The *why* (decisions/architecture) lives in **[plan.md](plan.md)**
(frozen). Verification = `tests/` + the eval gate.

Convention: each stage is a function in `ragcpt/<stage>.py`, called by `run_mvp.py` (Stage 0) or the
`ragcpt` CLI (Stage A+). Paths come from `pipeline.yaml`. The same `CONTRACT` block is mirrored in
each module's docstring.

---

## extract  (`ragcpt/extract.py`)
- **in:** `docs/**` — `.pdf` (text-based) / `.docx` / `.txt`
- **out:** `data/chunks.jsonl` — one record per section `{id, carrier, doc_type, doc_name, heading, header, text}`
- **invariants:**
  - metadata (carrier/doc_type/doc_name) inferred from path tokens (hint, not a parser)
  - PDFs: cover + table-of-contents pages skipped; remaining pages packed into ~2500-char chunks
  - sections shorter than 40 chars dropped
- **acceptance:** every doc represented; `header` present; chunks contain substantive text, not TOC.

## gen_qa  (`ragcpt/gen_qa.py`)
- **in:** `data/chunks.jsonl`  (teacher = OpenAI GPT)
- **out:** `data/qa_raw.jsonl` — `{qa_type, question, answer, carrier, header, chunk_id, chunk_text}`
- **invariants:**
  - per section: up to N `plain` + N `think` + N `unknown` (counts from `gen_qa` config)
  - `think` answers start with a short structured `<think>` (baseline/deviation/confidence)
  - `unknown` answers are deliberate "defer to RAG" refusals
  - **metadata-only sections (revision history, ToC, glossary, dates) yield ZERO pairs**
  - runs concurrently (`concurrency`); elastic on 429 (SDK `max_retries` + backoff)
- **acceptance:** spot-read shows substantive carrier-technical Qs (no revision-date/version trivia).

## filter_qa  (`ragcpt/filter_qa.py`) — the auto-cleaner
- **in:** `data/qa_raw.jsonl`
- **out:** `data/qa.jsonl` (kept) + `data/qa_rejected.jsonl` (dropped, with `reject_reason`)
- **invariants:**
  - `unknown` kept unjudged; others kept only if answer supported by `chunk_text`
  - questions deduped (normalized, first wins)
  - kept ∪ rejected accounts for all inputs (no silent drops; judge errors logged + rejected)
- **acceptance:** spot-check grounded precision ≥ ~90% (`run_mvp audit` / `ragcpt review`).

## build_sft  (`ragcpt/build_sft.py`)
- **in:** `data/chunks.jsonl` + `data/qa.jsonl`
- **out:** `data/sft.jsonl` (messages JSONL) + `data/dataset_manifest.json` + registers `carrier_sft`
- **invariants:**
  - **every** chunk becomes a raw-text sample (full coverage) — raw count is fixed = #chunks
  - QA volume scaled **relative to raw** to hit `dataset_mix` (coverage ≠ ratio):
    `qa_target[type] = raw_count * (mix[type] / mix[raw_text])`
    (e.g. raw=100, mix 30/30/30/10 → plain=100, think=100, unknown=33; raw share 100/333≈30%).
    Pools smaller than target are upsampled with replacement.
  - all samples are OpenAI `messages` format; QA samples carry the reasoning-skeleton system prompt
- **acceptance:** `final_counts` ≈ `dataset_mix`; raw_text count == chunk count.

## train  (`ragcpt/train.py`)  — GPU host
- **in:** registered `carrier_sft` + `configs/sft_lora.yaml`
- **out:** `models/<run>/run_<ts>/` — adapter, train_config.yaml, git_commit.txt, pipeline snapshot
- **invariants:** LoRA bf16; `--smoke` caps samples/steps; never mutates the base model.
- **acceptance:** smoke completes w/o OOM; loss decreases.

## merge / serve  (`ragcpt/merge.py`, `ragcpt/serve.py`)  — GPU host
- **merge:** adapter → `models/<run>/run_<ts>/merged/` (bf16).
- **serve:** `merged/` (or base, `--base`) → OpenAI-compatible vLLM endpoint; prints the `.env` URL.

## evaluate  (`ragcpt/evaluate.py`)
- **in:** `eval/gold.jsonl` + `eval/chat_probes.jsonl`, against a served endpoint (`base`|`tuned`)
- **out:** `data/eval_report_<which>.json`
- **invariants:**
  - knowledge Qs graded correct/incorrect; `unknown` Qs scored on deferral (not recall)
  - **FAIRNESS RULE:** base and tuned run under *identical* conditions — same system prompt, chat
    template, decoding params (`EVAL_DECODING`: temperature/max_tokens/seed), and gold set. Improvement
    must come from training, not prompt/decoding differences. Enforced via the single `_answer()` path;
    never special-case base vs tuned.
- **acceptance:** the Pass/Kill thresholds in `pipeline.yaml: success` (tuned ≫ base; defer ≥ 80%).

## review  (`ragcpt/review.py`) — human-in-loop
- **audit:** show 30 kept `(chunk,Q,A)` → judge ✓/✗ → grounded precision (checks the cleaner).
- **curate:** teacher drafts gold Q&A → approve/edit → append to `eval/gold.jsonl` (hard items
  hand-written for leakage safety).

## leakage guard  (`ragcpt/leakage.py`)
- **HARD RULE:** no gold question (or close paraphrase) may appear in training QA — gold must be
  independent of training data, or the eval overstates lift (train/test contamination).
- **Stage 0:** `find_leaks` does normalized-exact + high-threshold token-Jaccard (no deps); `evaluate`
  runs it and **warns** (never auto-deletes — a human drops the gold item or the training QA).
- **Stage A:** upgrade to true fuzzy (rapidfuzz / embeddings) for harder paraphrases.

---

### TODO (post-MVP / Stage A)
- **Domain-agnostic reasoning skeleton:** the corpus is NOT only telecom — ~half is Android-ecosystem
  (app behavior, services, settings, UX). The current skeleton forces a 3GPP frame
  (`reasoning_skeleton.md`, `gen_qa` `_THINK_SHAPE`), so for non-telecom reqs the teacher writes
  hollow "3GPP baseline: N/A" or invents fake 3GPP refs. Generalize to:
  `Default behavior` (3GPP for telecom; AOSP/Android for device/app; or general expectation) /
  `Carrier requirement` / `Confidence`. Make `gen_qa` **domain-aware** (infer domain, ground "Default
  behavior" in the right baseline, never invent 3GPP for non-telecom) and add a **free-form `domain`
  tag** on QA records (semantic filtering later; no fixed enum). **Requires regenerating the dataset.**
  Touches: `ragcpt/reasoning_skeleton.md` (system prompt + training `<think>`), `ragcpt/gen_qa.py`.
- **filter_qa retry pass:** add app-level retry for failed batch judges (mirror gen_qa's
  retry-failed-sections) so valid pairs aren't dropped on a stubborn 429 after SDK retries exhaust.
  MVP relies on SDK-level retry (`max_retries`) only. (`ragcpt/filter_qa.py` `judge_group`.)
- **Taxonomy-aware chunking:** chunk by the document's section hierarchy so each chunk is coherent +
  self-contained. Merge too-small subsections; split oversized sections at natural sub-boundaries.
  **No `VZ_REQ_` splitting** (too granular). **No overlap** (training data, not embedding/RAG).
  LLM-driven segmentation OK. Currently packs whole pages to ~2500 chars. (`ragcpt/extract.py` `_pack`.)

### Evidence (Stage 0, no infra)
`data/` is the evidence folder and records what produced a result: `chunks/qa_raw/qa/qa_rejected/sft`
JSONL, `dataset_manifest.json`, `eval_report_*.json`, and `prep_snapshot.json` (timestamp + git
commit + pipeline.yaml + counts). Generated artifacts stay local (gitignored).
