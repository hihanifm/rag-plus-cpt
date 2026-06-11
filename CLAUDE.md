# CLAUDE.md — rag-plus-cpt

Project conventions and context for Claude Code. Full design is in `plan.md`.

## What this is
Stage 1 of a 2-pass telecom assistant: fine-tune **Qwen3-14B-Instruct** to be **carrier-aware**
(Verizon/AT&T/T-Mobile configs on top of 3GPP) via **one LoRA SFT run**; a later **RAG** pass
validates exact facts with citations. Currently at **Stage 0 (MVP)** — prove the training signal on
~10 docs before building scaffolding.

## Guiding principles (do not violate)
- **Simplicity & flexibility over cost. LLM-first.** Cost is not a constraint.
- **LLM-first:** use an LLM wherever a task needs *understanding* content (find intro, derive
  metadata, generate/judge QA). Plain code for plumbing only (walk files, write JSONL, launch jobs).
- **Simple over clever:** fewest moving parts; one runner, one data format, one client. No
  speculative abstractions.
- **Flexible by config:** model/host/env are config switches (`pipeline.yaml`), never hardcoded.

## Architecture — one brain, thin frontends
- `ragcpt/` is **the engine** (a library). Every frontend is a thin wrapper calling the same
  functions — **no business logic in any frontend.**
- Frontends: `run_mvp.py` (Stage 0 orchestrator) and `cli.py` (`ragcpt` Typer CLI). Streamlit +
  FastAPI come later (Stage A/B) — don't build them until the MVP signal is proven.
- **Model roles, decoupled:** teacher = **OpenAI GPT** (`OPENAI_API_KEY`) for enrich/QA/gold-draft;
  student/base/serving = **Qwen3-14B bf16** everywhere. Teacher facts are doc-grounded via the
  auto-cleaner — the teacher contributes formulation/reasoning, not its own facts.

## Pipeline stages (in `ragcpt/`)
`extract` → `gen_qa` → `filter_qa` (auto-cleaner: keep only answers supported by the source) →
`build_sft` (messages JSONL, `dataset_mix` ratios) → `train` (LLaMA-Factory LoRA) → `merge` →
`serve` (vLLM, OpenAI-compatible) → `evaluate`. `review.py` = human-in-loop (spot-check the cleaner;
build the gold set).

## Conventions
- **SFT data = `messages` (OpenAI) JSONL.** Most portable; the trainer applies Qwen3's chat template.
- **Coverage ≠ ratio:** every doc section is always a raw sample (full coverage); `dataset_mix` sets
  proportions by how much QA we generate.
- **`qa_unknown` samples** teach the model to defer to RAG instead of hallucinating exact values.
- **Structured `<think>`** is short and fixed (baseline / deviation / confidence), filled per
  question — never identical boilerplate.
- **Gold eval is sacred:** teacher may draft, human corrects; trick/comparison/unknown items are
  hand-written (leakage guard). Define the Pass/Kill threshold (`pipeline.yaml: success`) *before*
  training.

## Run / test
- Dev (Mac): `pip install -r requirements.txt` → `pytest -q`. Data prep needs `OPENAI_API_KEY`.
- Train/serve: GPU box only (Lambda H100 / DGX) → `pip install -r requirements-train.txt`.
- **Two channels:** code via `git clone`; **docs + `.env` via `scp`/`rsync`** (gitignored, never in
  GitHub). Pull `models/` + `data/eval_report_*.json` off an ephemeral box before destroying it.

## Don't
- Don't commit `.env`, `docs/*`, `models/`, or generated `data/*.jsonl` (all gitignored).
- Don't add Streamlit/FastAPI/profiles before the Stage 0 decision gate passes.
- Don't hand-code parsing heuristics where an LLM call is more flexible.
