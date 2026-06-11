# Plan: Carrier-Specific SFT Pipeline for Qwen3-14B (DGX Spark) + Web Control Plane

## Guiding principle (applies to the whole project)

**Simplicity and flexibility over cost. LLM-first.** The user's north star: keep each piece simple,
keep the pipeline flexible (swap models/hosts/profiles freely), and never trade those away to save
money — cost is not a constraint (lab prod infra; external credits fine).
- **LLM-first:** use LLM inference wherever a task requires *understanding* content (find the intro,
  derive metadata, generate QA, judge grounding). Reserve plain code for **plumbing only** (walk
  files, write JSONL, merge, launch jobs). When in doubt, ask the LLM rather than hardcode a rule.
- **Simple over clever:** fewest moving parts that meet the need; one runner, one data format, one
  client. Don't add stages/abstractions speculatively.
- **Flexible by config:** model, host, and environment are profile/config switches, never hardcoded.

## Context

Goal: make Qwen3-14B-Instruct **carrier-aware** (Verizon, AT&T, T-Mobile configs/behaviors) on top
of its 3GPP baseline → fast carrier-aware answers. A separate **RAG** second pass validates
specifics with citations. This plan = **stage 1: the tuned model + the pipeline to build it.**

**Hardware:** training on a **DGX Spark** — GB10, **128 GB unified** memory (CPU+GPU shared).
Full-parameter 14B needs ~224 GB → won't fit; offload pointless on a unified pool. So training is
**LoRA, bf16**. Two work contexts: **personal = Mac** (external dev — CLI + Streamlit); **office =
headless Linux box over SSH** (internal — driven by **CLI or FastAPI**, never a local-only UI). The
remote office box is why a server/client web control plane (FastAPI) exists.

## Success criteria & kill switch (define BEFORE training)

The MVP exists to answer one falsifiable question: **does LoRA SFT on carrier docs measurably beat
base Qwen3-14B?** Write the bar before you train, so the result isn't vibes:

- **Pass (proceed to scale + scaffolding):** on the ~40 hand-written gold Qs, **tuned correct ≥ ~2×
  base** (e.g. ≥28/40 vs base ~14) **AND** insufficient-evidence Qs **deferred ≥ 80%** (not
  hallucinated) **AND** **zero regressions** on chat/thinking probes.
- **Kill / rethink (do NOT build the factory):** tuned ≈ base (lift within noise). Then stop and
  reconsider — likely **RAG-only** suffices, or you need more data / different method. Do not pour
  weeks into CLI/UI/profiles on an unproven signal.
- Numbers are starting targets — adjust once you see baseline, but **commit to a threshold up front.**

## Architecture — one brain, thin frontends

**The engine is a library; every UI is a thin wrapper over the same functions.** Logic lives once;
the CLI, Streamlit, and FastAPI all just *call it*. No business logic in any frontend.

1. **Pipeline engine** (`ragcpt/` package) — each stage is a **library function** driven by
   `pipeline.yaml`, writing artifacts + a **manifest** (resumable, inspectable). The user **never
   edits or runs the internal modules directly.**
2. **Frontends over the engine** (in build order):
   - **CLI (`ragcpt …`, Typer)** — *primary scripting surface.* One console command, a subcommand per
     stage, every option a flag (`--profile`, `--docs`, `--mix`, `--run-name`, `--dry-run`,
     `--inspect`). Auto `--help`. You script entirely through this — no touching `.py` files.
     **Phase A.**
   - **Streamlit panel** — a *cheap visual* over the same functions: widgets for the same options,
     Run buttons, log + artifact viewers. Mainly for **personal Mac dev** (local, or headless on the
     Linux box via `--server.address 0.0.0.0` + `ssh -L` if wanted). **Phase A**, about as little code
     as the CLI.
   - **FastAPI backend + web frontend** — the *production* control plane for the **office** (drive the
     headless Linux box remotely): jobs, log streaming, profile switch, chat proxy, polished UI.
     **Phase B.**
3. **Model roles + environment profiles** — one **OpenAI-compatible** `llm_client.py`. Two distinct
   model roles, decoupled:
   - **Teacher** (enrichment, QA, reasoning traces) = a **strong model**: Claude/GPT externally
     (user has accounts), strong internal model internally. Facts stay doc-grounded via the
     grounding filter, so the teacher contributes *formulation + reasoning quality*, not its own facts.
   - **Student / base / serving** = **Qwen3-14B bf16** — what we train and deploy, everywhere.

   Every stage targets an **environment profile**, switchable in `pipeline.yaml` or the UI:
   - **`external`** (Mac dev): teacher = **OpenAI GPT** (`OPENAI_API_KEY`); training = **Lambda Labs
     1×H100 80GB PCIe ($3.29/hr)** running the same LLaMA-Factory (single GPU — no distributed setup;
     config-swappable); serving = vLLM on that same box — small reference doc set, fast iter.
   - **`internal`** (lab): teacher = strong internal model; training = **DGX** (same LLaMA-Factory);
     serving = internal vLLM — hundreds of docs/test plans.
   Same runner both sides — only the *host* + teacher endpoint swap. A profile sets defaults for
   {teacher endpoint, train host, serving endpoint, docs root, scale knobs}; stages can override.

## Training design — one SFT run, mixed dataset

The mix is a **config knob** (`dataset_mix` in `pipeline.yaml`), not a hardcoded split:

```yaml
dataset_mix:        # starting point — eval-tuned, not fixed
  raw_text: 30      # every section still included; ratio set by how much QA we generate
  qa_plain: 30
  qa_think: 30
  qa_unknown: 10
```

| Sample type | Purpose | Form |
|---|---|---|
| **Raw text** (every doc section) | familiarity + full coverage (CPT-equivalent) | completion, no prompt |
| **QA + structured think** | retain `<think>` + teach the reasoning strategy | instr→`<think>`+answer |
| **QA plain** | fast direct answers (stage-1 mode) + chat repair | instr→answer |
| **QA unknown / insufficient-evidence** | teach the model to **defer to RAG** instead of hallucinating | instr→"not enough grounded info, verify in source" |

- **Coverage ≠ ratio:** *every* section is always included as a raw sample (full coverage); the
  *proportion* is controlled by how much QA we generate. So we keep coverage **and** a tunable mix —
  no conflict. Start ~30% raw, then let the **chat/thinking eval guard** drive it: push raw text up
  until eval degrades, back off. Never a blind cap.
- "Text-only training" == CPT machinery → raw samples give familiarity *inside* the SFT run; no
  separate CPT stage. QA samples auto-repair chat drift; no separate repair stage.
- **`qa_unknown` is the RAG handoff:** carrier docs are incomplete, so the model must learn where its
  grounded knowledge ends and defer to the RAG second pass. Prevents confident hallucination on
  exact values it was never reliably taught. This is the brake pedal.
- **Auto-cleaner (grounding check) is load-bearing — spec it, don't hand-wave it.** Garbage QA →
  garbage model, so this gate decides the project. The auto-cleaner is an LLM that keeps a QA pair
  **only if the answer is supported by its source paragraph** (`{grounded: bool, reason}` given
  (chunk, Q, A); reject on false). Dedup near-identical Qs. **Then spot-check the auto-cleaner** with
  `ragcpt review`: it shows 30 *kept* `(chunk,Q,A)`, you just judge ✓/✗ (no editing), it prints a
  score = ✓/30. ≥~90% → trust the full set; lower → tighten the cleaner's prompt and re-check.
  A one-time check on the checker, not a gold set. Don't scale QA generation until it passes.
- **Structured think — short and boring** (harder to ritualize into fake reasoning). Same skeleton as
  the system prompt, filled per question, never identical boilerplate:
  ```
  <think>
  - 3GPP baseline:
  - Carrier-specific deviation:
  - Confidence:
  </think>
  ```
- Exact recall/citations are **RAG's job** in stage 2 — the model only needs familiarity + reasoning.

## Metadata — LLM-enriched, cached as sidecars

Per the LLM-first principle, metadata comes from the model, not path-parsing:
- For **every** doc, the LLM derives `{carrier, doc_type, doc_name, topic, summary}`. Path is a
  *hint* in the prompt, not a parser → any folder layout works.
- The LLM also **locates the real introduction** (skipping TOC / revision history / index) — no
  hardcoded style heuristics; it reads the structure and decides.
- Cached as a **sidecar `<doc>.meta.json` next to each doc**; re-runs skip when present and the
  `content_hash` matches (docs ~never change). Idempotent, overnight-safe.
- Metadata is baked **into training text** (compact header `[Carrier|Type|Doc|Topic]` on every
  sample → model attributes facts) **and** kept as **structured fields** (RAG/eval filtering).

## Decisions (confirmed with user)

| Choice | Value |
|---|---|
| Base model | Qwen3-14B-**Instruct**, bf16 |
| Training | **LoRA** bf16, all linear targets, **one SFT run** (raw-text + QA mix) |
| Train host | **same LLaMA-Factory both profiles** — DGX (internal) or **Lambda Labs 1×H100 80GB PCIe** (external default; swappable) |
| Dataset mix | **config knob** (`dataset_mix`): raw / qa_plain / qa_think / qa_unknown — eval-tuned, all sections always covered |
| Think format | **short structured** `<think>` (baseline / deviation / confidence), per-question, never boilerplate |
| Unknown samples | **`qa_unknown`** — model defers to RAG instead of hallucinating exact values (the RAG handoff) |
| Run tracking | every run → `models/<name>/run_<ts>/` with adapter, merged, train_config, dataset_manifest, eval_report, git_commit |
| Teacher model | **strong model** (OpenAI GPT ext via `OPENAI_API_KEY`; strong internal model in lab) — decoupled from student |
| Student/base/serving | **Qwen3-14B bf16** — trained + deployed everywhere |
| Environments | **`external` (Mac → rented GPU + Claude/GPT)** ↔ **`internal` (lab self-hosted)**; per-stage switchable |
| Doc sets | external: ~10 Verizon reqs + ~10 test plans (dev); internal: hundreds of docs + test plans |
| Metadata | LLM-enriched every doc, LLM finds intro, sidecar cache + hash |
| SFT data format | **`messages` (OpenAI) JSONL** — most portable; trainer applies Qwen3 chat template |
| Engine | `ragcpt/` **library** (one brain); UIs are thin wrappers — no logic duplicated |
| CLI | **`ragcpt` (Typer)** — primary scripting surface, flag per option, auto-help. **Phase A** |
| Streamlit | thin visual panel over the same fns — personal **Mac** dev (or headless via `ssh -L`). **Phase A** |
| Control plane | **FastAPI backend + web frontend** — **office** (drive headless Linux box), chat tab. **Phase B** |
| Verify chatbot | chat against the served tuned model (Streamlit chat in A, full chat tab in B) |
| Runner | **LLaMA-Factory** `stage: sft` — run on whichever host the profile points to (DGX / rented GPU) |
| Serving | OpenAI-compatible endpoint for the tuned model (internal vLLM / hosted) — in scope for verify |
| Recall | **RAG** second pass (deferred) |
| Tests | **pytest** per stage |
| Validation | baseline-first + gold eval (knowledge + chat + thinking) **and** interactive chat verify |

## Repo structure

```
rag-plus-cpt/
├── plan.md                      # copy of this plan
├── pipeline.yaml                # profiles (external/internal) + knobs: endpoints, dataset_mix, budgets, paths
├── .env.example                 # per-profile base_url / model / key (lab vs external)
├── pyproject.toml               # console entrypoint: `ragcpt` → ragcpt.cli:app
├── requirements.txt
├── ragcpt/                      # THE ENGINE (library) — never run/edited directly by the user
│   ├── llm_client.py            # OpenAI-compatible client; resolves active profile
│   ├── enrich.py                # LLM → <doc>.meta.json sidecar (finds intro, derives metadata, hash-cached)
│   ├── extract.py               # docx + sidecar → raw_text.jsonl (per-section, metadata header)
│   ├── gen_qa.py                # strong teacher → Q→A per chunk: plain / structured-<think> / unknown (per dataset_mix)
│   ├── filter_qa.py             # auto-cleaner: LLM keeps QA only if answer supported by source paragraph + dedupe → qa.jsonl
│   ├── build_sft.py             # raw_text + qa → sft.jsonl (messages JSONL) + dataset_manifest.json
│   ├── train.py                 # LLaMA-Factory LoRA SFT on the profile's host (DGX / rented GPU via SSH)
│   ├── merge.py                 # merge LoRA → bf16
│   ├── serve.py                 # serve merged model as OpenAI-compatible vLLM endpoint
│   ├── evaluate.py              # base vs run on gold + chat_probes; writes eval_report.json
│   ├── review.py                # human-in-loop helper, 2 modes: judge ✓/✗ (spot-check auto-cleaner) OR edit/approve (build gold)
│   ├── manifest.py              # run-versioning + resumable state
│   └── reasoning_skeleton.md    # shared short-think structure == inference system prompt
├── cli.py                       # Typer CLI: `ragcpt enrich|extract|gen-qa|build|train|merge|serve|eval|chat`
│                                #   + global flags (--profile --docs --mix --run-name --dry-run --inspect). PHASE A
├── app_streamlit.py             # thin Streamlit panel over ragcpt fns; personal Mac dev (or headless via ssh -L). PHASE A
├── configs/
│   ├── sft_lora.yaml            # stage: sft  (+ sft_lora_smoke.yaml)
├── eval/
│   ├── gold.jsonl               # teacher-DRAFTS + you-CORRECT (via review.py); trick/comparison/unknown
│   │                            #   items HAND-WRITTEN (leakage guard). Categories: facts, comparisons,
│   │                            #   "what changed from 3GPP?", trick, insufficient-info, messy chat
│   └── chat_probes.jsonl        # chat + thinking guard prompts
├── backend/                     # PHASE B — FastAPI over ragcpt fns: jobs, log stream, profile switch, chat proxy
│   ├── app.py
│   └── jobs.py
├── frontend/                    # PHASE B — production web UI + CHAT tab (verify model)
├── models/                      # run-versioned: <name>/run_<ts>/{adapter,merged,train_config.yaml,
│                                #   dataset_manifest.json,eval_report.json,git_commit.txt}
├── tests/                       # pytest: parsing, metadata schema, grounding, sample format, manifest, profile, CLI
└── data/
    ├── dataset_info.json
    └── sft.jsonl                # generated   (per-doc metadata = <doc>.meta.json sidecars in the docs folders)
```

## Build sequence — ruthless MVP, then scaffold

**Stage 0 — MVP vertical slice (prove the signal, least code).** ~10 Verizon dev docs, **one
hardcoded path** (Mac + Lambda H100 + Claude teacher — **no profile abstraction yet**), **no CLI
polish / Streamlit / FastAPI / run-versioning yet.** Just the `ragcpt` functions, called from a
throwaway `run_mvp.py`:
1. `extract` → `raw_text.jsonl` (enrichment can be a stub/manual carrier tag for 10 docs).
2. `gen_qa` (OpenAI GPT teacher) → `filter_qa` (auto-cleaner) → **spot-check the cleaner** via
   `ragcpt review` (judge 30 kept pairs ✓/✗, score ≥~90%) → `build` → `sft.jsonl`.
3. Build `gold.jsonl` (~40) via `ragcpt review` (teacher drafts, you correct; trick/comparison/unknown
   **hand-written**) + `chat_probes.jsonl`. **Write the success/kill threshold down now.**
4. `eval --base-only` (baseline) → `train --smoke` → full → `merge` → `serve` → `eval` (base vs run).
5. **Decision gate:** hit the Pass bar? → continue to Stage A. Hit Kill? → stop, reconsider (RAG-only?).

**Stage A — wrap the proven engine (CLI + Streamlit + profiles + versioning):**
6. Package `ragcpt/` + `pyproject.toml` + `cli.py` (Typer): `ragcpt enrich|extract|gen-qa|build|
   train|merge|serve|eval|chat`, flagged options, auto-help. Now everything is scriptable, no `.py`
   editing. Add full `enrich` (LLM finds intro, hash-cached sidecars) + run-versioning (`manifest.py`).
7. Add the **`external`↔`internal` profile** abstraction (now that one path is proven) + scale to the
   full internal corpus.
8. **Streamlit** (`app_streamlit.py`) — same fns as widgets + log/artifact/eval viewers. Mac dev.

**Stage B — production control plane (wraps the same `ragcpt` fns, no logic duplication):**
10. `backend/app.py` + `jobs.py`: endpoints to trigger each stage, stream logs, report status, browse
    artifacts (sidecars, raw/qa/sft samples, eval results), **switch profile**, and **proxy chat** to
    the served model.
11. `frontend/`: per-stage pages (config knobs, Run, live logs, artifact viewers, eval side-by-side),
    a **profile selector** (external/internal), and a **Chat tab** to talk to the tuned model (applies
    the reasoning-skeleton system prompt) and verify responses. Served on the Linux box; opened from
    an office client browser (LAN or `ssh -L`).
12. `tests/`: pytest across stages run from the UI or CLI.

## Verification (end-to-end) — all via the `ragcpt` CLI

1. `pytest` green; `ragcpt --help` lists every stage.
2. `ragcpt enrich --docs <dir>` → sidecars correct; re-run is a no-op.
3. `ragcpt extract && ragcpt gen-qa && ragcpt build` → inspect `sft.jsonl`.
4. `ragcpt eval --base-only` (baseline) → `ragcpt train --run-name mvp --smoke` then full →
   `ragcpt serve --run <ts>`.
5. `ragcpt eval --run <ts>` → run beats base on gold (incl. **unknown** Qs answered with deferral,
   not hallucination) AND keeps chat + thinking; `eval_report.json` saved in the run dir.
6. `ragcpt chat --run <ts>` — manually probe the tuned model.
7. Streamlit (Phase A): from your **Mac** browser, run a stage, watch logs, inspect samples, chat with
   the model. Flip `--profile external↔internal` and confirm the same stages run elsewhere.

## Deferred / contingency

- **RAG second pass** (retrieval + cited validation) — the fact-recall safety net.
- Auth on the web UI (internal; `ssh -L` tunnel is the simple secure default for now).
- If recall weak: raise raw-text share / LoRA rank / epochs (watch chat+thinking guard).
- External GPU host default = **Lambda Labs H100-80GB** (config-swappable to RunPod for value, or
  Modal if you later want serverless scale-to-zero). One 80GB card runs train + serve.
- Managed fine-tune vendors (Together / Fireworks / Predibase) remain a fallback if raw-GPU babysitting
  becomes a hassle — would need a messages-JSONL push adapter then.
- enrich/QA/gold-draft (OpenAI GPT) are the only teacher touchpoints — base + serving never leave Qwen3-14B.
