# rag-plus-cpt — Stage 0 (MVP)

Carrier-aware Qwen3-14B via one LoRA SFT run. Stage 0 proves the signal on ~10 Verizon docs before
any UI. Full design: `plan.md`.

**One question Stage 0 answers:** *does LoRA SFT on carrier docs measurably beat base Qwen3-14B?*
Pass → build the rest. Kill (tuned ≈ base) → stop, go RAG-only. Thresholds live in `pipeline.yaml`
under `success:` — set real numbers before you train.

---

## ☀️ Morning checklist (do these 4 things)

1. **Drop the docs.** Put your ~10 Verizon `.docx` in `docs/` (subfolders by carrier are fine, e.g.
   `docs/Verizon/requirements/...` — metadata is inferred from the path; not required).
2. **Set the teacher key.** `cp .env.example .env` and put your `OPENAI_API_KEY` in it.
3. **Install + test (Mac).**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pytest -q          # should be green
   ```
4. **Buy Lambda credits** → launch **1× H100 80GB PCIe**, image **"Lambda Stack"** (see GPU steps).

---

## How Lambda actually works (read first)

Lambda is **not** an inference API — it's a **GPU box** (a cloud VM with an H100). You **SSH in and
run things there.** There is no Lambda key to call from your Mac.

- **OpenAI key** = the **teacher** API (real cloud service). Used by `gen_qa` / `filter` / `eval`
  grading — runs wherever you execute those commands.
- **Lambda box** = the hardware. You `git clone` this repo onto it and run `train` / `serve`.
- **Tuned-model endpoint** = **vLLM running on the Lambda box**, exposing an OpenAI-*compatible* URL
  like `http://<box-ip>:8000/v1`. Put that in `.env` as `TUNED_BASE_URL` (or reach it via
  `ssh -L 8000:localhost:8000 ubuntu@<box-ip>`).

### Two channels: code via git, data via scp
Docs + `.env` are **gitignored** — they never go to GitHub (public or private). Move them to the box
directly over SSH (`scp`/`rsync` use the same key):
```bash
# from your Mac → the box (bypasses GitHub):
scp -r ./docs      ubuntu@<box-ip>:~/rag-plus-cpt/
scp     .env       ubuntu@<box-ip>:~/rag-plus-cpt/
# pull results back BEFORE destroying the box:
scp -r ubuntu@<box-ip>:~/rag-plus-cpt/models ./
scp     ubuntu@<box-ip>:~/rag-plus-cpt/data/eval_report_*.json ./
```

**Simplest path — do everything on the box** (one machine, no syncing):
```bash
ssh ubuntu@<lambda-box-ip>
git clone https://github.com/hihanifm/rag-plus-cpt && cd rag-plus-cpt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt
cp .env.example .env && nano .env          # add OPENAI_API_KEY
# copy your ~10 Verizon .docx into docs/  (scp from your Mac)
python run_mvp.py prep                       # teacher = OpenAI; runs on the box
python run_mvp.py train --smoke && python run_mvp.py train
python -c "from ragcpt import merge; merge.merge('models/mvp/run_<ts>')"
python run_mvp.py serve-base &               # :8001 baseline
python run_mvp.py serve --run models/mvp/run_<ts> &   # :8000 tuned
# set BASE_BASE_URL / TUNED_BASE_URL in .env to the localhost ports, then:
python run_mvp.py eval --which base && python run_mvp.py eval --which tuned
```
⚠️ The box is **ephemeral** — `scp` the merged model + `data/eval_report_*.json` back to your Mac
**before you destroy the instance** (they're gitignored / too big for git).

## Run order

### A. Data prep — on the Mac (needs `OPENAI_API_KEY`)
```bash
python run_mvp.py prep        # extract → gen_qa → filter_qa → build_sft  (writes data/sft.jsonl)
python run_mvp.py audit       # spot-check the auto-cleaner: judge 30 kept QA y/n → precision ≥~90%?
python run_mvp.py curate      # teacher drafts gold Q&A; you approve/edit → eval/gold.jsonl
```
Then **hand-add** the hard gold items (trick / comparison / unknown) directly to `eval/gold.jsonl`
(leakage guard). See `eval/gold.example.jsonl` for the format. Aim ~40 total, incl. a few
`"category": "unknown"` (exact values NOT in your docs → model should defer to RAG).

### B. Train + serve + eval — on the Lambda H100 box
```bash
# sync the repo + data to the box, then:
pip install -r requirements.txt -r requirements-train.txt

python run_mvp.py train --smoke           # ~20 steps, confirm it runs, no OOM
python run_mvp.py train                    # full LoRA SFT → models/mvp/run_<ts>/adapter
python -c "from ragcpt import merge; merge.merge('models/mvp/run_<ts>')"   # → merged/

# two serving terminals (OpenAI-compatible vLLM):
python run_mvp.py serve-base                            # untuned Qwen3-14B on :8001  (baseline)
python run_mvp.py serve --run models/mvp/run_<ts>       # tuned model on :8000

# put the two URLs in .env (BASE_BASE_URL / TUNED_BASE_URL), then:
python run_mvp.py eval --which base        # baseline scores
python run_mvp.py eval --which tuned       # tuned scores  → compare!
```

### C. Decision gate
Open `data/eval_report_base.json` vs `data/eval_report_tuned.json`. Hit the Pass bar
(`success:` in `pipeline.yaml`)? → proceed to **Stage A** (CLI/Streamlit/profiles, scale to the full
corpus). Tuned ≈ base? → **stop and rethink** (RAG-only? more data?).

---

## What's here (Stage 0)
- `ragcpt/` — the engine (library). Stages: `extract, gen_qa, filter_qa, build_sft, train, merge,
  serve, evaluate, review`. One OpenAI-compatible `llm_client`.
- `run_mvp.py` — throwaway orchestrator for the MVP.
- `cli.py` — `ragcpt` Typer CLI (Stage A surface; works now via `pip install -e .`).
- `pipeline.yaml` — all knobs (models, `dataset_mix`, train hparams, success thresholds).
- `tests/` — `pytest` for the pure-logic parts (parsing, mix ratios, IO).

## Notes / honest caveats
- **Teacher = OpenAI GPT** (`OPENAI_API_KEY`); **student/base/serving = Qwen3-14B** everywhere.
- **Raw-text representation:** Stage 0 injects raw sections as SFT "reading" samples (one mixed run,
  no separate `pt` stage). Pragmatic + simple; if it underperforms we can switch to a true `pt` mix.
- **Profiles** (external↔internal) are **not** built yet — Stage 0 is one hardcoded path on purpose.
  Added in Stage A once the signal is proven.
- `gen_qa` / `filter_qa` / `curate` / `eval` call the LLM and cost API $ (small for 10 docs).
