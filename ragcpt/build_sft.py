"""Combine raw-text sections + cleaned QA into one messages-format SFT file.

Output `messages` (OpenAI) JSONL — most portable; LLaMA-Factory applies Qwen3's chat template.

Raw-text representation (Stage 0): each section becomes a "reading" sample — a generic user turn
asking for the section's content, assistant = the section text (with metadata header). This injects
familiarity inside the SFT run without a separate `pt` stage. It's the pragmatic single-run choice;
if we later want true next-token-on-raw-text, that's a `stage: pt` mix (noted in the plan).

`dataset_mix` in pipeline.yaml sets target proportions. Every section is always included as a raw
sample (full coverage); we down/upsample QA to hit the ratio (coverage != ratio).
Also writes data/dataset_manifest.json (counts + provenance).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .config import load_config
from .io_utils import read_jsonl, write_jsonl

_SYSTEM = (Path(__file__).parent / "reasoning_skeleton.md").read_text()

_READ_PROMPTS = [
    "Recall the {carrier} documentation for this section.",
    "What does the {carrier} document say in this section?",
    "Provide the {carrier} configuration details from this section.",
]


def _raw_sample(chunk: dict, rng: random.Random) -> dict:
    prompt = rng.choice(_READ_PROMPTS).format(carrier=chunk["carrier"])
    user = f"{chunk['header']} {prompt}"
    assistant = f"{chunk['header']}\n# {chunk['heading']}\n{chunk['text']}"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ], "sample_type": "raw_text"}


def _qa_sample(qa: dict) -> dict:
    user = f"{qa['header']} {qa['question']}"
    return {"messages": [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": qa["answer"]},
    ], "sample_type": f"qa_{qa['qa_type']}"}


def build_sft(chunks: str | None = None, qa: str | None = None,
              out: str | None = None, seed: int = 0) -> Path:
    cfg = load_config()
    chunks_path = Path(chunks) if chunks else cfg.data_dir / "chunks.jsonl"
    qa_path = Path(qa) if qa else cfg.data_dir / "qa.jsonl"
    out_path = Path(out) if out else cfg.data_dir / "sft.jsonl"
    rng = random.Random(seed)

    chunk_rows = list(read_jsonl(chunks_path))
    qa_rows = list(read_jsonl(qa_path))

    raw = [_raw_sample(c, rng) for c in chunk_rows]                       # full coverage
    qa_by_type: dict[str, list[dict]] = {"plain": [], "think": [], "unknown": []}
    for q in qa_rows:
        qa_by_type.get(q["qa_type"], qa_by_type["plain"]).append(_qa_sample(q))

    mix = cfg.get("dataset_mix", default={})
    samples = _apply_mix(raw, qa_by_type, mix, rng)
    rng.shuffle(samples)

    n = write_jsonl(out_path, samples)
    manifest = _manifest(out_path, raw, qa_by_type, samples)
    (cfg.data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    _register_dataset(cfg, out_path)
    print(f"[build_sft] {n} samples -> {out_path}  ({manifest['final_counts']})")
    return out_path


def _apply_mix(raw, qa_by_type, mix, rng):
    """Hit dataset_mix proportions while keeping FULL raw coverage (every section included once).

    Raw is fixed at len(raw); QA volume is scaled *relative to raw* so the final ratio matches `mix`:

        base_per_weight = raw_count / raw_weight
        qa_target[type] = round(base_per_weight * mix[type])
                        = raw_count * (mix[type] / raw_weight)

    Example — raw=100, mix raw/plain/think/unknown = 30/30/30/10:
        base_per_weight = 100/30 = 3.33
        plain=100, think=100, unknown=33  -> total QA=233, total=333, raw share=100/333≈30%.
    QA pools smaller than target are upsampled with replacement; larger are sampled without.
    """
    raw_w = mix.get("raw_text", 30) or 1
    pools = {"raw_text": raw, "qa_plain": qa_by_type["plain"],
             "qa_think": qa_by_type["think"], "qa_unknown": qa_by_type["unknown"]}
    out: list[dict] = list(raw)  # every section included once
    base_per_weight = len(raw) / raw_w
    for name in ("qa_plain", "qa_think", "qa_unknown"):
        pool = pools[name]
        if not pool:
            continue
        target = int(round(base_per_weight * (mix.get(name, 0) or 0)))
        if target <= 0:
            continue
        if target <= len(pool):
            out += rng.sample(pool, target)
        else:  # upsample with replacement to hit ratio
            out += [rng.choice(pool) for _ in range(target)]
    return out


def _manifest(out_path, raw, qa_by_type, samples):
    from collections import Counter
    counts = Counter(s["sample_type"] for s in samples)
    return {
        "output": str(out_path),
        "available": {"raw_text": len(raw), "qa_plain": len(qa_by_type["plain"]),
                      "qa_think": len(qa_by_type["think"]), "qa_unknown": len(qa_by_type["unknown"])},
        "final_counts": dict(counts),
        "total": len(samples),
    }


def _register_dataset(cfg, out_path: Path):
    """Register with LLaMA-Factory (data/dataset_info.json) in sharegpt-ish messages format."""
    info_path = cfg.data_dir / "dataset_info.json"
    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text())
    info["carrier_sft"] = {
        "file_name": out_path.name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
    }
    info_path.write_text(json.dumps(info, indent=2))
