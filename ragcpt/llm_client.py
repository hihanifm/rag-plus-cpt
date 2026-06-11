"""One OpenAI-compatible client. Teacher = OpenAI GPT; same client also talks to vLLM endpoints
(tuned/base Qwen) since vLLM exposes the OpenAI API. Frontends never call OpenAI directly."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from .config import load_config, require_env


def _client(base_url: str, api_key: str) -> OpenAI:
    # Elastic by default: on 429 / transient errors the SDK backs off (honoring Retry-After) and
    # retries, so high concurrency self-throttles instead of dropping work. Tune in pipeline.yaml.
    cfg = load_config()
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=cfg.get("gen_qa", "max_retries", default=8),
        timeout=cfg.get("gen_qa", "request_timeout", default=60),
    )


def teacher_client() -> tuple[OpenAI, str]:
    """Returns (client, model) for the teacher (OpenAI GPT)."""
    cfg = load_config()
    key = require_env(cfg.get("teacher", "api_key_env", default="OPENAI_API_KEY"))
    base = cfg.get("teacher", "base_url", default="https://api.openai.com/v1")
    return _client(base, key), cfg.teacher_model


def judge_client() -> tuple[OpenAI, str]:
    """Cheaper model for grounding cleanup + eval grading."""
    cfg = load_config()
    key = require_env(cfg.get("teacher", "api_key_env", default="OPENAI_API_KEY"))
    base = cfg.get("teacher", "base_url", default="https://api.openai.com/v1")
    return _client(base, key), cfg.judge_model


def gold_client() -> tuple[OpenAI, str]:
    """Gold-set drafting — prefer a DIFFERENT model family than the teacher (cross-model, avoids
    teaching-to-the-test). If the gold API key is set, use it (Anthropic by default); otherwise
    fall back to the teacher so the pipeline still works with a single key."""
    cfg = load_config()
    key = os.environ.get(cfg.get("gold", "api_key_env", default="ANTHROPIC_API_KEY"))
    if not key:
        print("[gold] no gold API key set — falling back to the teacher model for gold drafting "
              "(set ANTHROPIC_API_KEY for cross-model gold).")
        return teacher_client()
    base = cfg.get("gold", "base_url", default="https://api.anthropic.com/v1")
    return _client(base, key), cfg.get("models", "gold", default="claude-sonnet-4-5")


def endpoint_client(which: str) -> tuple[OpenAI, str]:
    """which='tuned' or 'base' — a vLLM OpenAI-compatible endpoint serving Qwen3-14B."""
    cfg = load_config()
    base_env = cfg.get("serving", f"{which}_base_url_env")
    key_env = cfg.get("serving", f"{which}_api_key_env")
    base = os.environ.get(base_env)
    if not base:
        raise RuntimeError(
            f"{base_env} not set — bring up the {which} model with `ragcpt serve` "
            f"(or run_mvp) and put its URL in .env."
        )
    key = os.environ.get(key_env, "EMPTY")
    return _client(base, key), cfg.base_model


def chat(client: OpenAI, model: str, messages: list[dict], **kw: Any) -> str:
    """Plain chat completion → text."""
    resp = client.chat.completions.create(model=model, messages=messages, **kw)
    return resp.choices[0].message.content or ""


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def chat_json(client: OpenAI, model: str, messages: list[dict], **kw: Any) -> Any:
    """Chat completion expected to return JSON. Tries response_format=json_object first,
    falls back to extracting the first JSON block from the text."""
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, response_format={"type": "json_object"}, **kw
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        text = chat(client, model, messages, **kw)
        m = _JSON_BLOCK.search(text)
        if not m:
            raise ValueError(f"No JSON found in model output:\n{text[:500]}")
        return json.loads(m.group(0))
