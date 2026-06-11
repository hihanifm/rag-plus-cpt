"""Load pipeline.yaml + .env, resolve paths. Single source of config truth."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv optional; env may already be set
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]

    # ---- paths (absolute) ----
    @property
    def docs_dir(self) -> Path:
        return self._path("paths", "docs")

    @property
    def data_dir(self) -> Path:
        return self._path("paths", "data")

    @property
    def eval_dir(self) -> Path:
        return self._path("paths", "eval")

    @property
    def models_dir(self) -> Path:
        return self._path("paths", "models")

    def _path(self, *keys: str) -> Path:
        node: Any = self.raw
        for k in keys:
            node = node[k]
        p = Path(node)
        return p if p.is_absolute() else (ROOT / p)

    # ---- convenience getters ----
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def base_model(self) -> str:
        return self.get("models", "base")

    @property
    def teacher_model(self) -> str:
        return self.get("models", "teacher")

    @property
    def judge_model(self) -> str:
        return self.get("models", "judge")


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Config:
    cfg_path = Path(path) if path else (ROOT / "pipeline.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw=raw)
    # ensure output dirs exist
    for d in (cfg.data_dir, cfg.models_dir, cfg.eval_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing env var {name}. Copy .env.example to .env and fill it in "
            f"(or export {name})."
        )
    return val
