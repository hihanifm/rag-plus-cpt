"""`ragcpt` CLI (Typer) — thin wrapper over the ragcpt engine. Same functions as run_mvp.py, with
flags and auto-help. After `pip install -e .` you get the `ragcpt` command.

    ragcpt extract
    ragcpt gen-qa [--limit N]
    ragcpt filter
    ragcpt build
    ragcpt audit [--n 30]
    ragcpt curate [--k 1]
    ragcpt train [--run-name mvp] [--smoke] [--dry-run]
    ragcpt merge --run models/mvp/run_XXXX
    ragcpt serve --run models/mvp/run_XXXX [--port 8000]   |   ragcpt serve --base [--port 8001]
    ragcpt eval [--which tuned|base]
    ragcpt chat --which tuned
"""
from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="Carrier-aware Qwen3-14B SFT pipeline.")


@app.command()
def extract():
    from ragcpt import extract as m
    m.extract()


@app.command("gen-qa")
def gen_qa(limit: int = typer.Option(None)):
    from ragcpt import gen_qa as m
    m.gen_qa(limit=limit)


@app.command()
def filter():
    from ragcpt import filter_qa as m
    m.filter_qa()


@app.command()
def build():
    from ragcpt import build_sft as m
    m.build_sft()


@app.command()
def audit(n: int = 30):
    from ragcpt import review as m
    m.audit(n=n)


@app.command()
def curate(k: int = 1):
    from ragcpt import review as m
    m.curate(k_per_section=k)


@app.command()
def train(run_name: str = "mvp", smoke: bool = False, dry_run: bool = False):
    from ragcpt import train as m
    m.train(run_name=run_name, smoke=smoke, dry_run=dry_run)


@app.command()
def merge(run: str, dry_run: bool = False):
    from ragcpt import merge as m
    m.merge(run_path=run, dry_run=dry_run)


@app.command()
def serve(run: str = typer.Option(None), base: bool = False, port: int = 8000, dry_run: bool = False):
    from ragcpt import serve as m
    m.serve(model_path=(f"{run}/merged" if run else None), base=base, port=port, dry_run=dry_run)


@app.command()
def eval(which: str = "tuned"):
    from ragcpt import evaluate as m
    m.evaluate(which=which)


@app.command()
def chat(which: str = "tuned"):
    """Interactive chat with a served model (manual verification)."""
    from ragcpt.llm_client import chat as do_chat, endpoint_client
    from pathlib import Path
    client, model = endpoint_client(which)
    system = (Path("ragcpt/reasoning_skeleton.md")).read_text()
    history = [{"role": "system", "content": system}]
    typer.echo(f"Chatting with [{which}] {model}. Ctrl-C to quit.")
    while True:
        try:
            user = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        history.append({"role": "user", "content": user})
        out = do_chat(client, model, history, temperature=0.3)
        history.append({"role": "assistant", "content": out})
        typer.echo(f"bot> {out}")


if __name__ == "__main__":
    app()
