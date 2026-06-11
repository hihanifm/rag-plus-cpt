#!/usr/bin/env python3
"""Stage 0 orchestrator — the MVP vertical slice.

Data-prep (runs on the Mac, needs OPENAI_API_KEY):
    python run_mvp.py prep        # extract -> gen_qa -> filter_qa -> build_sft
    python run_mvp.py audit       # spot-check the auto-cleaner (interactive)
    python run_mvp.py curate      # build gold.jsonl with the teacher (interactive)

Train/serve/eval (run on the Lambda H100 box, needs requirements-train.txt + GPU):
    python run_mvp.py train --smoke
    python run_mvp.py train
    python run_mvp.py serve-base  # serve untuned Qwen for baseline   (separate terminal)
    python run_mvp.py serve --run models/mvp/run_XXue           # serve tuned (separate terminal)
    python run_mvp.py eval --which base
    python run_mvp.py eval --which tuned

See README.md for the full morning checklist.
"""
from __future__ import annotations

import argparse
import sys

from ragcpt import build_sft, extract, filter_qa, gen_qa, review


def cmd_prep(args):
    extract.extract()
    gen_qa.gen_qa(limit=args.limit)
    filter_qa.filter_qa()
    build_sft.build_sft()
    print("\n[prep] done. Next: `python run_mvp.py audit` then `curate`, then train on the GPU box.")


def cmd_audit(args):
    review.audit(n=args.n)


def cmd_curate(args):
    review.curate(k_per_section=args.k)


def cmd_train(args):
    from ragcpt import train
    train.train(run_name=args.run_name, smoke=args.smoke, dry_run=args.dry_run)


def cmd_serve(args):
    from ragcpt import serve
    serve.serve(model_path=str(args.run) + "/merged" if args.run else None,
                port=args.port, base=False, dry_run=args.dry_run)


def cmd_serve_base(args):
    from ragcpt import serve
    serve.serve(base=True, port=args.port, dry_run=args.dry_run)


def cmd_eval(args):
    from ragcpt import evaluate
    evaluate.evaluate(which=args.which)


def main():
    p = argparse.ArgumentParser(description="Stage 0 MVP orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prep"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(fn=cmd_prep)
    sp = sub.add_parser("audit"); sp.add_argument("--n", type=int, default=30); sp.set_defaults(fn=cmd_audit)
    sp = sub.add_parser("curate"); sp.add_argument("--k", type=int, default=1); sp.set_defaults(fn=cmd_curate)
    sp = sub.add_parser("train"); sp.add_argument("--run-name", default="mvp"); sp.add_argument("--smoke", action="store_true"); sp.add_argument("--dry-run", action="store_true"); sp.set_defaults(fn=cmd_train)
    sp = sub.add_parser("serve"); sp.add_argument("--run", required=True); sp.add_argument("--port", type=int, default=8000); sp.add_argument("--dry-run", action="store_true"); sp.set_defaults(fn=cmd_serve)
    sp = sub.add_parser("serve-base"); sp.add_argument("--port", type=int, default=8001); sp.add_argument("--dry-run", action="store_true"); sp.set_defaults(fn=cmd_serve_base)
    sp = sub.add_parser("eval"); sp.add_argument("--which", choices=["base", "tuned"], default="tuned"); sp.set_defaults(fn=cmd_eval)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
