"""Public-only command line interface for StateReturn v1.0."""

from __future__ import annotations

import argparse
import json
import sys

from .text import PublicProtocolError, text_release_fingerprint
from .visual import VisualPublicError, run_visual_statecommit, visual_release_fingerprint
from .next_frame import build_public_next_frame_jobs, validate_next_frame_predictions


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="statereturn-public",
        description="Public-input validation and StateCommit inference for StateReturn v1.0. No evaluator assets are loaded.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    text_verify = commands.add_parser("verify-text", help="validate public Text records without opening labels")
    text_verify.add_argument("--dataset-root", required=True)
    visual_verify = commands.add_parser("verify-visual", help="validate public Visual inputs and public FSEM scopes")
    visual_verify.add_argument("--dataset-root", required=True)
    visual_run = commands.add_parser("run-visual-statecommit", help="run the canonical Visual Fast Path from public Grounder facts")
    visual_run.add_argument("--dataset-root", required=True)
    visual_run.add_argument("--split", choices=("train", "validation", "test"), required=True)
    visual_run.add_argument("--facts", required=True, help="JSONL records: stream_id/frame_index/raw_output")
    visual_run.add_argument("--output-root", required=True)
    jobs = commands.add_parser("build-next-frame-jobs", help="materialize public causal renderer inputs without targets")
    jobs.add_argument("--dataset-root", required=True)
    jobs.add_argument("--split", choices=("train", "validation", "test"), required=True)
    jobs.add_argument("--output", required=True)
    validate = commands.add_parser("validate-next-frame-predictions", help="validate a public renderer output file without scoring")
    validate.add_argument("--jobs", required=True)
    validate.add_argument("--predictions", required=True)
    validate.add_argument("--output-root", required=True)
    return root


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "verify-text":
            result = text_release_fingerprint(args.dataset_root)
        elif args.command == "verify-visual":
            result = visual_release_fingerprint(args.dataset_root)
        elif args.command == "run-visual-statecommit":
            result = run_visual_statecommit(args.dataset_root, split=args.split, facts_path=args.facts, output_root=args.output_root)
        elif args.command == "build-next-frame-jobs":
            result = build_public_next_frame_jobs(args.dataset_root, split=args.split, output_path=args.output)
        else:
            result = validate_next_frame_predictions(args.predictions, jobs_path=args.jobs, output_root=args.output_root)
    except (PublicProtocolError, VisualPublicError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
